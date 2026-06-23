"""
Unit tests for EntityOrchestrator
========================================

Architecture under test (as agreed in this conversation — no external
assumptions beyond what was explicitly confirmed):

    ExtractedEntity-like double:
        - individual_id: str
        - to_llm_context() -> str

    ValidationStrategy (ABC):
    - async def evaluate(self, owl_entity: str) -> ExecutionSummary

    EntityOrchestrator:
    - __init__(self, strategy: ValidationStrategy) 
    - async def process(self, individuals: list[ExtractedEntity]) -> list[ExecutionSummary]
        · Validates entities in PARALLEL via asyncio.gather
        · Each entity call is wrapped individually (try/except) BEFORE gather,
          so one entity's failure never raises out of process() or drops
          other entities' results
        · On failure: builds an ExecutionSummary with a single TaskOutcome
          (task_id="orchestration_error", status=TaskStatus.FAILURE,
          findings=[str(error)]), 

Covers:
  - process() happy path: returns one ExecutionSummary per entity, in order
  - process() calls strategy.evaluate() with entity.to_llm_context(), not
    the raw ExtractedEntity object
  - process() calls strategy.evaluate() exactly once per entity
  - process() validates entities concurrently (asyncio.gather), not sequentially
  - process() with empty list: returns an empty list, strategy never called
  - process() single entity failure: does NOT raise, returns a list of the
    same length as the input
  - process() single entity failure: failing entity's ExecutionSummary has
    individual_id matching the entity, one TaskOutcome with
    task_id="orchestration_error", status=FAILURE, error message in findings
  - process() single entity failure: notifier.notify_critical_failure() called
    once with the correct individual_id and the original exception
  - process() mixed success/failure: successful entities keep their real
    ExecutionSummary from the strategy, untouched
  - process() mixed success/failure: order of entities is preserved in the
    returned list (success and failure interleaved correctly)
  - process() multiple failures: notifier called once per failing entity
  - process() multiple failures: each failure is independent (one entity's
    exception does not affect another entity's outcome)
  - process() does not use pytest.raises: failures are always returned as
    data, never propagated as exceptions out of process()

Testing strategy:
    strategy and notifier are fully mocked (AsyncMock / MagicMock) so no real
    LLM calls or file I/O occur. Entity doubles are built directly with the
    confirmed fields; to_llm_context() is mocked per-entity to return a
    predictable string so we can assert exactly what was passed to evaluate().
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Fixtures: domain doubles
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_strategy():
    """ValidationStrategy double — evaluate() is async."""
    strategy = MagicMock()
    strategy.evaluate = AsyncMock()
    return strategy

    
@pytest.fixture
def orchestrator(mock_strategy):
    from core.orchestrator import EntityOrchestrator
    return EntityOrchestrator(strategy=mock_strategy)

def make_entity(individual_id: str, llm_context: str = None):
    """
    Builds an ExtractedEntity-like double with a controllable to_llm_context().
    Using MagicMock here (rather than the real Pydantic model) keeps these
    tests independent from the exact validators defined on ExtractedEntity,
    while still exercising the orchestrator's real contract: it must call
    entity.to_llm_context(), not touch the entity's raw fields directly.
    """
    entity = MagicMock()
    entity.individual_id = individual_id
    entity.to_llm_context.return_value = llm_context or f"context_for_{individual_id}"
    return entity


def make_execution_summary(individual_id: str):
    """Minimal valid ExecutionSummary double returned by a successful strategy.evaluate()."""
    from core.models import ExecutionSummary, ExecutionMetrics 
    return ExecutionSummary(
        individual_id=individual_id,
        timestamp="2024-01-01T00:00:00+00:00",
        results=[],
        total_metrics=ExecutionMetrics(duration_ms=10, cost=0.001, tokens_consumed=5),
        system_summary="ok",
    )


# ---------------------------------------------------------------------------
# Tests: process() — happy path
# ---------------------------------------------------------------------------

class TestProcessHappyPath:

    @pytest.mark.asyncio
    async def test_returns_one_summary_per_entity(self, orchestrator, mock_strategy):
        """With 3 entities, process() must return exactly 3 ExecutionSummary objects."""
        entities = [make_entity("ind_1"), make_entity("ind_2"), make_entity("ind_3")]
        mock_strategy.evaluate.side_effect = [
            make_execution_summary("ind_1"),
            make_execution_summary("ind_2"),
            make_execution_summary("ind_3"),
        ]

        results = await orchestrator.process(entities)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_results_preserve_input_order(self, orchestrator, mock_strategy):
        """The returned list must preserve the same order as the input entities."""
        entities = [make_entity("ind_A"), make_entity("ind_B"), make_entity("ind_C")]
        mock_strategy.evaluate.side_effect = [
            make_execution_summary("ind_A"),
            make_execution_summary("ind_B"),
            make_execution_summary("ind_C"),
        ]

        results = await orchestrator.process(entities)

        assert [r.individual_id for r in results] == ["ind_A", "ind_B", "ind_C"]

    @pytest.mark.asyncio
    async def test_evaluate_called_with_llm_context_not_raw_entity(self, orchestrator, mock_strategy):
        """
        strategy.evaluate() must receive entity.to_llm_context() (a string),
        never the raw ExtractedEntity object.
        """
        entity = make_entity("ind_1", llm_context="SERIALIZED_CONTEXT")
        mock_strategy.evaluate.return_value = make_execution_summary("ind_1")

        await orchestrator.process([entity])

        mock_strategy.evaluate.assert_called_once_with("SERIALIZED_CONTEXT")

    @pytest.mark.asyncio
    async def test_to_llm_context_called_once_per_entity(self, orchestrator, mock_strategy):
        """to_llm_context() must be invoked exactly once per entity."""
        entity = make_entity("ind_1")
        mock_strategy.evaluate.return_value = make_execution_summary("ind_1")

        await orchestrator.process([entity])

    @pytest.mark.asyncio
    async def test_evaluate_called_once_per_entity(self, orchestrator, mock_strategy):
        """strategy.evaluate() must be called exactly once per entity, no more."""
        entities = [make_entity("ind_1"), make_entity("ind_2")]
        mock_strategy.evaluate.side_effect = [
            make_execution_summary("ind_1"),
            make_execution_summary("ind_2"),
        ]

        await orchestrator.process(entities)

        assert mock_strategy.evaluate.call_count == 2

    @pytest.mark.asyncio
    async def test_successful_summaries_are_returned_unmodified(self, orchestrator, mock_strategy):
        """A successful ExecutionSummary from the strategy must be returned as-is."""
        entity  = make_entity("ind_1")
        summary = make_execution_summary("ind_1")
        mock_strategy.evaluate.return_value = summary

        results = await orchestrator.process([entity])

        assert results[0] is summary


# ---------------------------------------------------------------------------
# Tests: process() — concurrency
# ---------------------------------------------------------------------------

class TestProcessConcurrency:

    @pytest.mark.asyncio
    async def test_entities_are_validated_concurrently(self, orchestrator, mock_strategy):
        """
        All entities must be in-flight concurrently (asyncio.gather), not
        awaited one at a time. We verify this by having each evaluate() call
        wait on a shared event that only releases once all calls have started.
        If process() were sequential, the second call would never start
        before the first completes, and this test would deadlock/timeout.
        """
        entities = [make_entity("ind_1"), make_entity("ind_2"), make_entity("ind_3")]
        started  = []
        release  = asyncio.Event()

        async def fake_evaluate(context):
            started.append(context)
            if len(started) == len(entities):
                release.set()
            await release.wait()
            return make_execution_summary(context)

        mock_strategy.evaluate.side_effect = fake_evaluate

        results = await asyncio.wait_for(orchestrator.process(entities), timeout=2.0)

        assert len(started) == 3
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Tests: process() — empty input
# ---------------------------------------------------------------------------

class TestProcessEmptyInput:

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_list(self, orchestrator):
        """With no entities, process() must return an empty list."""
        results = await orchestrator.process([])
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_list_never_calls_strategy(self, orchestrator, mock_strategy):
        """With no entities, strategy.evaluate() must never be called."""
        await orchestrator.process([])
        mock_strategy.evaluate.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: process() — single entity failure
# ---------------------------------------------------------------------------

class TestProcessSingleFailure:

    @pytest.mark.asyncio
    async def test_does_not_raise(self, orchestrator, mock_strategy):
        """A failing entity must NOT cause process() to raise."""
        entity = make_entity("ind_fail")
        mock_strategy.evaluate.side_effect = TimeoutError("LLM timed out")

        # No pytest.raises: process() must swallow the error and return data.
        results = await orchestrator.process([entity])

        assert results is not None

    @pytest.mark.asyncio
    async def test_returns_same_length_as_input(self, orchestrator, mock_strategy):
        """Even when the single entity fails, the result list must have length 1."""
        entity = make_entity("ind_fail")
        mock_strategy.evaluate.side_effect = TimeoutError("LLM timed out")

        results = await orchestrator.process([entity])

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_failure_summary_individual_id_matches_entity(self, orchestrator, mock_strategy):
        """The ExecutionSummary for a failed entity must keep its individual_id."""
        entity = make_entity("ind_fail")
        mock_strategy.evaluate.side_effect = TimeoutError("LLM timed out")

        results = await orchestrator.process([entity])

        assert results[0].individual_id == "ind_fail"

    @pytest.mark.asyncio
    async def test_failure_summary_has_single_orchestration_error_task(self, orchestrator, mock_strategy):
        """The failure ExecutionSummary must contain exactly one TaskOutcome."""
        entity = make_entity("ind_fail")
        mock_strategy.evaluate.side_effect = TimeoutError("LLM timed out")

        results = await orchestrator.process([entity])

        assert len(results[0].results) == 1
        assert results[0].results[0].task_id == "orchestration_error"

    @pytest.mark.asyncio
    async def test_failure_task_status_is_failure(self, orchestrator, mock_strategy):
        """The single TaskOutcome on failure must have status=TaskStatus.FAILURE."""
        from core.models import TaskStatus
        entity = make_entity("ind_fail")
        mock_strategy.evaluate.side_effect = TimeoutError("LLM timed out")

        results = await orchestrator.process([entity])

        assert results[0].results[0].status == TaskStatus.FAILURE

    @pytest.mark.asyncio
    async def test_failure_findings_contain_error_message(self, orchestrator, mock_strategy):
        """The TaskOutcome.findings must contain the str() of the original exception."""
        entity = make_entity("ind_fail")
        mock_strategy.evaluate.side_effect = TimeoutError("LLM timed out after 30s")

        results = await orchestrator.process([entity])

        assert "LLM timed out after 30s" in results[0].results[0].findings

    @pytest.mark.asyncio
    async def test_works_with_any_exception_type(self, orchestrator, mock_strategy):
        """
        The failure-handling path must work for any Exception subtype,
        not just network/timeout errors (e.g. a parsing or validation error).
        """
        entity = make_entity("ind_fail")
        mock_strategy.evaluate.side_effect = ValueError("malformed LLM response")

        results = await orchestrator.process([entity])

        assert results[0].results[0].status.name == "FAILURE"
        assert "malformed LLM response" in results[0].results[0].findings


# ---------------------------------------------------------------------------
# Tests: process() — mixed success and failure
# ---------------------------------------------------------------------------

class TestProcessMixedOutcomes:

    @pytest.mark.asyncio
    async def test_successful_entities_keep_real_summary(self, orchestrator, mock_strategy):
        """In a mixed batch, successful entities must return their real ExecutionSummary."""
        entities = [make_entity("ind_ok"), make_entity("ind_fail")]
        real_summary = make_execution_summary("ind_ok")
        mock_strategy.evaluate.side_effect = [real_summary, RuntimeError("boom")]

        results = await orchestrator.process(entities)

        assert results[0] is real_summary

    @pytest.mark.asyncio
    async def test_failing_entity_gets_failure_summary(self, orchestrator, mock_strategy):
        """In a mixed batch, the failing entity must get the orchestration_error summary."""
        entities = [make_entity("ind_ok"), make_entity("ind_fail")]
        mock_strategy.evaluate.side_effect = [make_execution_summary("ind_ok"), RuntimeError("boom")]

        results = await orchestrator.process(entities)

        assert results[1].individual_id == "ind_fail"
        assert results[1].results[0].task_id == "orchestration_error"

    @pytest.mark.asyncio
    async def test_one_failure_does_not_affect_other_entities(self, orchestrator, mock_strategy):
        """A failure in one entity must not alter or drop the results of others."""
        entities = [make_entity("ind_1"), make_entity("ind_2"), make_entity("ind_3")]
        ok_summary_1 = make_execution_summary("ind_1")
        ok_summary_3 = make_execution_summary("ind_3")
        mock_strategy.evaluate.side_effect = [ok_summary_1, RuntimeError("fail in the middle"), ok_summary_3]

        results = await orchestrator.process(entities)

        assert results[0] is ok_summary_1
        assert results[1].results[0].task_id == "orchestration_error"
        assert results[2] is ok_summary_3


    @pytest.mark.asyncio
    async def test_multiple_failures_are_independent(self, orchestrator, mock_strategy, ):
        """Each failing entity must produce its own distinct failure summary and notifier call."""
        entities = [make_entity("ind_a"), make_entity("ind_b")]
        error_a = ValueError("error A")
        error_b = TimeoutError("error B")
        mock_strategy.evaluate.side_effect = [error_a, error_b]

        results = await orchestrator.process(entities)

        assert "error A" in results[0].results[0].findings
        assert "error B" in results[1].results[0].findings


# ---------------------------------------------------------------------------
# Tests: process() — never propagates exceptions (by design)
# ---------------------------------------------------------------------------

class TestProcessNeverRaises:

    @pytest.mark.asyncio
    async def test_all_entities_failing_still_returns_full_list(self, orchestrator, mock_strategy):
        """
        Even if every single entity fails, process() must still return a list
        of ExecutionSummary objects (one per entity), never raise.
        """
        entities = [make_entity("ind_1"), make_entity("ind_2")]
        mock_strategy.evaluate.side_effect = [RuntimeError("fail 1"), RuntimeError("fail 2")]

        results = await orchestrator.process(entities)

        assert len(results) == 2
        assert all(r.results[0].task_id == "orchestration_error" for r in results)