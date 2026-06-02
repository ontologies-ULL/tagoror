"""
Unit tests for ValidationStrategy (base) and PureLLMStrategy
==============================================================

Models used (from validation/models.py):
  - ExcecutionMetrics(duration_ms, cost, tokens_consumed)  [frozen dataclass]
  - TaskStatus(SUCCESS, FAILURE, SKIPPED, PARTIAL_SUCCESS) [Enum]
  - TaskOutcome(task_id, status, findings, metrics, rawResponse) [frozen dataclass]
  - ExecutionSummary(individual_id, timestamp, results,
                     total_metrics, system_summary)         [frozen dataclass]
  - PromptRegistry(loaded_prompts)                         [frozen dataclass]

ValidationStrategy (base_entity_validation.py)
-----------------------------------------------
Covers:
  - Cannot instantiate ValidationStrategy directly (abstract)
  - Subclass without evaluate() raises TypeError
  - Valid subclass instantiates correctly
  - evaluate() is declared as async

PureLLMStrategy (pure_llm_strategy.py)
---------------------------------------
Covers:
  - __init__: llm_client and context stored, inherits ValidationStrategy
  - evaluate() single prompt: ExecutionSummary returned, individual_id,
      one TaskOutcome, task_id, findings, rawResponse, metrics, timestamp,
      system_summary
  - evaluate() multiple prompts: one TaskOutcome per prompt, llm_client
      called once per prompt, metrics summed, all task_ids present,
      system_summary mentions correct count
  - evaluate() empty registry: empty results, zero metrics, client not called
  - evaluate() error propagation: exception propagates, stops on first failure
  - _build_payload: system_prompt, user_prompt, json_mode=False
  - _parse_response: task_id, findings, rawResponse, status=SUCCESS,
      per-outcome metrics match LLMResponse values
  - _build_summary: mentions task count, non-empty, zero for empty list

Assumption documented: metrics aggregation = sum across all LLMResponse values.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from core.models import (
    ExcecutionMetrics,
    TaskOutcome,
    TaskStatus,
    ExecutionSummary,
    PromptRegistry,
)
from llm.models import LLMPayload, LLMResponse


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_llm_response(
    raw_content="The entity looks correct.",
    tokens=100,
    duration_ms=200,
    cost=0.01,
) -> LLMResponse:
    return LLMResponse(
        raw_content=raw_content,
        tokens_consumed=tokens,
        duration_ms=duration_ms,
        cost=cost,
    )


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.query = AsyncMock()
    return client


@pytest.fixture
def single_prompt_registry():
    return PromptRegistry(
        loaded_prompts={"task_syntax": "Check the syntax of this OWL entity:"}
    )


@pytest.fixture
def multi_prompt_registry():
    return PromptRegistry(
        loaded_prompts={
            "task_syntax":    "Check the syntax of this OWL entity:",
            "task_semantics": "Check the semantics of this OWL entity:",
            "task_structure": "Check the structure of this OWL entity:",
        }
    )


@pytest.fixture
def empty_registry():
    return PromptRegistry(loaded_prompts={})


# ---------------------------------------------------------------------------
# Tests: ValidationStrategy abstract contract
# ---------------------------------------------------------------------------

class TestValidationStrategyAbstractContract:

    def test_cannot_instantiate_directly(self):
        """Instantiating ValidationStrategy without evaluate() must raise TypeError."""
        from entity_validation.base_entity_validation import ValidationStrategy
        with pytest.raises(TypeError):
            ValidationStrategy()

    def test_subclass_without_evaluate_raises_type_error(self):
        """A subclass that omits evaluate() must raise TypeError on instantiation."""
        from entity_validation.base_entity_validation import ValidationStrategy

        class IncompleteStrategy(ValidationStrategy):
            pass

        with pytest.raises(TypeError):
            IncompleteStrategy()

    def test_valid_subclass_instantiates_correctly(self):
        """A subclass that implements evaluate() must instantiate without errors."""
        from entity_validation.base_entity_validation import ValidationStrategy

        class MinimalStrategy(ValidationStrategy):
            async def evaluate(self, owl_entity: str):
                return None

        assert MinimalStrategy() is not None

    def test_evaluate_is_async(self):
        """evaluate must be declared as a coroutine function."""
        import inspect
        from entity_validation.base_entity_validation import ValidationStrategy
        assert inspect.iscoroutinefunction(ValidationStrategy.evaluate)


# ---------------------------------------------------------------------------
# Tests: PureLLMStrategy.__init__
# ---------------------------------------------------------------------------

class TestPureLLMStrategyInit:

    def test_llm_client_stored(self, mock_llm_client, single_prompt_registry):
        from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy
        s = PureLLMStrategy(llm_client=mock_llm_client, context=single_prompt_registry)
        assert s._llm_client is mock_llm_client

    def test_context_stored(self, mock_llm_client, single_prompt_registry):
        from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy
        s = PureLLMStrategy(llm_client=mock_llm_client, context=single_prompt_registry)
        assert s._context is single_prompt_registry

    def test_inherits_from_validation_strategy(self, mock_llm_client, single_prompt_registry):
        from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy
        from entity_validation.base_entity_validation import ValidationStrategy
        s = PureLLMStrategy(llm_client=mock_llm_client, context=single_prompt_registry)
        assert isinstance(s, ValidationStrategy)


# ---------------------------------------------------------------------------
# Tests: evaluate() — single prompt happy path
# ---------------------------------------------------------------------------

class TestEvaluateSinglePrompt:

    @pytest.fixture
    def strategy(self, mock_llm_client, single_prompt_registry):
        from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy
        return PureLLMStrategy(llm_client=mock_llm_client, context=single_prompt_registry)

    @pytest.mark.asyncio
    async def test_returns_execution_summary(self, strategy, mock_llm_client):
        """evaluate() must return an ExecutionSummary instance."""
        mock_llm_client.query.return_value = make_llm_response()
        result = await strategy.evaluate("owl:Entity1")
        assert isinstance(result, ExecutionSummary)

    @pytest.mark.asyncio
    async def test_individual_id_matches_owl_entity(self, strategy, mock_llm_client):
        """ExecutionSummary.individual_id must equal the owl_entity argument."""
        mock_llm_client.query.return_value = make_llm_response()
        report = await strategy.evaluate("owl:Entity1")
        assert report.individual_id == "owl:Entity1"

    @pytest.mark.asyncio
    async def test_results_has_one_task_outcome(self, strategy, mock_llm_client):
        """With one prompt, results must contain exactly one TaskOutcome."""
        mock_llm_client.query.return_value = make_llm_response()
        report = await strategy.evaluate("owl:Entity1")
        assert len(report.results) == 1
        assert isinstance(report.results[0], TaskOutcome)

    @pytest.mark.asyncio
    async def test_task_outcome_task_id_matches_prompt_key(self, strategy, mock_llm_client):
        """TaskOutcome.task_id must match the prompt key from the registry."""
        mock_llm_client.query.return_value = make_llm_response()
        report = await strategy.evaluate("owl:Entity1")
        assert report.results[0].task_id == "task_syntax"

    @pytest.mark.asyncio
    async def test_task_outcome_findings_contain_raw_content(self, strategy, mock_llm_client):
        """TaskOutcome.findings must contain the LLM raw_content."""
        mock_llm_client.query.return_value = make_llm_response(raw_content="Looks valid.")
        report = await strategy.evaluate("owl:Entity1")
        assert "Looks valid." in report.results[0].findings

    @pytest.mark.asyncio
    async def test_task_outcome_raw_response_matches_content(self, strategy, mock_llm_client):
        """TaskOutcome.rawResponse must equal the LLM raw_content."""
        mock_llm_client.query.return_value = make_llm_response(raw_content="Raw text here.")
        report = await strategy.evaluate("owl:Entity1")
        assert report.results[0].rawResponse == "Raw text here."

    @pytest.mark.asyncio
    async def test_task_outcome_status_is_success(self, strategy, mock_llm_client):
        """TaskOutcome.status must be TaskStatus.SUCCESS by default."""
        mock_llm_client.query.return_value = make_llm_response()
        report = await strategy.evaluate("owl:Entity1")
        assert report.results[0].status == TaskStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_task_outcome_metrics_match_llm_response(self, strategy, mock_llm_client):
        """TaskOutcome.metrics must reflect the individual LLMResponse values."""
        mock_llm_client.query.return_value = make_llm_response(tokens=50, duration_ms=120, cost=0.005)
        report = await strategy.evaluate("owl:Entity1")
        m = report.results[0].metrics
        assert m.tokens_consumed == 50
        assert m.duration_ms     == 120
        assert m.cost            == pytest.approx(0.005)

    @pytest.mark.asyncio
    async def test_total_metrics_tokens_equal_response_tokens(self, strategy, mock_llm_client):
        """total_metrics.tokens_consumed must equal the single LLMResponse tokens."""
        mock_llm_client.query.return_value = make_llm_response(tokens=150)
        report = await strategy.evaluate("owl:Entity1")
        assert report.total_metrics.tokens_consumed == 150

    @pytest.mark.asyncio
    async def test_total_metrics_cost_equals_response_cost(self, strategy, mock_llm_client):
        """total_metrics.cost must equal the single LLMResponse cost."""
        mock_llm_client.query.return_value = make_llm_response(cost=0.05)
        report = await strategy.evaluate("owl:Entity1")
        assert report.total_metrics.cost == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_total_metrics_duration_equals_response_duration(self, strategy, mock_llm_client):
        """total_metrics.duration_ms must equal the single LLMResponse duration."""
        mock_llm_client.query.return_value = make_llm_response(duration_ms=300)
        report = await strategy.evaluate("owl:Entity1")
        assert report.total_metrics.duration_ms == 300

    @pytest.mark.asyncio
    async def test_timestamp_is_valid_iso_string(self, strategy, mock_llm_client):
        """ExecutionSummary.timestamp must be a parseable ISO datetime string."""
        mock_llm_client.query.return_value = make_llm_response()
        report = await strategy.evaluate("owl:Entity1")
        assert isinstance(report.timestamp, str)
        datetime.fromisoformat(report.timestamp)  # raises if invalid

    @pytest.mark.asyncio
    async def test_system_summary_mentions_task_count(self, strategy, mock_llm_client):
        """system_summary must mention 1 task evaluated."""
        mock_llm_client.query.return_value = make_llm_response()
        report = await strategy.evaluate("owl:Entity1")
        assert "1" in report.system_summary


# ---------------------------------------------------------------------------
# Tests: evaluate() — multiple prompts
# ---------------------------------------------------------------------------

class TestEvaluateMultiplePrompts:

    @pytest.fixture
    def strategy(self, mock_llm_client, multi_prompt_registry):
        from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy
        return PureLLMStrategy(llm_client=mock_llm_client, context=multi_prompt_registry)

    @pytest.mark.asyncio
    async def test_one_task_outcome_per_prompt(self, strategy, mock_llm_client):
        """With 3 prompts, results must contain exactly 3 TaskOutcomes."""
        mock_llm_client.query.return_value = make_llm_response()
        report = await strategy.evaluate("owl:Entity1")
        assert len(report.results) == 3

    @pytest.mark.asyncio
    async def test_llm_client_called_once_per_prompt(self, strategy, mock_llm_client):
        """llm_client.query must be called exactly once per prompt."""
        mock_llm_client.query.return_value = make_llm_response()
        await strategy.evaluate("owl:Entity1")
        assert mock_llm_client.query.call_count == 3

    @pytest.mark.asyncio
    async def test_total_metrics_tokens_summed(self, strategy, mock_llm_client):
        """
        total_metrics.tokens_consumed must be the sum across all LLMResponses.
        Assumption: aggregation = sum (update if design changes).
        """
        mock_llm_client.query.side_effect = [
            make_llm_response(tokens=100),
            make_llm_response(tokens=200),
            make_llm_response(tokens=300),
        ]
        report = await strategy.evaluate("owl:Entity1")
        assert report.total_metrics.tokens_consumed == 600

    @pytest.mark.asyncio
    async def test_total_metrics_cost_summed(self, strategy, mock_llm_client):
        """total_metrics.cost must be the sum across all LLMResponses."""
        mock_llm_client.query.side_effect = [
            make_llm_response(cost=0.01),
            make_llm_response(cost=0.02),
            make_llm_response(cost=0.03),
        ]
        report = await strategy.evaluate("owl:Entity1")
        assert report.total_metrics.cost == pytest.approx(0.06)

    @pytest.mark.asyncio
    async def test_total_metrics_duration_summed(self, strategy, mock_llm_client):
        """total_metrics.duration_ms must be the sum across all LLMResponses."""
        mock_llm_client.query.side_effect = [
            make_llm_response(duration_ms=100),
            make_llm_response(duration_ms=200),
            make_llm_response(duration_ms=150),
        ]
        report = await strategy.evaluate("owl:Entity1")
        assert report.total_metrics.duration_ms == 450

    @pytest.mark.asyncio
    async def test_all_task_ids_present_in_results(self, strategy, mock_llm_client):
        """Each prompt key must appear as a task_id in the results list."""
        mock_llm_client.query.return_value = make_llm_response()
        report = await strategy.evaluate("owl:Entity1")
        task_ids = {o.task_id for o in report.results}
        assert task_ids == {"task_syntax", "task_semantics", "task_structure"}

    @pytest.mark.asyncio
    async def test_system_summary_mentions_correct_count(self, strategy, mock_llm_client):
        """system_summary must mention 3 tasks when 3 prompts are evaluated."""
        mock_llm_client.query.return_value = make_llm_response()
        report = await strategy.evaluate("owl:Entity1")
        assert "3" in report.system_summary


# ---------------------------------------------------------------------------
# Tests: evaluate() — empty registry
# ---------------------------------------------------------------------------

class TestEvaluateEmptyRegistry:

    @pytest.fixture
    def strategy(self, mock_llm_client, empty_registry):
        from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy
        return PureLLMStrategy(llm_client=mock_llm_client, context=empty_registry)

    @pytest.mark.asyncio
    async def test_returns_summary_with_empty_results(self, strategy, mock_llm_client):
        """With no prompts, results must be an empty list."""
        report = await strategy.evaluate("owl:Entity1")
        assert report.results == []

    @pytest.mark.asyncio
    async def test_llm_client_never_called(self, strategy, mock_llm_client):
        """With no prompts, llm_client.query must never be called."""
        await strategy.evaluate("owl:Entity1")
        mock_llm_client.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_total_metrics_are_zero(self, strategy):
        """With no prompts all aggregated metrics must be zero."""
        report = await strategy.evaluate("owl:Entity1")
        assert report.total_metrics.tokens_consumed == 0
        assert report.total_metrics.duration_ms     == 0
        assert report.total_metrics.cost            == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_is_successful_with_empty_results(self, strategy):
        """is_successful() must return True when results is empty (vacuous truth)."""
        report = await strategy.evaluate("owl:Entity1")
        assert report.is_successful() is True


# ---------------------------------------------------------------------------
# Tests: evaluate() — error propagation
# ---------------------------------------------------------------------------

class TestEvaluateErrorPropagation:

    @pytest.mark.asyncio
    async def test_propagates_llm_client_exception(self, mock_llm_client, single_prompt_registry):
        """Any exception from llm_client.query must propagate unchanged."""
        from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy
        mock_llm_client.query.side_effect = ConnectionError("LLM unreachable")
        s = PureLLMStrategy(llm_client=mock_llm_client, context=single_prompt_registry)
        with pytest.raises(ConnectionError, match="LLM unreachable"):
            await s.evaluate("owl:Entity1")

    @pytest.mark.asyncio
    async def test_stops_on_first_failure(self, mock_llm_client, multi_prompt_registry):
        """
        If the first LLM call fails, no subsequent calls must be made.
        evaluate() does not implement partial recovery by design.
        """
        from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy
        mock_llm_client.query.side_effect = RuntimeError("fatal")
        s = PureLLMStrategy(llm_client=mock_llm_client, context=multi_prompt_registry)
        with pytest.raises(RuntimeError):
            await s.evaluate("owl:Entity1")
        assert mock_llm_client.query.call_count == 1


# ---------------------------------------------------------------------------
# Tests: ExecutionSummary.is_successful()
# ---------------------------------------------------------------------------

class TestExecutionSummaryIsSuccessful:

    def _make_outcome(self, status: TaskStatus) -> TaskOutcome:
        return TaskOutcome(
            task_id="t",
            status=status,
            findings=[],
            metrics=ExcecutionMetrics(duration_ms=0, cost=0.0, tokens_consumed=0),
            rawResponse="",
        )

    def test_true_when_all_tasks_succeed(self):
        """is_successful() must return True when all TaskOutcomes are SUCCESS."""
        summary = ExecutionSummary(
            individual_id="e",
            timestamp="2024-01-01T00:00:00+00:00",
            results=[
                self._make_outcome(TaskStatus.SUCCESS),
                self._make_outcome(TaskStatus.SUCCESS),
            ],
            total_metrics=ExcecutionMetrics(duration_ms=0, cost=0.0, tokens_consumed=0),
            system_summary="",
        )
        assert summary.is_successful() is True

    def test_false_when_any_task_fails(self):
        """is_successful() must return False when any TaskOutcome is FAILURE."""
        summary = ExecutionSummary(
            individual_id="e",
            timestamp="2024-01-01T00:00:00+00:00",
            results=[
                self._make_outcome(TaskStatus.SUCCESS),
                self._make_outcome(TaskStatus.FAILURE),
            ],
            total_metrics=ExcecutionMetrics(duration_ms=0, cost=0.0, tokens_consumed=0),
            system_summary="",
        )
        assert summary.is_successful() is False

    def test_false_when_task_is_partial_success(self):
        """is_successful() must return False for PARTIAL_SUCCESS outcomes."""
        summary = ExecutionSummary(
            individual_id="e",
            timestamp="2024-01-01T00:00:00+00:00",
            results=[self._make_outcome(TaskStatus.PARTIAL_SUCCESS)],
            total_metrics=ExcecutionMetrics(duration_ms=0, cost=0.0, tokens_consumed=0),
            system_summary="",
        )
        assert summary.is_successful() is False

    def test_false_when_task_is_skipped(self):
        """is_successful() must return False when any task is SKIPPED."""
        summary = ExecutionSummary(
            individual_id="e",
            timestamp="2024-01-01T00:00:00+00:00",
            results=[self._make_outcome(TaskStatus.SKIPPED)],
            total_metrics=ExcecutionMetrics(duration_ms=0, cost=0.0, tokens_consumed=0),
            system_summary="",
        )
        assert summary.is_successful() is False

    def test_true_when_results_empty(self):
        """is_successful() must return True for an empty results list (vacuous truth)."""
        summary = ExecutionSummary(
            individual_id="e",
            timestamp="2024-01-01T00:00:00+00:00",
            results=[],
            total_metrics=ExcecutionMetrics(duration_ms=0, cost=0.0, tokens_consumed=0),
            system_summary="",
        )
        assert summary.is_successful() is True


# ---------------------------------------------------------------------------
# Tests: _build_payload
# ---------------------------------------------------------------------------

class TestBuildPayload:

    @pytest.fixture
    def strategy(self, mock_llm_client, single_prompt_registry):
        from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy
        return PureLLMStrategy(llm_client=mock_llm_client, context=single_prompt_registry)

    def test_system_prompt_equals_prompt_template(self, strategy):
        payload = strategy._build_payload("Check this:", "owl:Entity1")
        assert payload.system_prompt == "Check this:"

    def test_user_prompt_equals_owl_entity(self, strategy):
        payload = strategy._build_payload("Check this:", "owl:Entity1")
        assert payload.user_prompt == "owl:Entity1"

    def test_json_mode_is_false(self, strategy):
        """json_mode must be False since the LLM returns free-text responses."""
        payload = strategy._build_payload("Check this:", "owl:Entity1")
        assert payload.json_mode is False


# ---------------------------------------------------------------------------
# Tests: _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:

    @pytest.fixture
    def strategy(self, mock_llm_client, single_prompt_registry):
        from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy
        return PureLLMStrategy(llm_client=mock_llm_client, context=single_prompt_registry)

    def test_task_id_stored_correctly(self, strategy):
        response = make_llm_response(raw_content="Valid.")
        outcome  = strategy._parse_response("task_syntax", response)
        assert outcome.task_id == "task_syntax"

    def test_raw_content_stored_in_findings(self, strategy):
        response = make_llm_response(raw_content="The entity is valid.")
        outcome  = strategy._parse_response("task_syntax", response)
        assert "The entity is valid." in outcome.findings

    def test_raw_response_matches_content(self, strategy):
        response = make_llm_response(raw_content="Raw text.")
        outcome  = strategy._parse_response("task_syntax", response)
        assert outcome.rawResponse == "Raw text."

    def test_status_is_success_by_default(self, strategy):
        outcome = strategy._parse_response("task_syntax", make_llm_response())
        assert outcome.status == TaskStatus.SUCCESS

    def test_per_outcome_metrics_match_response(self, strategy):
        """TaskOutcome.metrics must reflect the individual LLMResponse values."""
        response = make_llm_response(tokens=77, duration_ms=333, cost=0.007)
        outcome  = strategy._parse_response("task_syntax", response)
        assert outcome.metrics.tokens_consumed == 77
        assert outcome.metrics.duration_ms     == 333
        assert outcome.metrics.cost            == pytest.approx(0.007)

    def test_findings_is_a_list(self, strategy):
        outcome = strategy._parse_response("task_syntax", make_llm_response())
        assert isinstance(outcome.findings, list)


# ---------------------------------------------------------------------------
# Tests: _build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:

    @pytest.fixture
    def strategy(self, mock_llm_client, single_prompt_registry):
        from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy
        return PureLLMStrategy(llm_client=mock_llm_client, context=single_prompt_registry)

    def test_mentions_task_count(self, strategy):
        outcomes = [MagicMock(), MagicMock(), MagicMock()]
        assert "3" in strategy._build_summary(outcomes)

    def test_is_non_empty_string(self, strategy):
        assert len(strategy._build_summary([])) > 0

    def test_mentions_zero_for_empty_list(self, strategy):
        assert "0" in strategy._build_summary([])