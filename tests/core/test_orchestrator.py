"""
Extreme Unit tests for the EntityOrchestrator class.

These tests verify the core responsibilities of the orchestrator under extreme conditions:
1. Exact call count delegation (verifying the strategy is called the exact expected number of times).
2. Unbounded concurrency handling (validating that 10,000+ entities process in parallel).
3. Absolute error isolation (validating that chaotic, random exceptions do not crash the batch).
"""

import asyncio
import random
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

# Import domain models and classes
from core.models import ExecutionSummary, TaskOutcome, TaskStatus, ExecutionMetrics
from core.pipeline.orchestrator import EntityOrchestrator
from core.pipeline.evaluation.entity_auditor import EntityAuditor


# ============================================================================
# Helpers & Mocks
# ============================================================================

def make_mock_thing(individual_id: str) -> MagicMock:
    """
    Creates a MagicMock simulating an owlready2 Thing.
    Provides the individual_id attribute expected by the orchestrator.
    """
    mock_thing = MagicMock()
    mock_thing.individual_id = individual_id
    mock_thing.name = individual_id
    return mock_thing


def make_success_summary(individual_id: str) -> ExecutionSummary:
    """
    Creates a standard successful ExecutionSummary to be returned by our mock auditor.
    """
    return ExecutionSummary(
        individual_id=individual_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        results=[
            TaskOutcome(task_id="test_task", status=TaskStatus.SUCCESS, findings=["OK"])
        ],
        total_metrics=ExecutionMetrics(duration_ms=10, cost=0.0, tokens_consumed=5),
        system_summary="Success"
    )


# ============================================================================
# Unit Tests
# ============================================================================

class TestEntityOrchestratorExtreme:

    @pytest.mark.asyncio
    async def test_process_empty_list_returns_empty_and_does_not_call_auditor(self):
        """
        Test 1: If the input list is empty, the orchestrator must short-circuit
        and return an empty list immediately without interacting with the auditor.
        """
        mock_auditor = AsyncMock(spec=EntityAuditor)
        mock_ontology = MagicMock()
        
        orchestrator = EntityOrchestrator(strategy=mock_auditor)
        results = await orchestrator.process([], mock_ontology)

        assert results == []
        # Explicitly verify the strategy was never triggered
        mock_auditor.run.assert_not_called()
        assert mock_auditor.run.call_count == 0


    @pytest.mark.asyncio
    async def test_exact_call_count_and_arguments_verification(self):
        """
        Test 2: Verifies that if we pass exactly 3 entities, the strategy is called 
        exactly 3 times with the correct respective arguments.
        """
        mock_auditor = AsyncMock(spec=EntityAuditor)
        mock_ontology = MagicMock()
        
        entities = [
            make_mock_thing("Entity_1"),
            make_mock_thing("Entity_2"),
            make_mock_thing("Entity_3")
        ]

        async def mock_run(entity, base_ontology):
            return make_success_summary(entity.individual_id)
            
        mock_auditor.run.side_effect = mock_run

        orchestrator = EntityOrchestrator(strategy=mock_auditor)
        results = await orchestrator.process(entities, mock_ontology)

        assert len(results) == 3
        
        # EXACT CALL COUNT ASSERTION
        assert mock_auditor.run.call_count == 3
        
        # ARGUMENT VERIFICATION: Inspect the exact arguments passed to the mock
        call_args = mock_auditor.run.call_args_list
        
        # Ensure the auditor received the correct entity and the correct base ontology in every call
        assert call_args[0].args[0].individual_id == "Entity_1"
        assert call_args[0].args[1] == mock_ontology
        
        assert call_args[1].args[0].individual_id == "Entity_2"
        assert call_args[1].args[1] == mock_ontology
        
        assert call_args[2].args[0].individual_id == "Entity_3"
        assert call_args[2].args[1] == mock_ontology


    @pytest.mark.asyncio
    async def test_massive_unbounded_concurrency_stress_test(self):
        """
        Test 3: STRESS TEST. Proves mathematically that process() executes tasks in parallel 
        via asyncio.gather. We send 10,000 mocked entities.
        
        If it was sequential, 10,000 * 0.001s = 10 seconds.
        If parallel, it will take less than 0.5 seconds to process all of them.
        """
        mock_auditor = AsyncMock(spec=EntityAuditor)
        mock_ontology = MagicMock()
        
        MASSIVE_AMOUNT = 10000
        entities = [make_mock_thing(f"Entity_{i}") for i in range(MASSIVE_AMOUNT)]
        
        async def fast_mock_run(entity, base_ontology):
            await asyncio.sleep(0.001)
            return make_success_summary(entity.individual_id)
            
        mock_auditor.run.side_effect = fast_mock_run
        
        orchestrator = EntityOrchestrator(strategy=mock_auditor)
        
        start_time = asyncio.get_event_loop().time()
        results = await orchestrator.process(entities, mock_ontology)
        end_time = asyncio.get_event_loop().time()
        
        elapsed_time = end_time - start_time
        
        assert len(results) == MASSIVE_AMOUNT
        assert mock_auditor.run.call_count == MASSIVE_AMOUNT
        # Proof of parallel execution
        assert elapsed_time < 1.0, f"Execution took {elapsed_time}s. It is blocking/sequential!"


    @pytest.mark.asyncio
    async def test_chaotic_mixed_failures_isolation(self):
        """
        Test 4: CHAOS TEST. We send 100 entities. We program the mock to randomly crash 
        for 50% of them using completely different fatal exceptions.
        
        The orchestrator MUST survive, process all 100, return exactly 100 summaries, 
        and encapsulate the 50 failures safely.
        """
        mock_auditor = AsyncMock(spec=EntityAuditor)
        mock_ontology = MagicMock()
        
        TOTAL_ENTITIES = 100
        entities = [make_mock_thing(f"Entity_{i}") for i in range(TOTAL_ENTITIES)]

        async def chaotic_mock_run(entity, base_ontology):
            entity_number = int(entity.individual_id.split("_")[1])
            if entity_number % 2 == 0:
                exceptions = [
                    ValueError("Malformed OWL structure"),
                    TypeError("NoneType object is not subscriptable"),
                    KeyError("Missing configuration key")
                ]
                raise random.choice(exceptions)
            
            return make_success_summary(entity.individual_id)
            
        mock_auditor.run.side_effect = chaotic_mock_run

        orchestrator = EntityOrchestrator(strategy=mock_auditor)
        results = await orchestrator.process(entities, mock_ontology)

        assert len(results) == TOTAL_ENTITIES
        assert mock_auditor.run.call_count == TOTAL_ENTITIES
        
        success_count = sum(1 for r in results if r.is_successful())
        failure_count = sum(1 for r in results if not r.is_successful())
        
        # Exactly 50 should succeed and 50 should be isolated orchestration errors
        assert success_count == 50
        assert failure_count == 50


    @pytest.mark.asyncio
    async def test_all_entities_fail_returns_all_orchestration_errors(self):
        """
        Test 5: If every single call to the auditor crashes, the orchestrator
        must not bubble up the exception. It must return a full list of failure summaries.
        """
        mock_auditor = AsyncMock(spec=EntityAuditor)
        mock_ontology = MagicMock()
        
        entities = [make_mock_thing(f"Entity_{i}") for i in range(10)]

        # Auditor always fails
        mock_auditor.run.side_effect = Exception("General system failure")

        orchestrator = EntityOrchestrator(strategy=mock_auditor)
        results = await orchestrator.process(entities, mock_ontology)

        assert len(results) == 10
        assert mock_auditor.run.call_count == 10
        
        for summary in results:
            assert summary.is_successful() is False
            assert summary.results[0].task_id == "orchestration_error"
            assert summary.total_metrics.duration_ms == 0


    @pytest.mark.asyncio
    async def test_auditor_timeout_handling(self):
        """
        Test 6: Validates that an asyncio.TimeoutError raised by the strategy 
        is safely caught and wrapped like any other standard exception.
        """
        mock_auditor = AsyncMock(spec=EntityAuditor)
        mock_ontology = MagicMock()
        
        entity = make_mock_thing("Timeout_Entity")

        async def timeout_mock_run(entity, base_ontology):
            raise asyncio.TimeoutError("The LLM took too long to respond")
            
        mock_auditor.run.side_effect = timeout_mock_run

        orchestrator = EntityOrchestrator(strategy=mock_auditor)
        results = await orchestrator.process([entity], mock_ontology)

        assert len(results) == 1
        summary = results[0]
        
        assert summary.is_successful() is False
        assert summary.results[0].task_id == "orchestration_error"
        # FIXED: Check for the actual string message, since str(TimeoutError("...")) 
        # only outputs the message itself, not the class name.
        assert "The LLM took too long to respond" in summary.results[0].findings[0]


    @pytest.mark.asyncio
    async def test_tracing_and_telemetry_on_failure(self):
        """
        Test 7: Ensures that when an entity fails, the OpenTelemetry tracer 
        records the exception and sets the span status to ERROR.
        """
        mock_auditor = AsyncMock(spec=EntityAuditor)
        mock_ontology = MagicMock()
        
        entity = make_mock_thing("Telemetry_Entity")
        simulated_error = ValueError("Database connection lost")
        
        mock_auditor.run.side_effect = simulated_error

        # We inject a mock tracer into the orchestrator
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        # Mock the context manager behavior of start_as_current_span
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        orchestrator = EntityOrchestrator(strategy=mock_auditor)
        orchestrator._tracer = mock_tracer # Override internal tracer

        await orchestrator.process([entity], mock_ontology)

        # Assert telemetry methods were called with the correct error parameters
        mock_span.set_status.assert_called_once()
        mock_span.record_exception.assert_called_once_with(simulated_error)
        
        # Verify stacktrace was recorded
        call_args = mock_span.set_attribute.call_args_list
        attributes_set = [args[0][0] for args in call_args]
        assert "error.stacktrace" in attributes_set


    @pytest.mark.asyncio
    async def test_returned_execution_summaries_are_ordered(self):
        """
        Test 8: asyncio.gather guarantees that the order of the results matches 
        the order of the inputs, regardless of which concurrent task finishes first.
        This test proves the orchestrator honors that guarantee.
        """
        mock_auditor = AsyncMock(spec=EntityAuditor)
        mock_ontology = MagicMock()
        
        # Order matters
        entities = [
            make_mock_thing("Entity_First"),
            make_mock_thing("Entity_Second"),
            make_mock_thing("Entity_Third")
        ]

        async def randomized_delay_mock_run(entity, base_ontology):
            # We make the "First" entity the slowest one, and the "Third" the fastest.
            if entity.individual_id == "Entity_First":
                await asyncio.sleep(0.15)
            elif entity.individual_id == "Entity_Second":
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(0.01)
                
            return make_success_summary(entity.individual_id)
            
        mock_auditor.run.side_effect = randomized_delay_mock_run

        orchestrator = EntityOrchestrator(strategy=mock_auditor)
        results = await orchestrator.process(entities, mock_ontology)

        assert len(results) == 3
        # The result list MUST preserve the original input order, 
        # even though Entity_Third finished processing long before Entity_First.
        assert results[0].individual_id == "Entity_First"
        assert results[1].individual_id == "Entity_Second"
        assert results[2].individual_id == "Entity_Third"