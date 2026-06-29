import asyncio
from datetime import datetime, timezone
from typing import Any

from owlready2 import Thing
import traceback
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from core.models import ExecutionSummary, ExecutionMetrics, TaskOutcome, TaskStatus
from evaluation.entity_auditor import EntityAuditor 

class EntityOrchestrator:
    """
    Orchestrates the parallel validation of extracted ontology entities.
    Delegates validation to a ValidationStrategy and critical errors to a ValidationNotifier.
    """

    def __init__(self, strategy: EntityAuditor) -> None:
        self.strategy = strategy
        self._tracer = trace.get_tracer(__name__)

    async def process(self, individuals: list[Thing], base_ontology: list[Thing]) -> list[ExecutionSummary]:
        """
        Validates a list of entities concurrently.
        Failures are encapsulated into fallback ExecutionSummaries; exceptions are never raised.
        """
        if not individuals:
            return []

        tasks = [self._process_single(entity, base_ontology) for entity in individuals]
        return await asyncio.gather(*tasks)

    async def _process_single(self, entity: Thing, base_ontology: list[Thing]) -> ExecutionSummary:
        """
        isolates the validation of a single entity, capturing any exceptions 
        and returning a fallback ExecutionSummary if needed. 
        """
        with self._tracer.start_as_current_span("validation_orchestrator.process_single") as span:
            span.set_attribute("entity.individual_id", entity.individual_id)
            try:
                return await self.strategy.run(entity, base_ontology)

            except Exception as error:
                error_msg = f"Critical failure processing {entity.individual_id}: {str(error)}"
                span.set_status(Status(StatusCode.ERROR, error_msg))
                span.record_exception(error)
                span.set_attribute("error.stacktrace", traceback.format_exc()) 

                return ExecutionSummary(
                    individual_id=entity.individual_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    results=[
                        TaskOutcome(
                            task_id="orchestration_error",
                            status=TaskStatus.FAILURE,
                            findings=[str(error)]
                        )
                    ],
                    total_metrics=ExecutionMetrics(
                        duration_ms=0,
                        cost=0.0,
                        tokens_consumed=0
                    ),
                    system_summary="Validation failed during orchestration."
                )