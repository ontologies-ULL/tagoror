import asyncio
import traceback

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from owlready2 import Thing, Ontology
from datetime import datetime, timezone

from tagoror.core.models import ExecutionSummary, ExecutionMetrics, TaskOutcome, TaskStatus
from tagoror.core.pipeline.evaluation.entity_auditor import EntityAuditor 
from typing import Optional


class EntityOrchestrator:
    """
    Orchestrates the parallel validation of extracted ontology entities.
    Delegates validation to a ValidationStrategy and critical errors to a ValidationNotifier.
    """

    def __init__(self, strategy: EntityAuditor, semaphore: asyncio.Semaphore = None) -> None:
        self.strategy = strategy
        self._tracer = trace.get_tracer(__name__)
        # TODO: change this after we have a better way to handle token input limits. 
        self._semaphore = semaphore or asyncio.Semaphore(2)

    async def process(self, individuals: list[Thing], base_ontology: Ontology) -> list[ExecutionSummary]:
        """
        Validates a list of entities concurrently.
        Failures are encapsulated into fallback ExecutionSummaries; exceptions are never raised.
        """
        if not individuals:
            return []

        cache_name = None
        if hasattr(self.strategy, "setup_cache"):
            with self._tracer.start_as_current_span("validation_orchestrator.setup_cache") as span:
                try:
                    cache_name = await self.strategy.setup_cache(base_ontology, self._tracer)
                    span.set_status(Status(StatusCode.OK))
                except Exception as error:
                    error_msg = f"Critical failure during cache setup: {str(error)}"
                    span.set_status(Status(StatusCode.ERROR, error_msg))
                    span.record_exception(error)
                    span.set_attribute("error.stacktrace", traceback.format_exc())
                    raise RuntimeError("Cache setup failed. Aborting validation.") from error

        try:
            tasks = [self._process_single(entity, base_ontology, cache_name) for entity in individuals]
            return await asyncio.gather(*tasks)
        finally:
            if cache_name and hasattr(self.strategy, "teardown_cache"):
                with self._tracer.start_as_current_span("validation_orchestrator.teardown_cache") as span:
                    try:
                        await self.strategy.teardown_cache(cache_name)
                        span.set_status(Status(StatusCode.OK))
                    except Exception as error:
                        error_msg = f"Critical failure during cache teardown: {str(error)}"
                        span.set_status(Status(StatusCode.ERROR, error_msg))
                        span.record_exception(error)
                        span.set_attribute("error.stacktrace", traceback.format_exc())


    async def _process_single(self, entity: Thing, base_ontology: Ontology, cache_name: str = None) -> ExecutionSummary:
        """
        isolates the validation of a single entity, capturing any exceptions 
        and returning a fallback ExecutionSummary if needed. 
        """
        async with self._semaphore:
            with self._tracer.start_as_current_span("validation_orchestrator.process_single") as span:
                span.set_attribute("entity.individual_id", entity.individual_id)
                try:
                    result = await self.strategy.run(entity, base_ontology, cache_name=cache_name)
                    span.set_status(Status(StatusCode.OK))
                    await asyncio.sleep(15.0)
                    return result

                except Exception as error:
                    error_msg = f"Critical failure processing {entity.individual_id}: {str(error)}"
                    span.set_status(Status(StatusCode.ERROR, error_msg))
                    span.record_exception(error)
                    span.set_attribute("error.stacktrace", traceback.format_exc()) 

                    await asyncio.sleep(5.0)

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