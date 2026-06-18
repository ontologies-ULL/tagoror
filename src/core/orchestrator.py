import asyncio
from datetime import datetime, timezone
from typing import Any

from core.models import ExecutionSummary, ExecutionMetrics, TaskOutcome, TaskStatus

class ValidationOrchestrator:
  """
  Orchestrates the parallel validation of extracted ontology entities.
  Delegates validation to a ValidationStrategy and critical errors to a ValidationNotifier.
  """

  def __init__(self, strategy: Any, notifier: Any):
    self.strategy = strategy
    self.notifier = notifier

  async def process(self, individuals: list[Any]) -> list[ExecutionSummary]:
    """
    Validates a list of entities concurrently.
    Failures are encapsulated into fallback ExecutionSummaries; exceptions are never raised.
    """
    if not individuals:
      return []

    tasks = [self._process_single(entity) for entity in individuals]
    return await asyncio.gather(*tasks)

  async def _process_single(self, entity: Any) -> ExecutionSummary:
    """
    Aisla la ejecución de un solo individuo. Si la estrategia falla, 
    captura la excepción, notifica y devuelve un sumario de error.
    """
    try:
      context = entity.to_llm_context()
      return await self.strategy.evaluate(context)
        
    except Exception as error:
      self.notifier.notify_critical_failure(entity.individual_id, error)
      
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