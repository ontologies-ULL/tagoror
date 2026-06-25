"""
Validation strategy that relies solely on an LLM to evaluate OWL entities.
One LLM call is made per prompt in the PromptRegistry, producing one
TaskOutcome per call. Metrics are aggregated across all calls.
"""

from datetime import datetime, timezone

from execution.base_entity_validation import ValidationStrategy
from core.models import (
    ExecutionSummary,
    ExecutionMetrics,
    TaskOutcome,
    TaskStatus,
    PromptRegistry,
)
from llm.base_llm_client import BaseLLMClient
from llm.models import LLMPayload, LLMResponse


class PureLLMStrategy(ValidationStrategy):
    """
    Validation strategy that uses a pure LLM approach.

    For each prompt in the PromptRegistry the strategy:
      1. Builds an LLMPayload with the prompt and the OWL entity.
      2. Calls the injected BaseLLMClient.
      3. Transforms the free-text LLM response into a TaskOutcome.
      4. Aggregates ExecutionMetrics across all calls into the ExecutionSummary.
    """

    def __init__(self, llm_client: BaseLLMClient, context: PromptRegistry):
        self._llm_client = llm_client
        self._context    = context

    async def evaluate(self, owl_entity: str) -> ExecutionSummary:
        """
        Evaluate the OWL entity against all prompts in the registry.

        Args:
            owl_entity: The OWL entity string to evaluate.

        Returns:
            ExecutionSummary with one TaskOutcome per prompt and
            aggregated ExecutionMetrics.
        """
        outcomes: list[TaskOutcome] = []
        total_duration_ms = 0
        total_tokens      = 0
        total_cost        = 0.0

        for task_id, prompt in self._context.loaded_prompts.items():
            payload  = self._build_payload(prompt, owl_entity)
            response = await self._llm_client.query(payload)

            outcome = self._parse_response(task_id, response)
            outcomes.append(outcome)

            total_duration_ms += response.duration_ms
            total_tokens      += response.tokens_consumed
            total_cost        += response.cost

        return ExecutionSummary(
            individual_id=owl_entity,
            timestamp=datetime.now(timezone.utc).isoformat(),
            results=outcomes,
            total_metrics=ExecutionMetrics(
                duration_ms=total_duration_ms,
                tokens_consumed=total_tokens,
                cost=total_cost,
            ),
            system_summary=self._build_summary(outcomes),
        )

    def _build_payload(self, prompt: str, owl_entity: str) -> LLMPayload:
        """Combines the prompt template with the OWL entity into an LLMPayload."""
        return LLMPayload(
            system_prompt=prompt,
            user_prompt=owl_entity,
            json_mode=False,
        )

    def _parse_response(self, task_id: str, response: LLMResponse) -> TaskOutcome:
        """
        Transforms a free-text LLMResponse into a TaskOutcome.
        Status is set to SUCCESS by default — extend this method when
        the response format becomes structured enough to derive a real status.
        """
        return TaskOutcome(
            task_id=task_id,
            status=TaskStatus.SUCCESS,
            findings=[response.raw_content],
            metrics=ExecutionMetrics(
                duration_ms=response.duration_ms,
                tokens_consumed=response.tokens_consumed,
                cost=response.cost,
            ),
            rawResponse=response.raw_content,
        )

    def _build_summary(self, outcomes: list[TaskOutcome]) -> str:
        """Produces a plain-text summary of how many tasks were evaluated."""
        return f"PureLLMStrategy completed. Tasks evaluated: {len(outcomes)}."