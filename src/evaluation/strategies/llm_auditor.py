from evaluation.entity_auditor import EntityAuditor
from llm.base_llm_client import BaseLLMClient
from core.prompt_manager import PromptManager 
from core.models import ExecutionSummary
from core.models import ExecutionMetrics, TaskOutcome, TaskStatus
from llm.models import LLMPayload
from owlready2 import Thing
from datetime import datetime, timezone
import json

class LLMEntityAuditor(EntityAuditor):
    """
    Concrete implementation of EntityAuditor that evaluates OWL individuals using a language model.
    """

    def __init__(self, model: BaseLLMClient, 
                 prompt_manager: PromptManager, 
                 suite_name: str = "owl_validations", 
                 model_name: str = "gemini-1.5-pro",
                 serializator) -> None:
        self.model = model
        self.prompt_manager = prompt_manager
        self.suite_name = suite_name
        self.model_name = model_name
        self.serializator = serializator

    async def run(self, individual: Thing, base_ontology: Any) -> ExecutionSummary:
        """
        Evaluates an OWL individual using a language model and returns an execution summary.

        Args:
            individual: The OWL individual extracted by OntologyExtractor.

        Returns:
            ExecutionSummary with results and aggregated real metrics.
        """
        developer_prompt = self.prompt_manager.get_assembled_system_prompt()
        evaluation_suite = self.prompt_manager.get_evaluation_suite(self.suite_name)
        output = []
        total_metrics = ExecutionMetrics(duration_ms=0, cost=0.0, tokens_consumed=0)
        context_data = {
            "individual_response": individual.to_llm_context(),
            "base_ontology":  

        }

        for task_id, task_config in evaluation_suite.items():
            raw_prompt = task_config.get("prompt", "")
            user_prompt = self._safe_format(raw_prompt, context_data)
            payload = LLMPayload(
                model_name=self.model_name,
                system_prompt=developer_prompt,
                user_prompt=user_prompt,
                json_mode=True
            )
            response = await self._llm_client.query(payload)
            total_metrics.duration_ms += getattr(response, 'duration_ms', 0)
            total_metrics.cost += getattr(response, 'cost', 0.0)
            total_metrics.tokens_consumed += getattr(response, 'tokens_consumed', 0)

            # TODO: Fix this 
            output = self._parse_single_task_response(response, task_id)
            output.append(output)

        return ExecutionSummary(
            individual_id=individual.individual_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            results=output,
            total_metrics=total_metrics,
            system_summary=f"Evaluated individual {individual.individual_id} using model {self.model_name} with suite {self.suite_name}. Total tasks: {len(suite_tasks)}."
        )

    def _safe_format(self, template_str: str, data: dict) -> str:
        """

        """
        formatter = string.Formatter()
        mapping = {k: data.get(k, f"{{{k}}}") for _, k, _, _ in formatter.parse(template_str) if k is not None}
        return template_str.format(**mapping)

    def _parse_single_task_response(self, response, task_id: str) -> TaskOutcome:
        """
        Parser for the response of a single task, extracting the status and findings.
        """
        raw_text = response.raw_content.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.removeprefix("```json").removesuffix("```").strip()

        try:
            parsed_data = json.loads(raw_text)
            return TaskOutcome(
                task_id=task_id,
                status=TaskStatus(parsed_data.get("status", "failure").lower()),
                findings=parsed_data.get("findings", [])
            )
        except json.JSONDecodeError as error:
            return TaskOutcome(
                task_id=task_id,
                status=TaskStatus.FAILURE,
                findings=[f"Error parsing JSON response: {str(error)}"])