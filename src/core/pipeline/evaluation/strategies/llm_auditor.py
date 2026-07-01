import json
import asyncio

from core.pipeline.evaluation.entity_auditor import EntityAuditor
from llm.retry import RetryableLLMClient
from core.prompt_manager import PromptManager 
from core.models import ExecutionSummary
from core.models import ExecutionMetrics, TaskOutcome, TaskStatus
from llm.models import LLMPayload
from serialization.base_serializer import BaseSerializer 
from .majority_vote import ConsensusResolver
from .ontology_cache import OntologyCache

from owlready2 import Thing, Ontology
from datetime import datetime, timezone
from aiolimiter import AsyncLimiter

class LLMEntityAuditor(EntityAuditor):
    """
    Concrete implementation of EntityAuditor that evaluates OWL individuals using a language model.
    """

    def __init__(self, model: RetryableLLMClient, 
                 prompt_manager: PromptManager,
                 serializer: BaseSerializer,
                 consensus_resolver: ConsensusResolver,
                 rate_limiter: AsyncLimiter,
                 user_input: str = "",
                 suite_name: str = "owl_validations", 
                 model_name: str = "gemini-1.5-pro",
                 ) -> None:
        self.model = model
        self.prompt_manager = prompt_manager
        self.suite_name = suite_name
        self.model_name = model_name
        self.serializer = serializer
        self.user_input = user_input
        self.consensus_resolver = consensus_resolver
        self.rate_limiter = rate_limiter
        self.temperatures = [0.0]
        self.allow_web_search = False

    async def run(self, individual: Thing, base_ontology: Ontology) -> ExecutionSummary:
        """
        Evaluates an OWL individual using a language model and returns an execution summary.

        Args:
            individual: The OWL individual extracted by OntologyExtractor.
            base_ontology: The base ontology for context.

        Returns:
            ExecutionSummary with results and aggregated real metrics.
        """
        developer_prompt = self.prompt_manager.get_assembled_system_prompt()
        evaluation_suite = self.prompt_manager.get_evaluation_suite(self.suite_name).copy()

        suite_config = evaluation_suite.pop("configurations", {})
        self.temperatures = suite_config.get("temperatures", self.temperatures)
        self.allow_web_search = suite_config.get("allow_web_search", self.allow_web_search)
        
        output = []
        total_metrics = ExecutionMetrics(duration_ms=0, cost=0.0, tokens_consumed=0)
        
        individual_response = await asyncio.to_thread(
            self.serializer.process_individual, individual
        )
        serialized_base_ontology = await OntologyCache.get_serialized(base_ontology, self.serializer)
        context_data = {
            "individual_response": individual_response, 
            "base_ontology": serialized_base_ontology,
            "user_input": self.user_input,
        }
        task_coroutines = [
            self._run_task_with_consensus(task_id, task_config, context_data, developer_prompt)
            for task_id, task_config in evaluation_suite.items()
        ]

        parallel_results = await asyncio.gather(*task_coroutines)
        output = []

        for final_outcome, total_cost, total_tokens, max_duration in parallel_results:
            output.append(final_outcome)
            total_metrics.cost += total_cost
            total_metrics.tokens_consumed += total_tokens
            total_metrics.duration_ms += max_duration 
            
        return ExecutionSummary(
            individual_id=individual.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            results=output,
            total_metrics=total_metrics,
            system_summary=f"Evaluated {individual.name}. Total tasks: {len(evaluation_suite)}."        )
    
    async def _run_task_with_consensus(self, task_id: str, task_config: dict, context_data: dict, developer_prompt: str, suite_config: dict = None) -> tuple:
        raw_prompt = task_config.get("prompt", "")
        if not raw_prompt:
            raise ValueError(f"Task {task_id} is missing a 'prompt'.")
            
        user_prompt = self._safe_format(raw_prompt, context_data)
        temperatures = task_config.get("temperatures", self.temperatures) 
        allow_web = task_config.get("allow_web_search", self.allow_web_search)

        async def _run_temperature_branch(temp: float):
            payload = LLMPayload(
                model_name=self.model_name,
                system_prompt=developer_prompt,
                user_prompt=user_prompt,
                json_mode=True,
                allow_web_search=allow_web,
                temperature=temp 
            )
            if self.rate_limiter:
                async with self.rate_limiter:
                    response = await self.model.query(payload)
            else:
                response = await self.model.query(payload)
            
            outcome = self._parse_single_task_response(response, task_id)
            return outcome, temp, response

        branch_coroutines = [_run_temperature_branch(temp) for temp in temperatures]
        branch_results = await asyncio.gather(*branch_coroutines)

        outcomes_with_temps = [(res[0], res[1]) for res in branch_results]
        network_responses = [res[2] for res in branch_results]
        
        final_outcome = self.consensus_resolver.resolve(task_id, outcomes_with_temps)

        if final_outcome is None:
            fallback_outcome, _, fallback_response = await _run_temperature_branch(0.0)
            
            fallback_outcome.findings.insert(0, "[CONSENSUS FAILURE: Resolved by fallback execution at temp 0.0]")
            final_outcome = fallback_outcome
            
            network_responses.append(fallback_response)

        total_cost = sum(getattr(resp, 'cost', 0.0) for resp in network_responses)
        total_tokens = sum(getattr(resp, 'tokens_consumed', 0) for resp in network_responses)
        max_duration = sum(getattr(resp, 'duration_ms', 0) for resp in network_responses)

        return final_outcome, total_cost, total_tokens, max_duration

    def _safe_format(self, template: str, context: dict) -> str:
        """
        Safely formats a string template with the provided context, ignoring missing keys.
        """
        class SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"
        return template.format_map(SafeDict(context))

    def _parse_single_task_response(self, response, task_id: str) -> TaskOutcome:
        """
        Parser for the response of a single task, extracting the status and findings.
        """
        raw_text = response.raw_content.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.removeprefix("```json").removesuffix("```").strip()

        try:
            parsed_data = json.loads(raw_text)
            status_str = str(parsed_data.get("status", "failure")).lower().strip()
            
            if status_str in ["success", "compliant", "passed", "pass", "ok", "true"]:
                status_val = TaskStatus.SUCCESS
            elif status_str in ["failure", "failed", "non-compliant", "non_compliant", "error", "false"]:
                status_val = TaskStatus.FAILURE
            else:
                try:
                    status_val = TaskStatus(status_str)
                except ValueError:
                    status_val = TaskStatus.FAILURE
                    findings = parsed_data.get("findings", [])
                    findings.insert(0, f"[SYSTEM WARNING: Unrecognized status '{status_str}' returned by LLM, mapped to FAILURE]")
                    return TaskOutcome(
                        task_id=task_id,
                        status=status_val,
                        findings=findings
                    )

            return TaskOutcome(
                task_id=task_id,
                status=status_val,
                findings=parsed_data.get("findings", [])
            )
        except json.JSONDecodeError as error:
            return TaskOutcome(
                task_id=task_id,
                status=TaskStatus.FAILURE,
                findings=[f"Error parsing JSON response: {str(error)}", f"Raw: {raw_text}"]
            )