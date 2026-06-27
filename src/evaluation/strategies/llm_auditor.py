from evaluation.entity_auditor import EntityAuditor
from llm.base_llm_client import BaseLLMClient
from core.prompt_manager import PromptManager 

class LLMEntityAuditor(EntityAuditor):
    """
    Concrete implementation of EntityAuditor that evaluates OWL individuals using a language model.
    """

    def __init__(self, model: BaseLLMClient, prompt_manager: PromptManager) -> None:
        self.model = model
        self.prompt_manager = prompt_manager

    async def run(self, individual: Any) -> ExecutionSummary:
        """
        Evaluates an OWL individual using a language model and returns an execution summary.

        Args:
            individual: The OWL individual extracted by OntologyExtractor.

        Returns:
            ExecutionSummary with results and aggregated real metrics.
        """
        pass