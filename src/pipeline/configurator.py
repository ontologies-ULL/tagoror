"""
Pipeline configuration and dependency initialization.
Handles all LLM, auditor, and orchestrator setup.
"""

from pathlib import Path
from opentelemetry import trace
from core.pipeline.pipeline import Pipeline
from core.pipeline.orchestrator import EntityOrchestrator
from core.pipeline.evaluation.strategies.llm_auditor import LLMEntityAuditor
from core.pipeline.evaluation.strategies.majority_vote import ConsensusResolver
from core.prompt_manager import PromptManager
from serialization.serializers.turtle_serializer import TurtleSerializer
from llm.clients.gemini import GeminiClient 
from llm.retry import RetryableLLMClient
from llm.config import RetryPolicyConfig, BackoffStrategy
from ui.status_reporter import StatusReporter
from aiolimiter import AsyncLimiter


class PipelineConfigurator:
    """
    Builds and configures the validation pipeline with all dependencies.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def build_llm(self) -> RetryableLLMClient:
        """
        Build and configure the LLM client with retry policy.
        """
        base_llm = GeminiClient(api_key=self.api_key)
        retry_config = RetryPolicyConfig(
            max_retries=3, 
            delay_between_retries=1.0, 
            backoff_strategy=BackoffStrategy.EXPONENTIAL
        )
        return RetryableLLMClient(llm_client=base_llm, config=retry_config)

    def build_pipeline(
        self,
        llm_client: RetryableLLMClient,
        prompts_path: Path,
        rate_limit: float,
        model_name: str,
        user_input: str,
    ) -> tuple[Pipeline, StatusReporter]:
        """
        Build the complete pipeline with all components.
        
        Returns:
            (Pipeline instance, StatusReporter instance)
        """
        prompt_manager = PromptManager(file_path=str(prompts_path))
        tracer = trace.get_tracer(__name__)
        serializer = TurtleSerializer(tracer)
        consensus_resolver = ConsensusResolver()
        global_rate_limiter = AsyncLimiter(max_rate=rate_limit, time_period=60)

        auditor = LLMEntityAuditor(
            model=llm_client,
            prompt_manager=prompt_manager,
            serializer=serializer,
            consensus_resolver=consensus_resolver,
            rate_limiter=global_rate_limiter,
            user_input=user_input,
            suite_name="owl_validations",
            model_name=model_name
        )

        status_reporter = StatusReporter()
        orchestrator = EntityOrchestrator(strategy=auditor, status_reporter=status_reporter)
        pipeline = Pipeline(orchestrator=orchestrator)

        return pipeline, status_reporter
