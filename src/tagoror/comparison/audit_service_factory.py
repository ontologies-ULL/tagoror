"""
The ONLY place that knows how to assemble the objects. It is what `run_batch` in
main.py does today, moved out so main.py and the tests share it.
"""

from pathlib import Path

from aiolimiter import AsyncLimiter
from opentelemetry import trace

from src.tagoror.core.audit.evaluation.strategies.llm_auditor import LLMEntityAuditor
from src.tagoror.core.audit.evaluation.strategies.majority_vote import ConsensusResolver
from src.tagoror.core.prompt_manager import PromptManager
from src.tagoror.serialization.serializers.turtle_serializer import TurtleSerializer

from src.tagoror..audit_service import AuditCoordinator, OntologyAuditService
from .llm_clients import BaseLLMClient, GeminiClient, LMStudioClient
from .model_catalog import ModelProfile, Provider


class AuditServiceFactory:

    def __init__(self, prompts_path: Path, user_input: str, suite_name: str = "owl_validations") -> None:
        self._prompts_path = prompts_path
        self._user_input = user_input
        self._suite_name = suite_name

    def create_client(self, profile: ModelProfile) -> BaseLLMClient:
        if profile.provider is Provider.GEMINI:
            return GeminiClient(profile)
        if profile.provider is Provider.LM_STUDIO:
            return LMStudioClient(profile)
        raise ValueError(f"No client for provider {profile.provider}")

    def create_service(self, client: BaseLLMClient, profile: ModelProfile) -> OntologyAuditService:
        limiter = (
            AsyncLimiter(max_rate=profile.requests_per_minute, time_period=60)
            if profile.requests_per_minute
            else None 
        )
        auditor = LLMEntityAuditor(
            model=client,
            prompt_manager=PromptManager(file_path=str(self._prompts_path)),
            serializer=TurtleSerializer(tracer=trace.get_tracer(__name__)),
            consensus_resolver=ConsensusResolver(),
            rate_limiter=limiter,
            user_input=self._user_input,
            suite_name=self._suite_name,
            model_name=profile.model_name,
        )
        return OntologyAuditService(AuditCoordinator(auditor))