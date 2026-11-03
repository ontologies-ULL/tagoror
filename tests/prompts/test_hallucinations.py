"""
Unit and integration tests for prompt evaluation suites and LLM client workflows.
Covers:
  - PromptManager prompt assembling for all 4 hallucination evaluation tasks:
    structural, semantic, contextual, and factuality.
  - End-to-end task execution through LLMEntityAuditor using:
    * GeminiClient
    * LocalStudioClient (AsyncOpenAI + Instructor + JIT VRAM load/unload)
    * RetryableLLMClient wrapper
  - Detection and classification of structural, semantic, contextual, and factual failures.
  - Flexible Live benchmarking suite (--live-models) testing single, multiple, or ALL local models.
  - Security constraints: zero leakage of local/remote endpoints, ports, or raw command URLs.
"""

import os
import json
import httpx
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from tagoror.core.prompt_manager import PromptManager
from tagoror.core.pipeline.evaluation.strategies.llm_auditor import LLMEntityAuditor
from tagoror.core.pipeline.evaluation.strategies.majority_vote import ConsensusResolver
from tagoror.core.models import TaskStatus, TaskOutcome, ExecutionSummary
from tagoror.llm.models import LLMPayload, LLMResponse
from tagoror.llm.config import RetryPolicyConfig, BackoffStrategy
from tagoror.llm.retry import RetryableLLMClient
from tagoror.llm.clients.gemini import GeminiClient
from tagoror.llm.clients.local_studio import LocalStudioClient


# ---------------------------------------------------------------------------
# Dynamic Model Discovery Helper
# ---------------------------------------------------------------------------
def get_server_host() -> str:
    """Returns the base host without trailing paths or slashes."""
    raw_host = os.getenv("LOCAL_STUDIO_HOST") or os.getenv("LOCAL_STUDIO_URL", "http://127.0.0.1:1234")
    return raw_host.removesuffix("/api/v1").removesuffix("/v1").rstrip("/")

def check_local_server_active(host: str, api_key: str = "lm-studio") -> bool:
    """Checks if the LM Studio local server is up and listening on /api/v1/models."""
    endpoint = f"{host}/api/v1/models"
    try:
        res = httpx.get(
            endpoint, 
            headers={"Authorization": f"Bearer {api_key}"}, 
            timeout=1.5
        )
        return res.status_code == 200
    except Exception:
        return False

def _is_embedding_model(model_id: str, raw_item: dict | None = None) -> bool:
    """
    Returns True if the model identifier or its metadata belongs to an embedding model.
    """
    lower_id = model_id.lower()
    embedding_keywords = ("embed", "nomic", "bge-", "e5-", "gte-", "minilm")
    if any(keyword in lower_id for keyword in embedding_keywords):
        return True

    if isinstance(raw_item, dict):
        model_type = str(raw_item.get("type", "")).lower()
        arch = str(raw_item.get("arch", "")).lower()
        if "embed" in model_type or "bert" in arch:
            return True
    return False

def resolve_target_models() -> list[str]:
    """
    Resolves available models directly from the LM Studio /api/v1/models endpoint,
    ensuring only the clean model string identifier is extracted.
    """
    host = get_server_host()
    api_key = os.getenv("LOCAL_STUDIO_API_KEY", "lm-studio")
    configured_model = os.getenv("LOCAL_STUDIO_MODEL", "").strip()

    if not check_local_server_active(host, api_key):
        return []

    if configured_model and configured_model.lower() != "all":
        return [m.strip() for m in configured_model.split(",") if m.strip()]

    endpoint = f"{host}/api/v1/models"
    res = httpx.get(
        endpoint, 
        headers={"Authorization": f"Bearer {api_key}"}, 
        timeout=2.0
    )
    if res.status_code == 200:
        data = res.json()
        models_list = data.get("models", data.get("data", []))
        
        discovered = []
        for item in models_list:
            if isinstance(item, dict):
                model_id = (
                    item.get("key")
                    or item.get("id") 
                    or item.get("selected_variant") 
                )
                if model_id and isinstance(model_id, str) and not _is_embedding_model(model_id, raw_item=item): 
                    discovered.append(model_id)
            elif isinstance(item, str):
                discovered.append(item)

        if discovered:
            return discovered
    
    return ["default-local-model"]

TARGET_MODELS = resolve_target_models()


# ---------------------------------------------------------------------------
# Global Environment & Sanitization Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def sanitize_environment(monkeypatch, request):
    if request.node.get_closest_marker("live_llm"):
        return
    if "LOCAL_STUDIO_HOST" not in os.environ and "LOCAL_STUDIO_URL" not in os.environ:
        monkeypatch.setenv("LOCAL_STUDIO_HOST", "http://redacted-endpoint.internal")
    if "GEMINI_API_KEY" not in os.environ:
        monkeypatch.setenv("GEMINI_API_KEY", "REDACTED_API_KEY")

@pytest.fixture
def base_ontology_mock():
    """Mock ontology instance representing the baseline schema."""
    ontology = MagicMock()
    ontology.base_iri = "http://www.ull.es/iis/simulation/ontologies/chocolate-soldier#"
    ontology.name = "chocolate-soldier"
    return ontology


@pytest.fixture
def mock_serializer():
    """Mock serializer transforming entities into simulated RDF/Turtle/JSON-LD structures."""
    serializer = MagicMock()
    serializer.process_ontology.return_value = (
        "@prefix : <http://www.ull.es/iis/simulation/ontologies/chocolate-soldier#> .\n"
        ":Soldado a owl:Class .\n"
        ":Alimento a owl:Class .\n"
        ":Chocolate rdfs:subClassOf :Alimento .\n"
    )
    serializer.process_individual.side_effect = lambda ind: (
        f'{{"entity_id": "{ind.name}", "class_types": ["{ind.type}"], '
        f'"properties": {json.dumps(getattr(ind, "properties", {}))}}}'
    )
    return serializer


@pytest.fixture
def mock_rate_limiter():
    """Asynchronous no-op limiter to avoid timing stalls during testing."""
    limiter = MagicMock()
    limiter.__aenter__ = AsyncMock(return_value=None)
    limiter.__aexit__ = AsyncMock(return_value=None)
    return limiter


@pytest.fixture
def prompt_manager():
    """Loads configuration and task suites from prompts.yaml."""
    return PromptManager()


@pytest.fixture
def retry_config():
    """Configures immediate retry policies without sleep delays for fast test execution."""
    return RetryPolicyConfig(
        max_retries=2,
        delay_between_retries=0,
        backoff_strategy=BackoffStrategy.FIXED
    )


# ---------------------------------------------------------------------------
# Tests: PromptManager & Evaluation Suite Structure
# ---------------------------------------------------------------------------

class TestPromptManagerTasks:

    def test_evaluation_suite_tasks_exist(self, prompt_manager):
        """Verifies that all four evaluation tasks are correctly parsed from prompts.yaml."""
        tasks = prompt_manager.get_suite_tasks("owl_validations")
        expected_tasks = {
            "structural_evaluation",
            "semantic_evaluation",
            "contextual_evaluation",
            "factuality_evaluation"
        }
        assert expected_tasks.issubset(tasks.keys())

    def test_structural_evaluation_prompt_construction(self, prompt_manager):
        """Ensures structural evaluation prompt injects individual payload and CoT directives."""
        task_config = prompt_manager.get_suite_tasks("owl_validations")["structural_evaluation"]
        context = {"individual_response": '{"entity_id": "Vehiculo_WarpTank_01", "class_types": ["TanqueDeVapor"]}'}

        prompt = prompt_manager.build_task_prompt(task_config, context, default_self_criticism=True)
        assert "TYPE: Structural Hallucination" in prompt
        assert "Vehiculo_WarpTank_01" in prompt
        assert "Self-Criticism Required" in prompt

    def test_semantic_evaluation_prompt_construction(self, prompt_manager):
        """Ensures semantic evaluation prompt correctly injects input context."""
        task_config = prompt_manager.get_suite_tasks("owl_validations")["semantic_evaluation"]
        context = {"individual_response": '{"entity_id": "Soldado_De_Cacao", "class_types": ["Soldado", "Chocolate"]}'}

        prompt = prompt_manager.build_task_prompt(task_config, context, default_self_criticism=True)
        assert "TYPE: Semantic Hallucination" in prompt
        assert "Soldado_De_Cacao" in prompt

    def test_contextual_evaluation_prompt_construction(self, prompt_manager):
        """Ensures contextual evaluation prompt contains both user input and individual response."""
        task_config = prompt_manager.get_suite_tasks("owl_validations")["contextual_evaluation"]
        context = {
            "user_input": "Soldado en trinchera perdiendo cordura",
            "individual_response": '{"entity_id": "Soldado_Ceremonial_Retirado"}'
        }

        prompt = prompt_manager.build_task_prompt(task_config, context, default_self_criticism=True)
        assert "TYPE: Contextual Hallucination" in prompt
        assert "Soldado en trinchera perdiendo cordura" in prompt
        assert "Soldado_Ceremonial_Retirado" in prompt

    def test_factuality_evaluation_prompt_construction(self, prompt_manager):
        """Ensures factuality evaluation prompt is properly built and allow_web_search flag is set."""
        task_config = prompt_manager.get_suite_tasks("owl_validations")["factuality_evaluation"]
        context = {"individual_response": '{"hasSource": "Fake Journal 2021"}'}

        prompt = prompt_manager.build_task_prompt(task_config, context, default_self_criticism=True)
        assert "TYPE: Evidence-based (Factuality) Hallucination" in prompt
        assert task_config.get("allow_web_search") is True


# ---------------------------------------------------------------------------
# Tests: LLM Clients Integration with the Refactored BaseLLMClient
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAuditorWithAllLLMClients:

    @pytest.fixture
    def test_individual(self):
        ind = MagicMock()
        ind.name = "Test_Entity"
        ind.type = "Soldado"
        ind.properties = {"nivelDeLocura": 5}
        return ind

    async def test_audit_flow_with_gemini_client(
        self, prompt_manager, mock_serializer, mock_rate_limiter, base_ontology_mock, test_individual, mocker
    ):
        """Tests the auditor executing validation tasks using GeminiClient."""
        mock_gemini = MagicMock(spec=GeminiClient)
        gemini_response = LLMResponse(
            raw_content='{"reasoning": "Schema verified", "status": "SUCCESS", "findings": []}',
            tokens_consumed=150,
            duration_ms=250,
            cost=0.0003
        )
        mock_gemini.query = AsyncMock(return_value=gemini_response)

        auditor = LLMEntityAuditor(
            model=mock_gemini,
            prompt_manager=prompt_manager,
            serializer=mock_serializer,
            consensus_resolver=ConsensusResolver(),
            rate_limiter=mock_rate_limiter,
            suite_name="owl_validations"
        )

        mocker.patch("asyncio.sleep", new=AsyncMock())
        summary: ExecutionSummary = await auditor.run(test_individual, base_ontology_mock)

        assert summary.individual_id == "Test_Entity"
        assert len(summary.results) == 4
        assert all(res.status == TaskStatus.SUCCESS for res in summary.results)
        assert mock_gemini.query.call_count >= 4

    async def test_audit_flow_with_local_studio_client_redacted(
        self, prompt_manager, mock_serializer, mock_rate_limiter, base_ontology_mock, test_individual, mocker
    ):
        """
        Tests the auditor with LocalStudioClient while verifying zero exposure
        of target endpoints, internal ports, or execution commands.
        """
        with patch("tagoror.llm.clients.local_studio.AsyncOpenAI"), \
             patch("tagoror.llm.clients.local_studio.instructor.from_openai"):

            client = LocalStudioClient(base_host="http://[REDACTED_ENDPOINT]:0")

            mock_instructor_client = MagicMock()
            mock_raw_response = MagicMock()
            mock_raw_response.text = '{"reasoning": "Valid entity", "status": "SUCCESS", "findings": []}'
            mock_raw_response.usage = MagicMock()
            mock_raw_response.usage.prompt_tokens = 50
            mock_raw_response.usage.completion_tokens = 30

            mock_instructor_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(None, mock_raw_response)
            )

            mocker.patch.object(LocalStudioClient, "client", new_callable=lambda: mock_instructor_client)

            auditor = LLMEntityAuditor(
                model=client,
                prompt_manager=prompt_manager,
                serializer=mock_serializer,
                consensus_resolver=ConsensusResolver(),
                rate_limiter=mock_rate_limiter,
                suite_name="owl_validations"
            )

            mocker.patch("asyncio.sleep", new=AsyncMock())
            summary: ExecutionSummary = await auditor.run(test_individual, base_ontology_mock)

            assert summary.individual_id == "Test_Entity"
            assert summary.is_successful() is True

            all_findings_str = " ".join([f for res in summary.results for f in res.findings])
            assert "http" not in all_findings_str
            assert "localhost" not in all_findings_str
            assert "127.0.0.1" not in all_findings_str
            assert "1234" not in all_findings_str

    async def test_audit_flow_with_retryable_client(
        self, prompt_manager, mock_serializer, mock_rate_limiter, base_ontology_mock, test_individual, retry_config, mocker
    ):
        """Tests the auditor wrapped with RetryableLLMClient transparent resilience."""
        mock_underlying_client = MagicMock()
        mock_underlying_client.provider_name = "mock_provider"
        mock_underlying_client.cost_per_m_in = 0.0
        mock_underlying_client.cost_per_m_out = 0.0
        mock_underlying_client.client = MagicMock()

        success_response = LLMResponse(
            raw_content='{"reasoning": "Validated after transient retry", "status": "SUCCESS", "findings": []}',
            tokens_consumed=110,
            duration_ms=190,
            cost=0.0001
        )
        mock_underlying_client.query = AsyncMock(return_value=success_response)

        # Ensure compatibility with BaseLLMClient abstract properties
        mocker.patch.object(RetryableLLMClient, "__abstractmethods__", set())
        mocker.patch.object(
            RetryableLLMClient,
            "client",
            property(lambda self: getattr(self.llm_client, "client", None))
        )

        retry_client = RetryableLLMClient(llm_client=mock_underlying_client, config=retry_config)

        auditor = LLMEntityAuditor(
            model=retry_client,
            prompt_manager=prompt_manager,
            serializer=mock_serializer,
            consensus_resolver=ConsensusResolver(),
            rate_limiter=mock_rate_limiter,
            suite_name="owl_validations"
        )

        mocker.patch("asyncio.sleep", new=AsyncMock())
        summary: ExecutionSummary = await auditor.run(test_individual, base_ontology_mock)

        assert summary.is_successful() is True
        assert mock_underlying_client.query.call_count >= 4


# ---------------------------------------------------------------------------
# Tests: Hallucination Verification Suites (Unit / Mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestHallucinationDetections:

    @pytest.fixture
    def auditor_factory(self, prompt_manager, mock_serializer, mock_rate_limiter):
        def _create_auditor(client_response: LLMResponse, user_input: str = ""):
            client = MagicMock()
            client.query = AsyncMock(return_value=client_response)
            return LLMEntityAuditor(
                model=client,
                prompt_manager=prompt_manager,
                serializer=mock_serializer,
                consensus_resolver=ConsensusResolver(),
                rate_limiter=mock_rate_limiter,
                user_input=user_input,
                suite_name="owl_validations"
            )
        return _create_auditor

    async def test_detects_structural_hallucination(self, auditor_factory, base_ontology_mock, mocker):
        """Verifies detection of non-existent classes and properties (#TanqueDeVapor, #disparaA)."""
        mocker.patch("asyncio.sleep", new=AsyncMock())
        mock_ind = MagicMock()
        mock_ind.name = "Vehiculo_WarpTank_01"
        mock_ind.type = "TanqueDeVapor"
        mock_ind.properties = {"disparaA": "RataSoldado_Krit"}

        structural_failure_response = LLMResponse(
            raw_content=json.dumps({
                "reasoning": "Class 'TanqueDeVapor' and property 'disparaA' are undefined in the TBox.",
                "status": "FAILURE",
                "findings": ["Class 'TanqueDeVapor' undefined", "Property 'disparaA' undefined"]
            }),
            tokens_consumed=90,
            duration_ms=100,
            cost=0.0001
        )

        auditor = auditor_factory(structural_failure_response)
        summary = await auditor.run(mock_ind, base_ontology_mock)

        assert summary.is_successful() is False
        structural_outcome = next(res for res in summary.results if res.task_id == "structural_evaluation")
        assert structural_outcome.status == TaskStatus.FAILURE
        assert "Class 'TanqueDeVapor' undefined" in structural_outcome.findings

    async def test_detects_semantic_hallucination(self, auditor_factory, base_ontology_mock, mocker):
        """Verifies detection of disjoint class conflicts and range/domain violations (#Soldado_De_Cacao)."""
        mocker.patch("asyncio.sleep", new=AsyncMock())
        mock_ind = MagicMock()
        mock_ind.name = "Soldado_De_Cacao"
        mock_ind.type = "Soldado"
        mock_ind.properties = {"consume": "Soldado_Queek"}

        semantic_failure_response = LLMResponse(
            raw_content=json.dumps({
                "reasoning": "Individual instantiates disjoint classes Soldado and Chocolate simultaneously.",
                "status": "FAILURE",
                "findings": ["Disjoint classes violation", "Range violation on property 'consume'"]
            }),
            tokens_consumed=95,
            duration_ms=110,
            cost=0.0001
        )

        auditor = auditor_factory(semantic_failure_response)
        summary = await auditor.run(mock_ind, base_ontology_mock)

        semantic_outcome = next(res for res in summary.results if res.task_id == "semantic_evaluation")
        assert semantic_outcome.status == TaskStatus.FAILURE
        assert any("Disjoint classes" in f for f in semantic_outcome.findings)

    async def test_detects_contextual_hallucination(self, auditor_factory, base_ontology_mock, mocker):
        """Verifies detection of entities that deviate from the prompt's operational boundary."""
        mocker.patch("asyncio.sleep", new=AsyncMock())
        mock_ind = MagicMock()
        mock_ind.name = "Soldado_Ceremonial_Retirado"
        mock_ind.type = "Soldado"
        mock_ind.properties = {"tiempoEnTrinchera": 0, "nivelDeLocura": 0}

        contextual_failure_response = LLMResponse(
            raw_content=json.dumps({
                "reasoning": "User input specified active trench warfare; entity describes peacetime ceremonial role.",
                "status": "FAILURE",
                "findings": ["Entity does not address requested context"]
            }),
            tokens_consumed=105,
            duration_ms=130,
            cost=0.0001
        )

        user_input_spec = "Soldados skaven en trincheras de primera línea perdiendo la cordura"
        auditor = auditor_factory(contextual_failure_response, user_input=user_input_spec)
        summary = await auditor.run(mock_ind, base_ontology_mock)

        contextual_outcome = next(res for res in summary.results if res.task_id == "contextual_evaluation")
        assert contextual_outcome.status == TaskStatus.FAILURE
        assert "Entity does not address requested context" in contextual_outcome.findings

    async def test_detects_factuality_hallucination(self, auditor_factory, base_ontology_mock, mocker):
        """Verifies detection of fabricated academic citations and unsubstantiated numerical metrics."""
        mocker.patch("asyncio.sleep", new=AsyncMock())
        mock_ind = MagicMock()
        mock_ind.name = "Soldado_Snikch"
        mock_ind.type = "Soldado"
        mock_ind.properties = {
            "hasSource": "Lancet Psychiatry 2021; Theobromine Warfare",
            "hasExpectedValue": 0.023
        }

        factuality_failure_response = LLMResponse(
            raw_content=json.dumps({
                "reasoning": "The cited reference 'Theobromine Warfare' is non-existent in Lancet Psychiatry.",
                "status": "FAILURE",
                "findings": ["Fabricated study citation", "Value not found in source"]
            }),
            tokens_consumed=120,
            duration_ms=160,
            cost=0.0002
        )

        auditor = auditor_factory(factuality_failure_response)
        summary = await auditor.run(mock_ind, base_ontology_mock)

        factuality_outcome = next(res for res in summary.results if res.task_id == "factuality_evaluation")
        assert factuality_outcome.status == TaskStatus.FAILURE
        assert "Fabricated study citation" in factuality_outcome.findings


# ---------------------------------------------------------------------------
# Tests: Live Local LLM Benchmark with JIT VRAM Lifecycle
# Evaluates all 4 hallucination types against 1, multiple, or ALL local models
# ---------------------------------------------------------------------------

@pytest.mark.live_llm
@pytest.mark.asyncio
class TestLiveLocalStudioAllHallucinations:
    """
    Executes live prompts against all resolved local models (single, comma-separated, or all).
    Explicitly loads models into VRAM and unloads them upon test completion.
    Tests all 4 hallucination dimensions: Structural, Semantic, Contextual, and Factuality.
    """

    @pytest.fixture
    def live_client(self):
        return LocalStudioClient(
            base_host=get_server_host(),
            api_key=os.getenv("LOCAL_STUDIO_API_KEY", "lm-studio"),
        )

    @pytest_asyncio.fixture
    async def managed_model(self, live_client, request):
        model_name = request.param
        if _is_embedding_model(model_name):
            pytest.skip(f"Skipping embedding model: {model_name}")

        instance_id = None
        try:
            load_result = await live_client.load_model(model_name=model_name, ttl=300)
            instance_id = load_result.get("instance_id")
        except Exception as e:
            print(f"\n[WARN] Explicit load_model failed for '{model_name}': {e}")
            pytest.skip(f"Skipping model '{model_name}' due to load failure.")

        yield model_name

        if instance_id:
            try:
                await live_client.unload_model(instance_id=instance_id)
            except Exception as e:
                print(f"\n[WARN] Explicit unload_model failed for '{instance_id}': {e}")

    def _create_live_auditor(self, client: LocalStudioClient, model_name: str, prompt_manager, mock_serializer, user_input: str = ""):
        return LLMEntityAuditor(
            model=client,
            prompt_manager=prompt_manager,
            serializer=mock_serializer,
            consensus_resolver=ConsensusResolver(),
            rate_limiter=None,
            model_name=model_name,
            user_input=user_input,
            suite_name="owl_validations"
        )
    
    
    @pytest.mark.parametrize("managed_model", TARGET_MODELS, indirect=True)
    async def test_live_structural_hallucination_discard(
        self, live_client, managed_model, prompt_manager, mock_serializer, base_ontology_mock
    ):
        """
        Dimension 1: Structural Hallucination.
        Verifies if model detects non-existent classes (#TanqueDeVapor) and properties (#disparaA).
        """
        auditor = self._create_live_auditor(live_client, managed_model, prompt_manager, mock_serializer)

        mock_ind = MagicMock()
        mock_ind.name = "Vehiculo_WarpTank_01"
        mock_ind.type = "TanqueDeVapor"
        mock_ind.properties = {"disparaA": "RataSoldado_Krit"}

        summary: ExecutionSummary = await auditor.run(mock_ind, base_ontology_mock)
        outcome = next(res for res in summary.results if res.task_id == "structural_evaluation")

        assert outcome.status == TaskStatus.FAILURE, (
            f"DISCARD [{managed_model}]: Model failed to detect structural hallucination. "
            f"Raw Response: {outcome.rawResponse}"
        )

    @pytest.mark.parametrize("managed_model", TARGET_MODELS, indirect=True)
    async def test_live_semantic_hallucination_discard(
        self, live_client, managed_model, prompt_manager, mock_serializer, base_ontology_mock
    ):
        """
        Dimension 2: Semantic Hallucination.
        Verifies if model flags disjoint class inheritance (#Soldado and #Chocolate).
        """
        auditor = self._create_live_auditor(live_client, managed_model, prompt_manager, mock_serializer)

        mock_ind = MagicMock()
        mock_ind.name = "Soldado_De_Cacao"
        mock_ind.type = "Soldado"
        mock_ind.properties = {"consume": "Soldado_Queek"}

        summary: ExecutionSummary = await auditor.run(mock_ind, base_ontology_mock)
        outcome = next(res for res in summary.results if res.task_id == "semantic_evaluation")

        assert outcome.status == TaskStatus.FAILURE, (
            f"DISCARD [{managed_model}]: Model failed to detect semantic disjoint conflict. "
            f"Raw Response: {outcome.rawResponse}"
        )

    @pytest.mark.parametrize("managed_model", TARGET_MODELS, indirect=True)
    async def test_live_contextual_hallucination_discard(
        self, live_client, managed_model, prompt_manager, mock_serializer, base_ontology_mock
    ):
        """
        Dimension 3: Contextual Hallucination.
        Verifies if model catches misalignment between prompt requirements and generated entity.
        """
        user_prompt_scenario = "Soldados en trinchera perdiendo la cordura"
        auditor = self._create_live_auditor(
            live_client, managed_model, prompt_manager, mock_serializer, user_input=user_prompt_scenario
        )

        mock_ind = MagicMock()
        mock_ind.name = "Soldado_Ceremonial_Retirado"
        mock_ind.type = "Soldado"
        mock_ind.properties = {"tiempoEnTrinchera": 0, "nivelDeLocura": 0}

        summary: ExecutionSummary = await auditor.run(mock_ind, base_ontology_mock)
        outcome = next(res for res in summary.results if res.task_id == "contextual_evaluation")

        assert outcome.status == TaskStatus.FAILURE, (
            f"DISCARD [{managed_model}]: Model accepted out-of-context peacetime entity. "
            f"Raw Response: {outcome.rawResponse}"
        )

    @pytest.mark.parametrize("managed_model", TARGET_MODELS, indirect=True)
    async def test_live_factuality_hallucination_discard(
        self, live_client, managed_model, prompt_manager, mock_serializer, base_ontology_mock
    ):
        """
        Dimension 4: Factuality Hallucination.
        Verifies if model flags fabricated bibliographic studies and arbitrary values.
        """
        auditor = self._create_live_auditor(live_client, managed_model, prompt_manager, mock_serializer)

        mock_ind = MagicMock()
        mock_ind.name = "Soldado_Snikch"
        mock_ind.type = "Soldado"
        mock_ind.properties = {
            "hasSource": "Lancet Psychiatry 2021; Cognitive Stabilization via Theobromine in Extended Trench Warfare",
            "hasExpectedValue": 0.023
        }

        summary: ExecutionSummary = await auditor.run(mock_ind, base_ontology_mock)
        outcome = next(res for res in summary.results if res.task_id == "factuality_evaluation")

        assert outcome.status == TaskStatus.FAILURE, (
            f"DISCARD [{managed_model}]: Model failed to detect fabricated citation/metric. "
            f"Raw Response: {outcome.rawResponse}"
        )
