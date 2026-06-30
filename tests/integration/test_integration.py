"""
Integration test for the ontology validation pipeline.

Philosophy: this uses the REAL classes (Pipeline, EntityOrchestrator, LLMEntityAuditor,
ConsensusResolver, RetryableLLMClient, PromptManager, GeminiClient, the pydantic
models in core.models / llm.models) without mocking their internal logic. Only the
two genuinely external boundaries of the system are replaced:

  1. Reading an OWL file from disk          -> OntologyExtractor (not provided)
  2. The network call to the LLM provider   -> google.genai SDK

This is exactly what an integration test should do: exercise the real wiring
between modules instead of reimplementing their logic with mocks.

RetryableLLMClient is treated as the correct place for retry/backoff resilience
(decorator over BaseLLMClient) and is exercised both in isolation and as part of
the full Orchestrator -> Auditor -> RetryableLLMClient -> transport chain.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from core.models import ExecutionSummary, TaskStatus
from core.pipeline.pipeline import Pipeline
from core.pipeline.orchestrator import EntityOrchestrator
from core.pipeline.evaluation.entity_auditor import EntityAuditor
from core.pipeline.evaluation.strategies.llm_auditor import LLMEntityAuditor
from core.pipeline.evaluation.strategies.majority_vote import ConsensusResolver
from core.prompt_manager import PromptManager
from core.pipeline import extractor as extractor_module

from llm.retry import RetryableLLMClient
from llm.config import RetryPolicyConfig, BackoffStrategy
from llm.clients.gemini import GeminiClient
from llm.models import LLMPayload

from exceptions import TransientNetworkException, LLMParseException

from .fakes import (
    FakeIndividual,
    FakeOntology,
    FakeSerializer,
    ScriptedLLMClient,
    FlakyLLMClient,
    AlwaysFailingLLMClient,
    make_success_response,
    make_failure_response,
    make_malformed_response,
    BASIC_PROMPTS_FIXTURE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prompts_file(tmp_path):
    path = tmp_path / "prompts.yaml"
    path.write_text(yaml.safe_dump(BASIC_PROMPTS_FIXTURE), encoding="utf-8")
    return str(path)


@pytest.fixture
def prompt_manager(prompts_file):
    return PromptManager(file_path=prompts_file)


def build_auditor(model, prompt_manager, suite_name="owl_validations"):
    return LLMEntityAuditor(
        model=model,
        prompt_manager=prompt_manager,
        serializer=FakeSerializer(),
        consensus_resolver=ConsensusResolver(),
        suite_name=suite_name,
    )


# ---------------------------------------------------------------------------
# 1) Pipeline -> Orchestrator -> Auditor (happy path, multi-individual)
# ---------------------------------------------------------------------------

async def test_pipeline_happy_path_end_to_end(monkeypatch, prompt_manager):
    individuals = [FakeIndividual("Entity_A"), FakeIndividual("Entity_B")]
    ontology = FakeOntology()

    monkeypatch.setattr(extractor_module.OntologyExtractor, "extract", staticmethod(lambda file_path: individuals))
    monkeypatch.setattr(extractor_module.OntologyExtractor, "get_base_ontology", staticmethod(lambda: ontology))

    client = ScriptedLLMClient([
        make_success_response(["consistent"]),  # Entity_A
        make_success_response(["consistent"]),  # Entity_B
    ])
    auditor = build_auditor(client, prompt_manager)
    pipeline = Pipeline(EntityOrchestrator(auditor))

    results = await pipeline.execute("fake.owl")

    assert len(results) == 2
    ids = {r.individual_id for r in results}
    assert ids == {"Entity_A", "Entity_B"}
    for r in results:
        assert isinstance(r, ExecutionSummary)
        assert r.is_successful()
        assert r.total_metrics.tokens_consumed == 100
        assert r.total_metrics.cost == pytest.approx(0.001)

    # The system prompt assembled by PromptManager actually reached the LLM client.
    assert "ONTOLOGY VALIDATION AUDITOR" in client.calls[0].system_prompt or \
           "ROLE" in client.calls[0].system_prompt
    assert "INDIVIDUAL::Entity_A" in client.calls[0].user_prompt
    assert "BASE_ONTOLOGY_CONTEXT" in client.calls[0].user_prompt


async def test_pipeline_returns_empty_list_when_no_individuals(monkeypatch, prompt_manager):
    monkeypatch.setattr(extractor_module.OntologyExtractor, "extract", staticmethod(lambda file_path: []))
    monkeypatch.setattr(extractor_module.OntologyExtractor, "get_base_ontology", staticmethod(lambda: FakeOntology()))

    client = ScriptedLLMClient([])
    auditor = build_auditor(client, prompt_manager)
    pipeline = Pipeline(EntityOrchestrator(auditor))

    results = await pipeline.execute("empty.owl")

    assert results == []
    assert client.calls == []  # the LLM must never have been invoked


# ---------------------------------------------------------------------------
# 2) EntityOrchestrator isolates per-individual failures (EntityAuditor contract)
# ---------------------------------------------------------------------------

class _ExplodingAuditor(EntityAuditor):
    """Real auditor (honors the abstract contract) that fails for a single individual."""

    def __init__(self, boom_for: str):
        self.boom_for = boom_for

    async def run(self, individual, base_ontology):
        if individual.name == self.boom_for:
            raise RuntimeError("simulated catastrophic auditor failure")
        return ExecutionSummary(
            individual_id=individual.name,
            timestamp="2026-06-30T00:00:00+00:00",
            results=[],
            total_metrics=__import__("core.models", fromlist=["ExecutionMetrics"]).ExecutionMetrics(
                duration_ms=1, cost=0.0, tokens_consumed=1
            ),
            system_summary="ok",
        )


async def test_orchestrator_isolates_failures_across_individuals():
    individuals = [FakeIndividual("Good_1"), FakeIndividual("Boom"), FakeIndividual("Good_2")]
    orchestrator = EntityOrchestrator(_ExplodingAuditor(boom_for="Boom"))

    results = await orchestrator.process(individuals, FakeOntology())

    by_id = {r.individual_id: r for r in results}
    assert len(results) == 3
    assert by_id["Good_1"].is_successful() or by_id["Good_1"].results == []
    assert by_id["Good_2"].is_successful() or by_id["Good_2"].results == []

    failed = by_id["Boom"]
    assert failed.results[0].task_id == "orchestration_error"
    assert failed.results[0].status == TaskStatus.FAILURE
    assert "simulated catastrophic auditor failure" in failed.results[0].findings[0]
    assert failed.total_metrics.cost == 0.0
    # gather() must never let the exception propagate outward
    assert all(isinstance(r, ExecutionSummary) for r in results)


# ---------------------------------------------------------------------------
# 3) LLMEntityAuditor + ConsensusResolver: tie -> fallback at temp 0.0
# ---------------------------------------------------------------------------

async def test_consensus_tie_triggers_temperature_zero_fallback(prompt_manager):
    suite = {
        "split_vote_task": {
            "prompt": "Evaluate $individual_response",
            "temperatures": [0.1, 0.2, 0.3, 0.4],
        }
    }
    prompt_manager._prompts["evaluation_suites"]["owl_validations"] = suite

    client = ScriptedLLMClient([
        make_success_response(["a"]),   # temp 0.1
        make_failure_response(["b"]),   # temp 0.2
        make_success_response(["c"]),   # temp 0.3
        make_failure_response(["d"]),   # temp 0.4  -> 2-2 tie
        make_success_response(["fallback-finding"]),  # fallback at temp 0.0
    ])
    auditor = build_auditor(client, prompt_manager)

    summary = await auditor.run(FakeIndividual("X"), FakeOntology())

    assert len(client.calls) == 5
    assert client.calls[-1].temperature == 0.0  # the fallback ran at temp 0.0
    outcome = summary.results[0]
    assert outcome.status == TaskStatus.SUCCESS
    assert outcome.findings[0] == "[CONSENSUS FAILURE: Resolved by fallback execution at temp 0.0]"
    assert "fallback-finding" in outcome.findings


async def test_consensus_without_tie_skips_fallback_call(prompt_manager):
    suite = {
        "agreement_task": {
            "prompt": "Evaluate $individual_response",
            "temperatures": [0.0, 0.3, 0.6],
        }
    }
    prompt_manager._prompts["evaluation_suites"]["owl_validations"] = suite

    client = ScriptedLLMClient([
        make_success_response(["finding-1"]),
        make_success_response(["finding-1"]),   # intentional duplicate -> must be deduped
        make_success_response(["finding-2"]),
    ])
    auditor = build_auditor(client, prompt_manager)

    summary = await auditor.run(FakeIndividual("X"), FakeOntology())

    assert len(client.calls) == 3  # no extra fallback call
    outcome = summary.results[0]
    assert outcome.status == TaskStatus.SUCCESS
    assert outcome.findings == ["finding-1", "finding-2"]  # deduped, no fallback marker


async def test_malformed_json_response_becomes_failure_outcome(prompt_manager):
    client = ScriptedLLMClient([make_malformed_response()])
    auditor = build_auditor(client, prompt_manager)

    summary = await auditor.run(FakeIndividual("X"), FakeOntology())

    assert summary.is_successful() is False
    outcome = summary.results[0]
    assert outcome.status == TaskStatus.FAILURE
    assert "Error parsing JSON response" in outcome.findings[0]


# ---------------------------------------------------------------------------
# 4) RetryableLLMClient + config.py, isolated from the rest of the stack
# ---------------------------------------------------------------------------

async def test_retry_recovers_after_transient_failures():
    flaky = FlakyLLMClient(fail_times=2, exception_factory=lambda: TransientNetworkException("timeout"))
    retryable = RetryableLLMClient(
        flaky,
        config=RetryPolicyConfig(max_retries=3, delay_between_retries=0, backoff_strategy=BackoffStrategy.FIXED),
    )

    response = await retryable.query(LLMPayload(user_prompt="hi"))

    assert flaky.call_count == 3
    assert response.raw_content == flaky.success_response.raw_content


async def test_retry_exhausts_and_propagates_domain_exception():
    flaky = AlwaysFailingLLMClient(exception_factory=lambda: LLMParseException("permanently broken"))
    retryable = RetryableLLMClient(
        flaky,
        config=RetryPolicyConfig(max_retries=2, delay_between_retries=0, backoff_strategy=BackoffStrategy.FIXED),
    )

    with pytest.raises(LLMParseException):
        await retryable.query(LLMPayload(user_prompt="hi"))

    assert flaky.call_count == 2


@pytest.mark.parametrize("strategy", [BackoffStrategy.FIXED, BackoffStrategy.EXPONENTIAL, BackoffStrategy.JITTER])
async def test_retry_all_backoff_strategies_eventually_recover(strategy):
    flaky = FlakyLLMClient(fail_times=2)
    retryable = RetryableLLMClient(
        flaky,
        config=RetryPolicyConfig(max_retries=3, delay_between_retries=0, backoff_strategy=strategy),
    )
    response = await retryable.query(LLMPayload(user_prompt="hi"))
    assert response is not None
    assert flaky.call_count == 3


# ---------------------------------------------------------------------------
# 5) Full stack: Orchestrator -> Auditor -> RetryableLLMClient exhausted
#    (proves failure isolation survives ALL layers, including retry)
# ---------------------------------------------------------------------------

async def test_full_stack_isolates_retry_exhaustion(prompt_manager):
    always_fails = AlwaysFailingLLMClient(exception_factory=lambda: TransientNetworkException("down"))
    retryable = RetryableLLMClient(
        always_fails,
        config=RetryPolicyConfig(max_retries=2, delay_between_retries=0, backoff_strategy=BackoffStrategy.FIXED),
    )
    auditor = build_auditor(retryable, prompt_manager)
    orchestrator = EntityOrchestrator(auditor)

    results = await orchestrator.process([FakeIndividual("X")], FakeOntology())

    assert len(results) == 1
    summary = results[0]
    assert summary.results[0].task_id == "orchestration_error"
    assert summary.results[0].status == TaskStatus.FAILURE
    assert always_fails.call_count == 2  # confirms it really went through retry.py before bubbling up


# ---------------------------------------------------------------------------
# 6) GeminiClient: SDK -> domain mapping, and the broken contract with retry.py
# ---------------------------------------------------------------------------

def _make_gemini_client_with_mocked_sdk():
    client = GeminiClient(api_key="test-key")
    client._client = MagicMock()
    client._client.aio.models.generate_content = AsyncMock()
    return client


async def test_gemini_client_happy_path_feeds_full_auditor_chain(prompt_manager):
    gemini = _make_gemini_client_with_mocked_sdk()
    raw_sdk_response = MagicMock()
    raw_sdk_response.text = json.dumps({"status": "success", "findings": ["looks consistent"]})
    raw_sdk_response.usage_metadata.total_token_count = 234
    gemini._client.aio.models.generate_content.return_value = raw_sdk_response

    auditor = build_auditor(gemini, prompt_manager)
    summary = await auditor.run(FakeIndividual("X"), FakeOntology())

    assert summary.is_successful()
    assert summary.total_metrics.tokens_consumed == 234
    outcome = summary.results[0]
    assert outcome.findings == ["looks consistent"]


async def test_gemini_error_path_does_not_satisfy_retry_contract():
    """
    This test documents a real incompatibility between gemini.py and retry.py.

    GeminiClient._handle_error re-raises the ORIGINAL SDK exception
    (errors.APIError / ValidationError / generic Exception); it never translates it
    into TransientNetworkException or LLMParseException. RetryableLLMClient only
    retries when it catches exactly those two domain exceptions (llm/retry.py,
    except (TransientNetworkException, LLMParseException)).

    Result: wrapping GeminiClient with RetryableLLMClient today NEVER retries on
    real Gemini API errors. The retry policy is correctly placed as a transport
    decorator, but its contract is broken because GeminiClient does not act as an
    Anti-Corruption Layer on the error path too (its own docstring only delivers
    on that promise for the success path).
    """
    gemini = _make_gemini_client_with_mocked_sdk()
    gemini._client.aio.models.generate_content.side_effect = RuntimeError("503 Service Unavailable")

    retryable = RetryableLLMClient(
        gemini,
        config=RetryPolicyConfig(max_retries=5, delay_between_retries=0, backoff_strategy=BackoffStrategy.FIXED),
    )

    with pytest.raises(RuntimeError):
        await retryable.query(LLMPayload(user_prompt="hi", model_name="gemini-2.5-flash"))

    assert gemini._client.aio.models.generate_content.call_count == 1