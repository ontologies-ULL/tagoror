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
import asyncio
import json
import trace
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
from llm.models import LLMPayload, LLMResponse

from exceptions import TransientNetworkException, LLMParseException
from serialization.serializers.turtle_serializer import TurtleSerializer

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
    failed = by_id["Boom"]
    assert failed.results[0].task_id == "orchestration_error"
    assert failed.results[0].status == TaskStatus.FAILURE
    assert "simulated catastrophic auditor failure" in failed.results[0].findings[0]


# ---------------------------------------------------------------------------
# 3) LLMEntityAuditor + ConsensusResolver
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
        make_success_response(["a"]),   
        make_failure_response(["b"]),   
        make_success_response(["c"]),   
        make_failure_response(["d"]),   # 2-2 tie
        make_success_response(["fallback-finding"]),  
    ])
    auditor = build_auditor(client, prompt_manager)
    summary = await auditor.run(FakeIndividual("X"), FakeOntology())

    assert len(client.calls) == 5
    assert client.calls[-1].temperature == 0.0  
    outcome = summary.results[0]
    assert "[CONSENSUS FAILURE: Resolved by fallback execution at temp 0.0]" in outcome.findings[0]


# ---------------------------------------------------------------------------
# 4) RetryableLLMClient + config.py
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


# ---------------------------------------------------------------------------
# 5) Full stack: Orchestrator -> Auditor -> RetryableLLMClient exhausted
# ---------------------------------------------------------------------------

async def test_full_stack_isolates_retry_exhaustion(prompt_manager):
    always_fails = AlwaysFailingLLMClient(exception_factory=lambda: TransientNetworkException("down"))
    retryable = RetryableLLMClient(always_fails, config=RetryPolicyConfig(max_retries=2, delay_between_retries=0))
    auditor = build_auditor(retryable, prompt_manager)
    orchestrator = EntityOrchestrator(auditor)

    results = await orchestrator.process([FakeIndividual("X")], FakeOntology())

    assert len(results) == 1
    assert results[0].results[0].task_id == "orchestration_error"
    assert always_fails.call_count == 2 


# ---------------------------------------------------------------------------
# 6) GeminiClient: SDK -> domain mapping
# ---------------------------------------------------------------------------

def _make_gemini_client_with_mocked_sdk():
    client = GeminiClient(api_key="test-key")
    client._client = MagicMock()
    client._client.aio.models.generate_content = AsyncMock()
    return client

async def test_gemini_error_path_does_not_satisfy_retry_contract():
    gemini = _make_gemini_client_with_mocked_sdk()
    gemini._client.aio.models.generate_content.side_effect = RuntimeError("503")
    retryable = RetryableLLMClient(gemini, config=RetryPolicyConfig(max_retries=5, delay_between_retries=0))

    with pytest.raises(RuntimeError):
        await retryable.query(LLMPayload(user_prompt="hi", model_name="gemini-2.5-flash"))

    assert gemini._client.aio.models.generate_content.call_count == 1


# ---------------------------------------------------------------------------
# 7) PureLLMStrategy 
# ---------------------------------------------------------------------------

async def test_pure_llm_strategy_aggregates_metrics_across_prompts(raw_prompts_dict):
    client = ScriptedLLMClient([
        make_success_response(["finding 1"]), 
        make_success_response(["finding 2"]), 
    ])
    
    strategy = PureLLMStrategy(llm_client=client, context=raw_prompts_dict)
    summary = await strategy.evaluate("Serialized_OWL_Entity_String")

    assert len(summary.results) == 2
    assert client.call_count == 2
    assert summary.total_metrics.tokens_consumed == 200 
    assert summary.total_metrics.cost == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# 8) TurtleSerializer (Architectural flaw demonstration)
# ---------------------------------------------------------------------------

def test_turtle_serializer_requires_real_owlready_objects():
    tracer = trace.get_tracer(__name__)
    serializer = TurtleSerializer(tracer=tracer)

    with pytest.raises(AttributeError):
        serializer.process_ontology(FakeOntology())


# ---------------------------------------------------------------------------
# 9) EXTREME EDGE CASES (Exposing architectural gaps)
# ---------------------------------------------------------------------------

async def test_pipeline_handles_extractor_catastrophic_failure(monkeypatch, prompt_manager):
    """
    PROVES: The Pipeline does NOT isolate catastrophic failures during the extraction phase.
    If the OWL file is violently malformed, the pipeline blows up and drops all processing,
    propagating the error upwards rather than returning a failed execution summary.
    """
    def broken_extractor(*args, **kwargs):
        raise ValueError("CRITICAL: OWL File binary corruption")

    monkeypatch.setattr(extractor_module.OntologyExtractor, "extract", staticmethod(broken_extractor))
    
    client = ScriptedLLMClient([])
    auditor = build_auditor(client, prompt_manager)
    pipeline = Pipeline(EntityOrchestrator(auditor))

    with pytest.raises(ValueError, match="CRITICAL: OWL File binary corruption"):
        await pipeline.execute("corrupted_file.owl")


async def test_llm_auditor_handles_completely_empty_response(prompt_manager):
    """
    PROVES: Robustness against blank/empty network responses. 
    The parser must fall back to JSONDecodeError and mark it as a FAILURE 
    without crashing the orchestrator thread.
    """
    empty_response = LLMResponse(raw_content="", tokens_consumed=0, duration_ms=100, cost=0.0)
    client = ScriptedLLMClient([empty_response])
    auditor = build_auditor(client, prompt_manager)

    summary = await auditor.run(FakeIndividual("X"), FakeOntology())
    
    assert not summary.is_successful()
    assert summary.results[0].status == TaskStatus.FAILURE
    assert "Error parsing JSON response" in summary.results[0].findings[0]


async def test_extreme_concurrency_simulates_massive_load(monkeypatch, prompt_manager):
    """
    PROVES/WARNING: The Orchestrator's `asyncio.gather(*tasks)` design is unbounded.
    While this test artificially passes here due to the mock running in RAM, 
    in production sending 5,000 tasks concurrently to Gemini will instantly 
    trigger a 429 Too Many Requests or exhaust local memory.
    """
    MASSIVE_AMOUNT = 5000
    individuals = [FakeIndividual(f"Entity_{i}") for i in range(MASSIVE_AMOUNT)]
    
    monkeypatch.setattr(extractor_module.OntologyExtractor, "extract", staticmethod(lambda x: individuals))
    monkeypatch.setattr(extractor_module.OntologyExtractor, "get_base_ontology", staticmethod(lambda: FakeOntology()))

    # Simulating 5000 identical fast responses
    responses = [make_success_response(["ok"]) for _ in range(MASSIVE_AMOUNT)]
    client = ScriptedLLMClient(responses)
    
    # We must patch the serializer to be fast and not crash the CPU for 5k fake objects
    auditor = build_auditor(client, prompt_manager)
    pipeline = Pipeline(EntityOrchestrator(auditor))

    # If this takes too long or crashes, it proves the unbounded gather() is a critical flaw.
    results = await asyncio.wait_for(pipeline.execute("massive.owl"), timeout=5.0)

    assert len(results) == MASSIVE_AMOUNT
    assert client.call_count == MASSIVE_AMOUNT