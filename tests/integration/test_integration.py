"""
Integration tests for the Ontology Validation Pipeline.

These tests verify the end-to-end integration of:
- Pipeline
- EntityOrchestrator
- LLMEntityAuditor
- ConsensusResolver (Majority Voting)
- RetryableLLMClient
- TurtleSerializer (using the real project ontology)

All network boundaries are simulated using a scripted LLM client to ensure
deterministic behaviors while preserving the real runtime wiring and serialization logic.
"""

import asyncio
import json
from pathlib import Path
import pytest
from datetime import datetime
from unittest.mock import MagicMock
from opentelemetry import trace
from owlready2 import get_ontology, Thing, Ontology, World
from aiolimiter import AsyncLimiter

# ============================================================================
# Global Integration Patching for Owlready2 Compatibility
# ============================================================================

# 1. Patch World.as_rdflib_graph to tolerate being called with an ontology argument
original_as_rdflib_graph = World.as_rdflib_graph

def patched_as_rdflib_graph(self, *args, **kwargs):
    """
    Wraps as_rdflib_graph to accept and discard extra arguments (e.g., ontology)
    passed by the serializer, preventing signature mismatches across owlready2 versions.
    """
    return original_as_rdflib_graph(self)

World.as_rdflib_graph = patched_as_rdflib_graph

if not hasattr(Thing, "individual_id"):
    type.__setattr__(Thing, "individual_id", property(lambda self: self.name))


from core.models import ExecutionSummary, TaskStatus, TaskOutcome, ExecutionMetrics
from core.pipeline.pipeline import Pipeline
from core.pipeline.orchestrator import EntityOrchestrator
from core.pipeline.evaluation.strategies.llm_auditor import LLMEntityAuditor
from core.pipeline.evaluation.strategies.majority_vote import ConsensusResolver
from core.prompt_manager import PromptManager
from core.pipeline import extractor as extractor_module
from serialization.serializers.turtle_serializer import TurtleSerializer
from llm.retry import RetryableLLMClient
from llm.config import RetryPolicyConfig, BackoffStrategy
from llm.models import LLMPayload, LLMResponse
from exceptions import TransientNetworkException, LLMParseException


# ============================================================================
# Test Fixtures & Real Ontology Setup
# ============================================================================

@pytest.fixture(scope="session")
def real_ontology():
    """
    Loads the real testing ontology from the filesystem.
    This guarantees the integration tests evaluate the actual schema and individuals.
    """
    current_dir = Path(__file__).parent
    ontology_path = current_dir.parent / "ontologies" / "osdi_CU1_P1_S1_M1.rdf"
    
    if not ontology_path.exists():
        ontology_path = Path("tests/ontologies/osdi_CU1_P1_S1_M1.rdf")
        
    if not ontology_path.exists():
        pytest.fail(f"Required test ontology not found at {ontology_path.absolute()}")
        
    onto = get_ontology(ontology_path.resolve().as_uri()).load()
    return onto


@pytest.fixture(scope="function")
def prompt_manager(tmp_path):
    """
    Saves a mini valid prompts.yaml into a temporary directory and loads it.
    """
    prompts_content = {
        "base_generic": {
            "role": "You are a health economist validator. Context: {base_ontology}",
            "constraints": "Strict verification output required."
        },
        "preprocessing": {
            "entity_extraction": {
                "prompt": "Extract entities from: {ai_entity}"
            }
        },
        "evaluation_suites": {
            "owl_validations": {
                "configurations": {
                    "temperatures": [0.2, 0.4], # Two branches for lightweight test
                    "allow_web_search": False
                },
                "structural_evaluation": {
                    "prompt": "Evaluate structure for: {individual_response}"
                },
                "semantic_evaluation": {
                    "prompt": "Evaluate semantics for: {individual_response}"
                }
            }
        }
    }
    
    prompts_file = tmp_path / "prompts.yaml"
    with open(prompts_file, "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(prompts_content, f)
        
    return PromptManager(file_path=str(prompts_file))


@pytest.fixture(scope="function")
def tracer():
    return trace.get_tracer("integration_test_tracer")


@pytest.fixture(scope="function")
def serializer(tracer):
    return TurtleSerializer(tracer=tracer)


@pytest.fixture(scope="function")
def rate_limiter():
    """
    Real AsyncLimiter instance (not mocked) so integration tests preserve the
    actual runtime wiring. The rate is set high enough that it never
    meaningfully throttles these tests, including the massive-concurrency one.
    """
    return AsyncLimiter(max_rate=1000, time_period=1)


# ============================================================================
# Scripted / Mocked LLM Transport Clients
# ============================================================================

class ScriptedLLMClient:
    """
    A controllable LLM client designed to simulate sequential LLM responses, 
    network exceptions, or payload-based routing.
    """
    def __init__(self, responses: list = None, router=None):
        self.responses = responses
        self.router = router
        self.call_history = []

    async def query(self, payload: LLMPayload) -> LLMResponse:
        self.call_history.append(payload)
        
        if self.router:
            resp = self.router(payload)
            if isinstance(resp, Exception):
                raise resp
            return resp
            
        if not self.responses:
            raise RuntimeError("ScriptedLLMClient has run out of predefined responses/exceptions.")
        
        next_resp = self.responses.pop(0)
        if isinstance(next_resp, Exception):
            raise next_resp
        return next_resp



def make_success_response(findings: list[str], status: str = "success") -> LLMResponse:
    """Helper to generate a structured JSON LLMResponse."""
    json_data = {
        "status": status,
        "findings": findings
    }
    return LLMResponse(
        raw_content=json.dumps(json_data),
        tokens_consumed=150,
        duration_ms=45,
        cost=0.00001
    )


# ============================================================================
# Integration Test Cases
# ============================================================================

@pytest.mark.asyncio
async def test_successful_pipeline_execution_flow(monkeypatch, real_ontology, prompt_manager, serializer, rate_limiter):
    """
    Verifies the happy path of the entire integration chain using real OWL individuals.
    """
    all_individuals = list(real_ontology.individuals())
    assert len(all_individuals) >= 2, "The real ontology must contain at least 2 individuals for this test."
    
    target_individuals = all_individuals[:2]
    ind1_name = target_individuals[0].name
    ind2_name = target_individuals[1].name

    monkeypatch.setattr(extractor_module.OntologyExtractor, "extract", staticmethod(lambda path: target_individuals))
    monkeypatch.setattr(extractor_module.OntologyExtractor, "get_base_ontology", staticmethod(lambda: real_ontology))

    def router(payload: LLMPayload):
        prompt_text = (payload.system_prompt or "") + " " + payload.user_prompt
        if ind2_name in prompt_text:
            if "Evaluate structure" in prompt_text:
                return make_success_response(["Missing mandatory relation 'has_provider'"], "failure")
            return make_success_response(["Contextually consistent"], "success")
        return make_success_response(["Clean structure"], "success")

    base_client = ScriptedLLMClient(router=router)
    retry_config = RetryPolicyConfig(max_retries=1, delay_between_retries=0, backoff_strategy=BackoffStrategy.FIXED)
    retryable_client = RetryableLLMClient(base_client, config=retry_config)
    
    consensus_resolver = ConsensusResolver()
    auditor = LLMEntityAuditor(
        model=retryable_client,
        prompt_manager=prompt_manager,
        serializer=serializer,
        consensus_resolver=consensus_resolver,
        rate_limiter=rate_limiter,
        suite_name="owl_validations"
    )
    
    orchestrator = EntityOrchestrator(strategy=auditor)
    pipeline = Pipeline(orchestrator=orchestrator)

    summaries = await pipeline.execute("fake_ontology_path.rdf")

    assert len(summaries) == 2
    
    summary_ind1 = next(s for s in summaries if s.individual_id == ind1_name)
    summary_ind2 = next(s for s in summaries if s.individual_id == ind2_name)
    
    assert summary_ind1.is_successful() is True
    assert len(summary_ind1.results) == 2
    
    assert summary_ind2.is_successful() is False
    structural_result = next(r for r in summary_ind2.results if r.task_id == "structural_evaluation")
    assert structural_result.status == TaskStatus.FAILURE
    assert "Missing mandatory relation 'has_provider'" in structural_result.findings


@pytest.mark.asyncio
async def test_transient_error_recovery_on_last_attempt(real_ontology, prompt_manager, serializer, rate_limiter):
    """
    Verifies that RetryableLLMClient transparently recovers from temporary network issues.
    """
    individual = list(real_ontology.individuals())[0]

    responses = [
        TransientNetworkException("Timeout communicating with Gemini gateway"),
        TransientNetworkException("Service Unavailable (503)"),
        make_success_response(["Clean layout"], "success")
    ]
    
    base_client = ScriptedLLMClient(responses)
    retry_config = RetryPolicyConfig(max_retries=3, delay_between_retries=0, backoff_strategy=BackoffStrategy.FIXED)
    retryable_client = RetryableLLMClient(base_client, config=retry_config)
    
    consensus_resolver = ConsensusResolver()
    prompt_manager._prompts["evaluation_suites"]["owl_validations"] = {
        "structural_evaluation": {"prompt": "Check {individual_response}", "temperatures": [0.2]}
    }

    auditor = LLMEntityAuditor(
        model=retryable_client,
        prompt_manager=prompt_manager,
        serializer=serializer,
        consensus_resolver=consensus_resolver,
        rate_limiter=rate_limiter,
        suite_name="owl_validations"
    )

    summary = await auditor.run(individual, real_ontology)
    assert len(base_client.call_history) == 3
    assert summary.is_successful() is True
    assert summary.results[0].status == TaskStatus.SUCCESS


@pytest.mark.asyncio
async def test_transient_error_exhaustion_propagates_exception(real_ontology, prompt_manager, serializer, rate_limiter):
    """
    Verifies exception propagation when retries are exhausted.
    """
    individual = list(real_ontology.individuals())[0]

    responses = [
        TransientNetworkException("API server is completely dead"),
        TransientNetworkException("API server is completely dead"),
        TransientNetworkException("API server is completely dead")
    ]
    
    base_client = ScriptedLLMClient(responses)
    retry_config = RetryPolicyConfig(max_retries=3, delay_between_retries=0, backoff_strategy=BackoffStrategy.FIXED)
    retryable_client = RetryableLLMClient(base_client, config=retry_config)
    
    consensus_resolver = ConsensusResolver()
    prompt_manager._prompts["evaluation_suites"]["owl_validations"] = {
        "configurations": {"temperatures": [0.2], "allow_web_search": False},
        "structural_evaluation": {"prompt": "Check {individual_response}"}
    }    
    auditor = LLMEntityAuditor(
        model=retryable_client,
        prompt_manager=prompt_manager,
        serializer=serializer,
        consensus_resolver=consensus_resolver,
        rate_limiter=rate_limiter,
        suite_name="owl_validations"
    )

    with pytest.raises(TransientNetworkException) as exc_info:
        await auditor.run(individual, real_ontology)
        
    assert "API server is completely dead" in str(exc_info.value)


@pytest.mark.asyncio
async def test_auditor_handles_json_decode_error_gracefully(real_ontology, prompt_manager, serializer, rate_limiter):
    """
    Verifies that malformed JSON responses are gracefully degraded to a FAILURE status.
    """
    individual = list(real_ontology.individuals())[0]

    corrupted_response = LLMResponse(
        raw_content="Sure! Here is the validation. Actually everything looks super nice, but I refuse to use JSON.",
        tokens_consumed=80,
        duration_ms=25,
        cost=0.000005
    )
    
    base_client = ScriptedLLMClient([corrupted_response])
    retry_config = RetryPolicyConfig(max_retries=1, delay_between_retries=0, backoff_strategy=BackoffStrategy.FIXED)
    retryable_client = RetryableLLMClient(base_client, config=retry_config)
    
    consensus_resolver = ConsensusResolver()
    prompt_manager._prompts["evaluation_suites"]["owl_validations"]["configurations"]["temperatures"] = [0.2]
    prompt_manager._prompts["evaluation_suites"]["owl_validations"] = {
        "configurations": {"temperatures": [0.2], "allow_web_search": False},
        "structural_evaluation": {"prompt": "Check {individual_response}"}
    }
    
    auditor = LLMEntityAuditor(
        model=retryable_client,
        prompt_manager=prompt_manager,
        serializer=serializer,
        consensus_resolver=consensus_resolver,
        rate_limiter=rate_limiter,
        suite_name="owl_validations"
    )

    summary = await auditor.run(individual, real_ontology)
    
    assert summary.is_successful() is False
    task_outcome = summary.results[0]
    assert task_outcome.status == TaskStatus.FAILURE
    assert "Error parsing JSON response" in task_outcome.findings[0]


@pytest.mark.asyncio
async def test_consensus_resolver_tie_fallback_assigned_properly(real_ontology, prompt_manager, serializer, rate_limiter):
    """
    Verifies proper handling of a voting tie in the ConsensusResolver.
    """
    individual = list(real_ontology.individuals())[0]
    call_counts = {
        "temp_0.2": 0,
        "temp_0.4": 0,
        "fallback_0.0": 0
    }

    def router(payload: LLMPayload):
        if "Evaluate structure" in payload.user_prompt:
            if payload.temperature == 0.2:
                call_counts["temp_0.2"] += 1
                return make_success_response([], "success")
            elif payload.temperature == 0.4:
                call_counts["temp_0.4"] += 1
                return make_success_response(["Property domain is incorrect"], "failure")
            elif payload.temperature == 0.0:
                call_counts["fallback_0.0"] += 1
                return make_success_response(["Tie broken: fallback default"], "failure")
        return make_success_response(["OK"], "success")

    base_client = ScriptedLLMClient(router=router)
    retry_config = RetryPolicyConfig(max_retries=1, delay_between_retries=0, backoff_strategy=BackoffStrategy.FIXED)
    retryable_client = RetryableLLMClient(base_client, config=retry_config)
    
    consensus_resolver = ConsensusResolver()
    auditor = LLMEntityAuditor(
        model=retryable_client,
        prompt_manager=prompt_manager,
        serializer=serializer,
        consensus_resolver=consensus_resolver,
        rate_limiter=rate_limiter,
        suite_name="owl_validations"
    )

    summary = await auditor.run(individual, real_ontology)
    
    structural_result = next(r for r in summary.results if r.task_id == "structural_evaluation")
    assert structural_result.status == TaskStatus.FAILURE
    assert "CONSENSUS FAILURE" in structural_result.findings[0]
    
    assert call_counts["temp_0.2"] == 1
    assert call_counts["temp_0.4"] == 1
    assert call_counts["fallback_0.0"] == 1



@pytest.mark.asyncio
async def test_consensus_discards_minority_findings(real_ontology, prompt_manager, serializer, rate_limiter):
    """
    Verifies that minority branch findings are discarded from the final TaskOutcome.
    """
    individual = list(real_ontology.individuals())[0]

    responses = [
        make_success_response(["Error A"], "failure"),
        make_success_response(["Error B"], "failure"),
        make_success_response([], "success")
    ]
    
    base_client = ScriptedLLMClient(responses)
    retry_config = RetryPolicyConfig(max_retries=1, delay_between_retries=0, backoff_strategy=BackoffStrategy.FIXED)
    retryable_client = RetryableLLMClient(base_client, config=retry_config)
    
    consensus_resolver = ConsensusResolver()
    prompt_manager._prompts["evaluation_suites"]["owl_validations"] = {
        "configurations": {"temperatures": [0.2, 0.4, 0.6], "allow_web_search": False},
        "structural_evaluation": {"prompt": "Check {individual_response}"}
    }
    
    auditor = LLMEntityAuditor(
        model=retryable_client,
        prompt_manager=prompt_manager,
        serializer=serializer,
        consensus_resolver=consensus_resolver,
        rate_limiter=rate_limiter,
        suite_name="owl_validations"
    )

    summary = await auditor.run(individual, real_ontology)
    
    assert len(summary.results) == 1
    final_result = summary.results[0]
    
    assert final_result.status == TaskStatus.FAILURE
    assert "Error A" in final_result.findings
    assert "Error B" in final_result.findings
    assert len(final_result.findings) == 2


@pytest.mark.asyncio
async def test_orchestrator_catastrophic_auditor_failure_fallback(monkeypatch, real_ontology, prompt_manager, serializer, rate_limiter):
    """
    Verifies that a catastrophic crash on a single individual is contained by the orchestrator.
    """
    all_individuals = list(real_ontology.individuals())
    assert len(all_individuals) >= 2
    
    target_individuals = all_individuals[:2]
    ind1_name = target_individuals[0].name
    ind2_name = target_individuals[1].name

    async def mocked_run(individual: Thing, base_ontology: Ontology) -> ExecutionSummary:
        if individual.name == ind1_name:
            raise ValueError(f"Catastrophic runtime crash on {ind1_name} serialization")
        
        return ExecutionSummary(
            individual_id=individual.name,
            timestamp=datetime.now().isoformat(),
            results=[TaskOutcome(task_id="structural", status=TaskStatus.SUCCESS, findings=[])],
            total_metrics=ExecutionMetrics(duration_ms=10, cost=0.0, tokens_consumed=50),
            system_summary="Processed successfully."
        )

    consensus_resolver = ConsensusResolver()
    base_client = ScriptedLLMClient([])
    retry_config = RetryPolicyConfig(max_retries=1, delay_between_retries=0, backoff_strategy=BackoffStrategy.FIXED)
    retryable_client = RetryableLLMClient(base_client, config=retry_config)
    
    auditor = LLMEntityAuditor(
        model=retryable_client,
        prompt_manager=prompt_manager,
        serializer=serializer,
        consensus_resolver=consensus_resolver,
        rate_limiter=rate_limiter,
        suite_name="owl_validations"
    )
    
    monkeypatch.setattr(auditor, "run", mocked_run)
    orchestrator = EntityOrchestrator(strategy=auditor)

    summaries = await orchestrator.process(target_individuals, real_ontology)
    
    assert len(summaries) == 2
    summary_ind1 = next(s for s in summaries if s.individual_id == ind1_name)
    summary_ind2 = next(s for s in summaries if s.individual_id == ind2_name)
    
    assert summary_ind1.is_successful() is False
    assert summary_ind1.results[0].task_id == "orchestration_error"
    assert f"Catastrophic runtime crash on {ind1_name} serialization" in summary_ind1.results[0].findings[0]
    
    assert summary_ind2.is_successful() is True
    assert summary_ind2.results[0].task_id == "structural"


@pytest.mark.asyncio
async def test_pipeline_empty_individuals_short_circuit(monkeypatch, real_ontology):
    """
    Verifies that if no new individuals are extracted, the pipeline short-circuits.
    """
    monkeypatch.setattr(extractor_module.OntologyExtractor, "extract", staticmethod(lambda path: []))
    monkeypatch.setattr(extractor_module.OntologyExtractor, "get_base_ontology", staticmethod(lambda: real_ontology))

    mock_orchestrator = MagicMock(spec=EntityOrchestrator)
    pipeline = Pipeline(orchestrator=mock_orchestrator)

    results = await pipeline.execute("empty_ontology.ttl")

    assert results == []
    mock_orchestrator.process.assert_not_called()


@pytest.mark.asyncio
async def test_extreme_concurrency_simulates_massive_load(monkeypatch, real_ontology, prompt_manager, serializer, rate_limiter):
    """
    PROVES/WARNING: The Orchestrator's `asyncio.gather(*tasks)` design is unbounded.
    """
    MASSIVE_AMOUNT = 200 
    
    class SimpleFakeThing:
        def __init__(self, name):
            self.name = name
            self.iri = f"http://example.org/{name}"
            
        @property
        def individual_id(self):
            return self.name
            
    individuals = [SimpleFakeThing(f"Entity_{i}") for i in range(MASSIVE_AMOUNT)]
    
    monkeypatch.setattr(extractor_module.OntologyExtractor, "extract", staticmethod(lambda x: individuals))
    monkeypatch.setattr(extractor_module.OntologyExtractor, "get_base_ontology", staticmethod(lambda: real_ontology))
        
    client = ScriptedLLMClient(router=lambda p: make_success_response(["ok"]))
    retry_config = RetryPolicyConfig(max_retries=1, delay_between_retries=0, backoff_strategy=BackoffStrategy.FIXED)
    retryable_client = RetryableLLMClient(client, config=retry_config)
    
    monkeypatch.setattr(serializer, "process_individual", lambda ind: f"<http://example.org/{ind.name}> a owl:Thing .")
    
    consensus_resolver = ConsensusResolver()
    auditor = LLMEntityAuditor(
        model=retryable_client,
        prompt_manager=prompt_manager,
        serializer=serializer,
        consensus_resolver=consensus_resolver,
        rate_limiter=rate_limiter,
        suite_name="owl_validations"
    )
    
    pipeline = Pipeline(EntityOrchestrator(auditor))

    results = await asyncio.wait_for(pipeline.execute("massive_batch.owl"), timeout=15.0)
    
    assert len(results) == MASSIVE_AMOUNT
    assert results[0].is_successful() is True