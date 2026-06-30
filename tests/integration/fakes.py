"""
Test doubles for the integration test suite.

Only the genuinely EXTERNAL boundaries of the system are mocked/faked:
- the contents of an OWL file on disk (OntologyExtractor)
- the raw response from an LLM provider (what would normally travel over the network)

Everything else (Pipeline, EntityOrchestrator, LLMEntityAuditor, ConsensusResolver,
RetryableLLMClient, PromptManager, the pydantic models) is the real uploaded code,
unmocked, so the test verifies actual interaction between modules.
"""
import json
from dataclasses import dataclass, field
from typing import Callable

from serialization.base_serializer import BaseSerializer
from llm.base_llm_client import BaseLLMClient
from llm.models import LLMPayload, LLMResponse
from exceptions import TransientNetworkException, LLMParseException


@dataclass
class FakeIndividual:
    """Stands in for owlready2.Thing. Only carries the attributes the real code uses."""
    name: str

    @property
    def individual_id(self) -> str:
        return self.name


class FakeOntology:
    """Stands in for owlready2.Ontology. Opaque object for the serializer."""
    pass


class FakeSerializer(BaseSerializer):
    """Deterministic serializer with no dependency on a real owlready2 instance."""

    def process_individual(self, individual: FakeIndividual) -> str:
        return f"INDIVIDUAL::{individual.name}"

    def process_ontology(self, ontology: FakeOntology) -> str:
        return "BASE_ONTOLOGY_CONTEXT"


def make_success_response(findings: list[str], tokens: int = 100, cost: float = 0.001, duration_ms: int = 10) -> LLMResponse:
    return LLMResponse(
        raw_content=json.dumps({"status": "success", "findings": findings}),
        tokens_consumed=tokens,
        duration_ms=duration_ms,
        cost=cost,
    )


def make_failure_response(findings: list[str], tokens: int = 50, cost: float = 0.0005, duration_ms: int = 5) -> LLMResponse:
    return LLMResponse(
        raw_content=json.dumps({"status": "failure", "findings": findings}),
        tokens_consumed=tokens,
        duration_ms=duration_ms,
        cost=cost,
    )


def make_malformed_response() -> LLMResponse:
    return LLMResponse(raw_content="not-valid-json{{{", tokens_consumed=10, duration_ms=1, cost=0.0001)


class ScriptedLLMClient(BaseLLMClient):
    """
    Real implementation of BaseLLMClient that returns scripted responses in call
    order. Lets us simulate consensus scenarios (split votes, fallback to
    temp 0.0) without depending on a real provider.
    """

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[LLMPayload] = []

    async def query(self, payload: LLMPayload) -> LLMResponse:
        self.calls.append(payload)
        if not self._responses:
            raise AssertionError("ScriptedLLMClient ran out of scripted responses")
        return self._responses.pop(0)


@dataclass
class FlakyLLMClient(BaseLLMClient):
    """
    Real implementation of BaseLLMClient that fails the first N calls with a
    domain exception (TransientNetworkException / LLMParseException) and then
    succeeds. Used to test retry.py in isolation from the rest of the stack.
    """
    fail_times: int
    exception_factory: Callable[[], Exception] = lambda: TransientNetworkException("simulated transient error")
    success_response: LLMResponse = field(default_factory=lambda: make_success_response(["ok"]))
    call_count: int = 0

    async def query(self, payload: LLMPayload) -> LLMResponse:
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise self.exception_factory()
        return self.success_response


@dataclass
class AlwaysFailingLLMClient(BaseLLMClient):
    """Real implementation of BaseLLMClient that always fails. Used to test retry exhaustion."""
    exception_factory: Callable[[], Exception] = lambda: LLMParseException("permanently broken")
    call_count: int = 0

    async def query(self, payload: LLMPayload) -> LLMResponse:
        self.call_count += 1
        raise self.exception_factory()


BASIC_PROMPTS_FIXTURE = {
    "base_generic": {
        "role": "You are an ontology validation auditor.",
        "constraints": "Always answer in strict JSON with 'status' and 'findings'.",
    },
    "evaluation_suites": {
        "owl_validations": {
            "consistency_check": {
                "prompt": "Validate consistency of $individual_response against $base_ontology",
                "temperatures": [0.0],
            },
        }
    },
}