"""
Unit tests for PureLLMStrategy and base validation orchestration.
"""

import json
import pytest
from unittest.mock import AsyncMock

from core.models import PromptRegistry, TaskStatus
from llm.base_llm_client import BaseLLMClient
from llm.models import LLMResponse
from exceptions import LLMParseException


class StubLLMClient(BaseLLMClient):
    def __init__(self, side_effects):
        self._query_mock = AsyncMock(side_effect=side_effects)

    async def _query(self, payload):
        return await self._query_mock(payload)


def _make_response(findings, *, duration_ms=10, cost=0.01, tokens=5):
    payload = {"findings": findings}
    return LLMResponse(
        raw_content=json.dumps(payload),
        tokens_consumed=tokens,
        duration_ms=duration_ms,
        cost=cost,
    )


@pytest.fixture
def prompt_registry():
    return PromptRegistry(
        loaded_prompts={
            "id_1": "Task 1: {owl_entity}",
            "id_2": "Task 2: {owl_entity}",
            "id_3": "Task 3: {owl_entity}",
            "id_4": "Task 4: {owl_entity}",
        }
    )


@pytest.mark.asyncio
async def test_runs_four_tasks_in_order(prompt_registry):
    from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy

    responses = [
        _make_response(["a1"]),
        _make_response(["b1"]),
        _make_response(["c1"]),
        _make_response(["d1"]),
    ]
    client = StubLLMClient(responses)
    strategy = PureLLMStrategy(client, prompt_registry)

    summary = await strategy.evaluate("ENTITY")

    assert [result.task_id for result in summary.results] == ["id_1", "id_2", "id_3", "id_4"]
    assert all(result.status == TaskStatus.SUCCESS for result in summary.results)
    assert client._query_mock.call_count == 4

    prompts = [call.args[0].user_prompt for call in client._query_mock.call_args_list]
    assert prompts == [
        "Task 1: ENTITY",
        "Task 2: ENTITY",
        "Task 3: ENTITY",
        "Task 4: ENTITY",
    ]


@pytest.mark.asyncio
async def test_aggregates_metrics(prompt_registry):
    from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy

    responses = [
        _make_response(["a"], duration_ms=10, cost=0.1, tokens=5),
        _make_response(["b"], duration_ms=20, cost=0.2, tokens=6),
        _make_response(["c"], duration_ms=30, cost=0.3, tokens=7),
        _make_response(["d"], duration_ms=40, cost=0.4, tokens=8),
    ]
    client = StubLLMClient(responses)
    strategy = PureLLMStrategy(client, prompt_registry)

    summary = await strategy.evaluate("ENTITY")

    assert summary.total_metrics.duration_ms == 100
    assert summary.total_metrics.cost == pytest.approx(1.0)
    assert summary.total_metrics.tokens_consumed == 26


@pytest.mark.asyncio
async def test_missing_prompt_skips_task(prompt_registry):
    from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy

    registry = PromptRegistry(loaded_prompts={"id_1": "Task 1: {owl_entity}"})
    responses = [_make_response(["a1"])]
    client = StubLLMClient(responses)
    strategy = PureLLMStrategy(client, registry)

    summary = await strategy.evaluate("ENTITY")

    assert [result.task_id for result in summary.results] == ["id_1", "id_2", "id_3", "id_4"]
    statuses = {result.task_id: result.status for result in summary.results}
    assert statuses["id_1"] == TaskStatus.SUCCESS
    assert statuses["id_2"] == TaskStatus.SKIPPED
    assert statuses["id_3"] == TaskStatus.SKIPPED
    assert statuses["id_4"] == TaskStatus.SKIPPED
    assert client._query_mock.call_count == 1


@pytest.mark.asyncio
async def test_parses_findings_from_json(prompt_registry):
    from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy

    responses = [
        _make_response(["f1", "f2"]),
        _make_response(["g1"]),
        _make_response(["h1"]),
        _make_response(["i1"]),
    ]
    client = StubLLMClient(responses)
    strategy = PureLLMStrategy(client, prompt_registry)

    summary = await strategy.evaluate("ENTITY")

    assert summary.results[0].findings == ["f1", "f2"]


@pytest.mark.asyncio
async def test_error_marks_failure_and_continues(prompt_registry):
    from entity_validation.strategies.pure_llm_strategy import PureLLMStrategy

    responses = [
        _make_response(["a1"]),
        LLMParseException("bad json"),
        _make_response(["c1"]),
        _make_response(["d1"]),
    ]
    client = StubLLMClient(responses)
    strategy = PureLLMStrategy(client, prompt_registry)

    summary = await strategy.evaluate("ENTITY")

    statuses = [result.status for result in summary.results]
    assert statuses == [
        TaskStatus.SUCCESS,
        TaskStatus.FAILURE,
        TaskStatus.SUCCESS,
        TaskStatus.SUCCESS,
    ]
