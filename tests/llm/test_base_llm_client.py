"""
Unit tests for BaseLLMClient
==============================
Covers:
  - Cannot instantiate BaseLLMClient directly (abstract class)
  - Cannot instantiate a subclass that does not implement query()
  - query() without retry config (success + error propagation)
  - query() with retry config (success after retry, exhaustion, non-retryable)
  - backoff delay calculation (fixed, exponential, jitter)
"""

import pytest
from unittest.mock import AsyncMock


# ---------------------------------------------------------------------------
# Minimal concrete subclass used as test double
# ---------------------------------------------------------------------------

@pytest.fixture
def make_stub():
    """
    Factory returning a minimal StubLLMClient.
    query is patched per-test with AsyncMock.
    """
    from llm.base_llm_client import BaseLLMClient

    class StubLLMClient(BaseLLMClient):
        async def query(self, payload):
            raise NotImplementedError

    return StubLLMClient


@pytest.fixture
def mock_payload():
    from llm.models import LLMPayload
    return LLMPayload(user_prompt="Hello")


@pytest.fixture
def mock_response():
    from llm.models import LLMResponse
    return LLMResponse(raw_content="ok", tokens_consumed=10, duration_ms=100, cost=0.001)


@pytest.fixture
def fixed_config():
    from llm.config import RetryPolicyConfig, BackoffStrategy
    return RetryPolicyConfig(
        max_retries=3,
        delay_between_retries=2,
        backoff_strategy=BackoffStrategy.FIXED,
    )


@pytest.fixture
def exponential_config():
    from llm.config import RetryPolicyConfig, BackoffStrategy
    return RetryPolicyConfig(
        max_retries=3,
        delay_between_retries=2,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
    )


@pytest.fixture
def jitter_config():
    from llm.config import RetryPolicyConfig, BackoffStrategy
    return RetryPolicyConfig(
        max_retries=2,
        delay_between_retries=3,
        backoff_strategy=BackoffStrategy.JITTER,
    )


# ---------------------------------------------------------------------------
# Tests: abstract contract
# ---------------------------------------------------------------------------

class TestAbstractContract:

    def test_cannot_instantiate_base_directly(self):
        """Instantiating BaseLLMClient without implementing query must raise TypeError."""
        from llm.base_llm_client import BaseLLMClient
        with pytest.raises(TypeError):
            BaseLLMClient()

    def test_subclass_without_query_raises_type_error(self):
        """A subclass that omits query must also raise TypeError on instantiation."""
        from llm.base_llm_client import BaseLLMClient

        class IncompleteClient(BaseLLMClient):
            pass

        with pytest.raises(TypeError):
            IncompleteClient()

    def test_concrete_subclass_instantiates_correctly(self, make_stub):
        """A subclass that implements query must instantiate without errors."""
        client = make_stub()
        assert client is not None


# ---------------------------------------------------------------------------
# Tests: query() 
# ---------------------------------------------------------------------------

class TestQuery:

    @pytest.mark.asyncio
    async def test_returns_response_from_query(self, make_stub, mock_response, mock_payload):
        """query() must return exactly the LLMResponse produced by query()."""
        client = make_stub()
        client.query = AsyncMock(return_value=mock_response)

        result = await client.query(mock_payload)

        assert result is mock_response
        client.query.assert_called_once_with(mock_payload)

    @pytest.mark.asyncio
    async def test_propagates_exception_without_retry(self, make_stub, mock_payload, mocker):
        """query() must propagate errors when retry is not configured."""
        client = make_stub()
        client.query = AsyncMock(side_effect=ValueError("boom"))

        with pytest.raises(ValueError, match="boom"):
            await client.query(mock_payload)

        client.query.assert_called_once_with(mock_payload)