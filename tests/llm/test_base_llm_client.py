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
# Tests: query() without retry config
# ---------------------------------------------------------------------------

class TestQueryNoRetry:

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
        mock_sleep = mocker.patch("llm.base_llm_client.sleep", new=AsyncMock())

        with pytest.raises(ValueError, match="boom"):
            await client.query(mock_payload)

        mock_sleep.assert_not_called()
        client.query.assert_called_once_with(mock_payload)


# ---------------------------------------------------------------------------
# Tests: query() with retry config
# ---------------------------------------------------------------------------

class TestQueryWithRetry:

    @pytest.mark.asyncio
    async def test_retries_on_transient_then_succeeds(self, make_stub, mock_response, mock_payload, fixed_config, mocker):
        """Must retry TransientNetworkException and return response on recovery."""
        from exceptions import TransientNetworkException

        client = make_stub()
        client.set_retry_config(fixed_config)
        client.query = AsyncMock(side_effect=[TransientNetworkException("timeout"), mock_response])
        mock_sleep = mocker.patch("llm.base_llm_client.sleep", new=AsyncMock())

        result = await client.query(mock_payload)

        assert result is mock_response
        assert client.query.call_count == 2
        mock_sleep.assert_called_once_with(fixed_config.delay_between_retries)

    @pytest.mark.asyncio
    async def test_retries_on_parse_exception_then_succeeds(self, make_stub, mock_response, mock_payload, fixed_config, mocker):
        """Must retry LLMParseException and return response on recovery."""
        from exceptions import LLMParseException

        client = make_stub()
        client.set_retry_config(fixed_config)
        client.query = AsyncMock(side_effect=[LLMParseException("bad json"), mock_response])
        mock_sleep = mocker.patch("llm.base_llm_client.sleep", new=AsyncMock())

        result = await client.query(mock_payload)

        assert result is mock_response
        assert client.query.call_count == 2
        mock_sleep.assert_called_once_with(fixed_config.delay_between_retries)

    @pytest.mark.asyncio
    async def test_exhausts_retries_and_raises_last_exception(self, make_stub, mock_payload, fixed_config, mocker):
        """Must raise the last transient error after exhausting retries."""
        from exceptions import TransientNetworkException

        client = make_stub()
        client.set_retry_config(fixed_config)
        client.query = AsyncMock(side_effect=[
            TransientNetworkException("fail 1"),
            TransientNetworkException("fail 2"),
            TransientNetworkException("fail 3"),
        ])
        mock_sleep = mocker.patch("llm.base_llm_client.sleep", new=AsyncMock())

        with pytest.raises(TransientNetworkException):
            await client.query(mock_payload)

        assert client.query.call_count == fixed_config.max_retries
        assert mock_sleep.call_count == fixed_config.max_retries - 1

    @pytest.mark.asyncio
    async def test_non_retryable_error_bubbles_immediately(self, make_stub, mock_payload, fixed_config, mocker):
        """Non-retryable exceptions must be raised without sleeping."""
        client = make_stub()
        client.set_retry_config(fixed_config)
        client.query = AsyncMock(side_effect=RuntimeError("nope"))
        mock_sleep = mocker.patch("llm.base_llm_client.sleep", new=AsyncMock())

        with pytest.raises(RuntimeError, match="nope"):
            await client.query(mock_payload)

        mock_sleep.assert_not_called()
        client.query.assert_called_once_with(mock_payload)


# ---------------------------------------------------------------------------
# Tests: backoff delay calculation
# ---------------------------------------------------------------------------

class TestBackoffCalculation:

    def test_fixed_delay(self, make_stub, fixed_config):
        client = make_stub()
        delay = client._calculate_retry_delay(fixed_config, attempt=1)
        assert delay == float(fixed_config.delay_between_retries)

    def test_exponential_delay_grows(self, make_stub, exponential_config):
        client = make_stub()
        first = client._calculate_retry_delay(exponential_config, attempt=0)
        second = client._calculate_retry_delay(exponential_config, attempt=1)
        assert second > first

    def test_jitter_uses_random_bounds(self, make_stub, jitter_config, mocker):
        client = make_stub()
        mock_uniform = mocker.patch("llm.base_llm_client.random.uniform", return_value=5.5)

        delay = client._calculate_retry_delay(jitter_config, attempt=1)

        mock_uniform.assert_called_once_with(
            jitter_config.delay_between_retries,
            jitter_config.delay_between_retries * (2 ** 1),
        )
        assert delay == 5.5
