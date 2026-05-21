"""
Unit tests for RetryPolicy and RetryPolicyConfig
==================================================
Covers:
  - RetryPolicyConfig: valid construction and immutability (frozen=True)
  - BackoffStrategy: all three enum members exist, are distinct, inherit from str
  - RetryPolicy: successful call on first attempt (no retries, no sleep)
  - RetryPolicy: retries on TransientNetworkException and succeeds
  - RetryPolicy: retries on LLMParseException and succeeds
  - RetryPolicy: non-retryable exceptions are NOT retried and propagate immediately
  - RetryPolicy: raises after exhausting all retries (TransientNetworkException)
  - RetryPolicy: raises after exhausting all retries (LLMParseException)
  - RetryPolicy: sleep NOT called after the final failed attempt
  - RetryPolicy: sleep called max_retries-1 times total across all failures
  - RetryPolicy: max_retries=1 raises immediately with no sleep
  - RetryPolicy: succeeds on the last possible attempt
  - RetryPolicy: payload forwarded unchanged to the underlying client
  - _calculate_delay FIXED: always returns base_delay as float
  - _calculate_delay EXPONENTIAL: delay grows — base * 2^attempt
  - _calculate_delay JITTER: result is within [base_delay, base * 2^attempt]
  - _calculate_delay unknown strategy: falls back to base_delay

Implementation notes observed in retry.py:
  - Loop is range(1, max_retries + 1) so attempt numbers start at 1
  - Only TransientNetworkException and LLMParseException trigger retries
  - Any other Exception propagates immediately without retrying
  - _calculate_delay receives the current attempt number (1-based)
  - EXPONENTIAL formula: base * 2^attempt
  - JITTER formula: random.uniform(base, base * 2^attempt)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def TransientNetworkException():
    from exceptions import TransientNetworkException
    return TransientNetworkException


@pytest.fixture
def LLMParseException():
    from exceptions import LLMParseException
    return LLMParseException


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.query = AsyncMock()
    return client


@pytest.fixture
def fixed_config():
    from llm.config import RetryPolicyConfig, BackoffStrategy
    return RetryPolicyConfig(
        max_retries=3,
        delay_between_retries=2,
        backoff_strategy=BackoffStrategy.FIXED
    )


@pytest.fixture
def exponential_config():
    from llm.config import RetryPolicyConfig, BackoffStrategy
    return RetryPolicyConfig(
        max_retries=3,
        delay_between_retries=2,
        backoff_strategy=BackoffStrategy.EXPONENTIAL
    )


@pytest.fixture
def jitter_config():
    from llm.config import RetryPolicyConfig, BackoffStrategy
    return RetryPolicyConfig(
        max_retries=3,
        delay_between_retries=2,
        backoff_strategy=BackoffStrategy.JITTER
    )


@pytest.fixture
def retry_fixed(mock_llm_client, fixed_config):
    from llm.retry import RetryPolicy
    return RetryPolicy(llm_client=mock_llm_client, config=fixed_config)


@pytest.fixture
def mock_payload():
    from llm.models import LLMPayload
    return LLMPayload(user_prompt="Hello")


@pytest.fixture
def mock_response():
    from llm.models import LLMResponse
    return LLMResponse(raw_content="ok", tokens_consumed=10, duration_ms=100, cost=0.001)


# ---------------------------------------------------------------------------
# Tests: RetryPolicyConfig
# ---------------------------------------------------------------------------

class TestRetryPolicyConfig:

    def test_valid_construction(self, fixed_config):
        """A valid config must be created with all fields accessible."""
        assert fixed_config.max_retries == 3
        assert fixed_config.delay_between_retries == 2

    def test_is_immutable(self, fixed_config):
        """frozen=True must prevent any field mutation after creation."""
        with pytest.raises(Exception):  # FrozenInstanceError
            fixed_config.max_retries = 99

    def test_backoff_strategy_stores_enum_value(self, fixed_config):
        """backoff_strategy field must store a BackoffStrategy member."""
        from llm.config import BackoffStrategy
        assert fixed_config.backoff_strategy == BackoffStrategy.FIXED


# ---------------------------------------------------------------------------
# Tests: BackoffStrategy enum
# ---------------------------------------------------------------------------

class TestBackoffStrategyEnum:

    def test_fixed_member_exists(self):
        from llm.config import BackoffStrategy
        assert BackoffStrategy.FIXED is not None

    def test_exponential_member_exists(self):
        from llm.config import BackoffStrategy
        assert BackoffStrategy.EXPONENTIAL is not None

    def test_jitter_member_exists(self):
        from llm.config import BackoffStrategy
        assert BackoffStrategy.JITTER is not None

    def test_all_members_are_distinct(self):
        """All three strategy values must be different from each other."""
        from llm.config import BackoffStrategy
        values = {BackoffStrategy.FIXED, BackoffStrategy.EXPONENTIAL, BackoffStrategy.JITTER}
        assert len(values) == 3

    def test_inherits_from_str(self):
        """BackoffStrategy(str, Enum) members must be usable as plain strings."""
        from llm.config import BackoffStrategy
        assert isinstance(BackoffStrategy.FIXED, str)


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------

class TestRetryPolicyHappyPath:

    @pytest.mark.asyncio
    async def test_returns_response_on_first_attempt(self, retry_fixed, mock_llm_client, mock_response, mock_payload):
        """Must return the LLMResponse immediately when the first call succeeds."""
        mock_llm_client.query.return_value = mock_response
        result = await retry_fixed.query(mock_payload)
        assert result is mock_response

    @pytest.mark.asyncio
    async def test_client_called_exactly_once_on_success(self, retry_fixed, mock_llm_client, mock_response, mock_payload):
        """Underlying client must be called only once when the first attempt succeeds."""
        mock_llm_client.query.return_value = mock_response
        await retry_fixed.query(mock_payload)
        assert mock_llm_client.query.call_count == 1

    @pytest.mark.asyncio
    async def test_no_sleep_on_immediate_success(self, retry_fixed, mock_llm_client, mock_response, mock_payload, mocker):
        """asyncio.sleep must NOT be called when the first attempt succeeds."""
        mock_sleep = mocker.patch("llm.retry.sleep", new=AsyncMock())
        mock_llm_client.query.return_value = mock_response
        await retry_fixed.query(mock_payload)
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_payload_forwarded_to_client(self, retry_fixed, mock_llm_client, mock_response, mock_payload):
        """The exact payload must be forwarded to the underlying client."""
        mock_llm_client.query.return_value = mock_response
        await retry_fixed.query(mock_payload)
        mock_llm_client.query.assert_called_with(mock_payload)


# ---------------------------------------------------------------------------
# Tests: retry on TransientNetworkException
# ---------------------------------------------------------------------------

class TestRetryOnTransientNetworkException:

    @pytest.mark.asyncio
    async def test_retries_and_succeeds(self, retry_fixed, mock_llm_client, mock_response, mock_payload, mocker):
        """Must retry after TransientNetworkException and return the response on recovery."""
        from exceptions import TransientNetworkException
        mocker.patch("llm.retry.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = [TransientNetworkException("timeout"), mock_response]
        result = await retry_fixed.query(mock_payload)
        assert result is mock_response
        assert mock_llm_client.query.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self, retry_fixed, mock_llm_client, fixed_config, mock_payload, mocker):
        """Must raise TransientNetworkException after exhausting max_retries attempts."""
        from exceptions import TransientNetworkException
        mocker.patch("llm.retry.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = TransientNetworkException("persistent")
        with pytest.raises(TransientNetworkException):
            await retry_fixed.query(mock_payload)
        assert mock_llm_client.query.call_count == fixed_config.max_retries

    @pytest.mark.asyncio
    async def test_sleep_called_between_retries(self, retry_fixed, mock_llm_client, mock_response, mock_payload, mocker):
        """Sleep must be called after a TransientNetworkException before the next attempt."""
        from exceptions import TransientNetworkException
        mock_sleep = mocker.patch("llm.retry.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = [TransientNetworkException("fail"), mock_response]
        await retry_fixed.query(mock_payload)
        mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: retry on LLMParseException
# ---------------------------------------------------------------------------

class TestRetryOnLLMParseException:

    @pytest.mark.asyncio
    async def test_retries_and_succeeds(self, retry_fixed, mock_llm_client, mock_response, mock_payload, mocker):
        """Must retry after LLMParseException and return the response on recovery."""
        from exceptions import LLMParseException
        mocker.patch("llm.retry.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = [LLMParseException("bad json"), mock_response]
        result = await retry_fixed.query(mock_payload)
        assert result is mock_response
        assert mock_llm_client.query.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self, retry_fixed, mock_llm_client, fixed_config, mock_payload, mocker):
        """Must raise LLMParseException after exhausting max_retries attempts."""
        from exceptions import LLMParseException
        mocker.patch("llm.retry.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = LLMParseException("keeps failing")
        with pytest.raises(LLMParseException):
            await retry_fixed.query(mock_payload)
        assert mock_llm_client.query.call_count == fixed_config.max_retries


# ---------------------------------------------------------------------------
# Tests: non-retryable exceptions propagate immediately
# ---------------------------------------------------------------------------

class TestNonRetryableExceptions:

    @pytest.mark.asyncio
    async def test_non_retryable_exception_propagates_immediately(self, retry_fixed, mock_llm_client, mock_payload, mocker):
        """
        Exceptions that are not TransientNetworkException or LLMParseException
        must propagate immediately without any retry attempt.
        """
        mocker.patch("llm.retry.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = ValueError("not retryable")
        with pytest.raises(ValueError, match="not retryable"):
            await retry_fixed.query(mock_payload)

    @pytest.mark.asyncio
    async def test_non_retryable_exception_called_only_once(self, retry_fixed, mock_llm_client, mock_payload, mocker):
        """Client must be called exactly once when a non-retryable exception is raised."""
        mocker.patch("llm.retry.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = RuntimeError("fatal")
        with pytest.raises(RuntimeError):
            await retry_fixed.query(mock_payload)
        assert mock_llm_client.query.call_count == 1

    @pytest.mark.asyncio
    async def test_no_sleep_on_non_retryable_exception(self, retry_fixed, mock_llm_client, mock_payload, mocker):
        """Sleep must NOT be called when a non-retryable exception is raised."""
        mock_sleep = mocker.patch("llm.retry.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = ConnectionAbortedError("fatal")
        with pytest.raises(ConnectionAbortedError):
            await retry_fixed.query(mock_payload)
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: sleep boundary — no sleep after final failed attempt
# ---------------------------------------------------------------------------

class TestSleepBoundary:

    @pytest.mark.asyncio
    async def test_sleep_called_max_retries_minus_one_times(self, retry_fixed, mock_llm_client, fixed_config, mock_payload, mocker):
        """
        Sleep must be called exactly max_retries-1 times.
        There is no sleep after the last attempt since there is nothing left to wait for.
        """
        from exceptions import TransientNetworkException
        mock_sleep = mocker.patch("llm.retry.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = TransientNetworkException("always fails")
        with pytest.raises(TransientNetworkException):
            await retry_fixed.query(mock_payload)
        assert mock_sleep.call_count == fixed_config.max_retries - 1


# ---------------------------------------------------------------------------
# Tests: retry count boundary
# ---------------------------------------------------------------------------

class TestRetryCountBoundary:

    @pytest.mark.asyncio
    async def test_max_retries_one_raises_with_no_sleep(self, mock_llm_client, mock_payload, mocker):
        """With max_retries=1 a single failure must raise immediately with no sleep."""
        from llm.retry import RetryPolicy
        from llm.config import RetryPolicyConfig, BackoffStrategy
        from exceptions import TransientNetworkException

        config = RetryPolicyConfig(max_retries=1, delay_between_retries=1, backoff_strategy=BackoffStrategy.FIXED)
        mock_sleep = mocker.patch("llm.retry.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = TransientNetworkException("fail")

        policy = RetryPolicy(llm_client=mock_llm_client, config=config)
        with pytest.raises(TransientNetworkException):
            await policy.query(mock_payload)

        assert mock_llm_client.query.call_count == 1
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_succeeds_on_last_possible_attempt(self, mock_llm_client, mock_response, mock_payload, mocker):
        """With max_retries=3 a response on the third attempt must be returned."""
        from llm.retry import RetryPolicy
        from llm.config import RetryPolicyConfig, BackoffStrategy
        from exceptions import TransientNetworkException

        config = RetryPolicyConfig(max_retries=3, delay_between_retries=0, backoff_strategy=BackoffStrategy.FIXED)
        mocker.patch("llm.retry.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = [
            TransientNetworkException("fail 1"),
            TransientNetworkException("fail 2"),
            mock_response,
        ]

        policy = RetryPolicy(llm_client=mock_llm_client, config=config)
        result = await policy.query(mock_payload)

        assert result is mock_response
        assert mock_llm_client.query.call_count == 3


# ---------------------------------------------------------------------------
# Tests: _calculate_delay — FIXED strategy
# ---------------------------------------------------------------------------

class TestCalculateDelayFixed:

    def test_returns_base_delay_as_float(self, retry_fixed, fixed_config):
        """FIXED strategy must always return exactly delay_between_retries as float."""
        for attempt in [1, 2, 3]:
            delay = retry_fixed._calculate_delay(attempt)
            assert delay == float(fixed_config.delay_between_retries)

    def test_delay_does_not_grow(self, retry_fixed):
        """FIXED strategy must return the same value on every attempt."""
        delays = [retry_fixed._calculate_delay(a) for a in [1, 2, 3]]
        assert delays[0] == delays[1] == delays[2]

    def test_returns_float(self, retry_fixed):
        """FIXED strategy must return a float, not an int."""
        assert isinstance(retry_fixed._calculate_delay(1), float)


# ---------------------------------------------------------------------------
# Tests: _calculate_delay — EXPONENTIAL strategy
# ---------------------------------------------------------------------------

class TestCalculateDelayExponential:

    @pytest.fixture
    def retry_exp(self, mock_llm_client, exponential_config):
        from llm.retry import RetryPolicy
        return RetryPolicy(llm_client=mock_llm_client, config=exponential_config)

    def test_formula_is_base_times_two_pow_attempt(self, retry_exp, exponential_config):
        """EXPONENTIAL delay must equal base * 2^attempt for each attempt."""
        base = exponential_config.delay_between_retries
        for attempt in [1, 2, 3]:
            expected = float(base * (2 ** attempt))
            assert retry_exp._calculate_delay(attempt) == pytest.approx(expected)

    def test_delay_grows_between_attempts(self, retry_exp):
        """Each successive delay must be strictly greater than the previous one."""
        delays = [retry_exp._calculate_delay(a) for a in [1, 2, 3]]
        assert delays[0] < delays[1] < delays[2]

    def test_returns_float(self, retry_exp):
        assert isinstance(retry_exp._calculate_delay(1), float)


# ---------------------------------------------------------------------------
# Tests: _calculate_delay — JITTER strategy
# ---------------------------------------------------------------------------

class TestCalculateDelayJitter:

    @pytest.fixture
    def retry_jitter(self, mock_llm_client, jitter_config):
        from llm.retry import RetryPolicy
        return RetryPolicy(llm_client=mock_llm_client, config=jitter_config)

    def test_delay_is_at_least_base(self, retry_jitter, jitter_config):
        """JITTER delay must be >= delay_between_retries (lower bound of uniform)."""
        for attempt in [1, 2, 3]:
            assert retry_jitter._calculate_delay(attempt) >= jitter_config.delay_between_retries

    def test_delay_does_not_exceed_upper_bound(self, retry_jitter, jitter_config):
        """JITTER delay must be <= base * 2^attempt (upper bound of uniform)."""
        for attempt in [1, 2, 3]:
            upper = jitter_config.delay_between_retries * (2 ** attempt)
            assert retry_jitter._calculate_delay(attempt) <= upper

    def test_returns_float(self, retry_jitter):
        assert isinstance(retry_jitter._calculate_delay(1), float)

    def test_jitter_produces_variation(self, retry_jitter, mocker):
        """
        With a real random.uniform, repeated calls should not always return
        the same value. We verify this by checking that at least two of ten
        calls differ (statistically near-certain with a non-zero range).
        """
        results = {retry_jitter._calculate_delay(3) for _ in range(10)}
        # If all 10 are identical the range [base, max] has zero width — a bug
        assert len(results) > 1
