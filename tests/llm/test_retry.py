"""
Unit tests for RetryPolicy and RetryPolicyConfig
==================================================
Covers:
  - RetryPolicyConfig: valid construction, immutability (frozen), invalid types
  - BackoffStrategy enum: all three values (FIXED, EXPONENTIAL, JITTER)
  - RetryPolicy: successful call on first attempt
  - RetryPolicy: successful call after one or more transient failures
  - RetryPolicy: exhausting all retries raises the original exception
  - RetryPolicy: TransientNetworkError triggers a retry
  - RetryPolicy: LLMParseException triggers a retry
  - RetryPolicy: sleep called with correct delay (fixed strategy)
  - RetryPolicy: sleep delay grows exponentially (exponential strategy)
  - RetryPolicy: sleep delay is within valid bounds (jitter strategy)
  - RetryPolicy: sleep NOT called after the final failed attempt
  - RetryPolicy: exact retry count boundary (max_retries=1)
  - RetryPolicy: succeeds on last possible attempt
  - RetryPolicy: payload forwarded unchanged to the underlying client
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_client():
    """Mocked BaseLLMClient — query is async."""
    client = MagicMock()
    client.query = AsyncMock()
    return client


@pytest.fixture
def fixed_config():
    """
    RetryPolicyConfig with 3 retries, 2-second delay, FIXED backoff.
    Uses the real dataclass to also exercise its construction.
    """
    from llm.config import RetryPolicyConfig, BackoffStrategy
    return RetryPolicyConfig(
        max_retries=3,
        delay_between_retries=2,
        backoff_strategy=BackoffStrategy.FIXED
    )


@pytest.fixture
def exponential_config():
    """RetryPolicyConfig with EXPONENTIAL backoff."""
    from llm.config import RetryPolicyConfig, BackoffStrategy
    return RetryPolicyConfig(
        max_retries=3,
        delay_between_retries=2,
        backoff_strategy=BackoffStrategy.EXPONENTIAL
    )


@pytest.fixture
def jitter_config():
    """RetryPolicyConfig with JITTER backoff."""
    from llm.config import RetryPolicyConfig, BackoffStrategy
    return RetryPolicyConfig(
        max_retries=3,
        delay_between_retries=2,
        backoff_strategy=BackoffStrategy.JITTER
    )


@pytest.fixture
def retry_policy_fixed(mock_llm_client, fixed_config):
    """RetryPolicy instance using FIXED backoff."""
    from llm.retry_policy import RetryPolicy
    return RetryPolicy(llm_client=mock_llm_client, config=fixed_config)


@pytest.fixture
def mock_response():
    """Minimal valid LLMResponse."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Tests: RetryPolicyConfig construction
# ---------------------------------------------------------------------------

class TestRetryPolicyConfig:

    def test_valid_construction(self, fixed_config):
        """A valid config must be created without errors."""
        assert fixed_config.max_retries == 3
        assert fixed_config.delay_between_retries == 2

    def test_is_immutable(self, fixed_config):
        """frozen=True must prevent any field mutation."""
        with pytest.raises(Exception):  # FrozenInstanceError
            fixed_config.max_retries = 99

    def test_backoff_strategy_is_enum(self, fixed_config):
        """backoff_strategy must be a BackoffStrategy enum member, not a raw string."""
        from llm.config import BackoffStrategy
        assert isinstance(fixed_config.backoff_strategy, BackoffStrategy)


# ---------------------------------------------------------------------------
# Tests: BackoffStrategy enum
# ---------------------------------------------------------------------------

class TestBackoffStrategyEnum:

    def test_fixed_member_exists(self):
        """BackoffStrategy must have a FIXED member."""
        from llm.config import BackoffStrategy
        assert hasattr(BackoffStrategy, "FIXED")

    def test_exponential_member_exists(self):
        """BackoffStrategy must have an EXPONENTIAL member."""
        from llm.config import BackoffStrategy
        assert hasattr(BackoffStrategy, "EXPONENTIAL")

    def test_jitter_member_exists(self):
        """BackoffStrategy must have a JITTER member."""
        from llm.config import BackoffStrategy
        assert hasattr(BackoffStrategy, "JITTER")

    def test_members_are_distinct(self):
        """All three strategies must have different values."""
        from llm.config import BackoffStrategy
        values = {BackoffStrategy.FIXED, BackoffStrategy.EXPONENTIAL, BackoffStrategy.JITTER}
        assert len(values) == 3


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------

class TestRetryPolicyHappyPath:

    @pytest.mark.asyncio
    async def test_returns_response_on_first_attempt(self, retry_policy_fixed, mock_llm_client, mock_response):
        """Must return the LLMResponse immediately when the first call succeeds."""
        mock_llm_client.query.return_value = mock_response

        result = await retry_policy_fixed.query(MagicMock())

        assert result is mock_response

    @pytest.mark.asyncio
    async def test_client_called_exactly_once_on_success(self, retry_policy_fixed, mock_llm_client, mock_response):
        """When the first call succeeds, the underlying client must be called only once."""
        mock_llm_client.query.return_value = mock_response

        await retry_policy_fixed.query(MagicMock())

        assert mock_llm_client.query.call_count == 1

    @pytest.mark.asyncio
    async def test_no_sleep_on_immediate_success(self, retry_policy_fixed, mock_llm_client, mock_response, mocker):
        """asyncio.sleep must NOT be called if the first attempt succeeds."""
        mock_sleep = mocker.patch("llm.retry_policy.sleep", new=AsyncMock())
        mock_llm_client.query.return_value = mock_response

        await retry_policy_fixed.query(MagicMock())

        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: retry on TransientNetworkError
# ---------------------------------------------------------------------------

class TestRetryOnTransientNetworkError:

    @pytest.mark.asyncio
    async def test_retries_and_eventually_succeeds(self, retry_policy_fixed, mock_llm_client, mock_response, mocker):
        """Must retry on TransientNetworkError and return the response when it succeeds."""
        from llm.exceptions import TransientNetworkError
        mocker.patch("llm.retry_policy.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = [TransientNetworkError("timeout"), mock_response]

        result = await retry_policy_fixed.query(MagicMock())

        assert result is mock_response
        assert mock_llm_client.query.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_all_retries(self, retry_policy_fixed, mock_llm_client, fixed_config, mocker):
        """Must raise TransientNetworkError after exhausting all retries."""
        from llm.exceptions import TransientNetworkError
        mocker.patch("llm.retry_policy.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = TransientNetworkError("persistent")

        with pytest.raises(TransientNetworkError):
            await retry_policy_fixed.query(MagicMock())

        assert mock_llm_client.query.call_count == fixed_config.max_retries


# ---------------------------------------------------------------------------
# Tests: retry on LLMParseException
# ---------------------------------------------------------------------------

class TestRetryOnLLMParseException:

    @pytest.mark.asyncio
    async def test_retries_and_eventually_succeeds(self, retry_policy_fixed, mock_llm_client, mock_response, mocker):
        """Must retry on LLMParseException and return the response when the LLM recovers."""
        from llm.exceptions import LLMParseException
        mocker.patch("llm.retry_policy.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = [LLMParseException("bad JSON"), mock_response]

        result = await retry_policy_fixed.query(MagicMock())

        assert result is mock_response
        assert mock_llm_client.query.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_all_retries(self, retry_policy_fixed, mock_llm_client, fixed_config, mocker):
        """Must raise LLMParseException after exhausting all retries."""
        from llm.exceptions import LLMParseException
        mocker.patch("llm.retry_policy.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = LLMParseException("keeps failing")

        with pytest.raises(LLMParseException):
            await retry_policy_fixed.query(MagicMock())

        assert mock_llm_client.query.call_count == fixed_config.max_retries


# ---------------------------------------------------------------------------
# Tests: FIXED backoff — sleep delay is always constant
# ---------------------------------------------------------------------------

class TestFixedBackoff:

    @pytest.mark.asyncio
    async def test_sleep_uses_constant_delay(self, retry_policy_fixed, mock_llm_client, mock_response, fixed_config, mocker):
        """Every sleep call must use exactly delay_between_retries (no growth)."""
        from llm.exceptions import TransientNetworkError
        mock_sleep = mocker.patch("llm.retry_policy.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = [
            TransientNetworkError("fail 1"),
            TransientNetworkError("fail 2"),
            mock_response
        ]

        await retry_policy_fixed.query(MagicMock())

        for call in mock_sleep.call_args_list:
            assert call.args[0] == fixed_config.delay_between_retries

    @pytest.mark.asyncio
    async def test_sleep_not_called_after_final_attempt(self, retry_policy_fixed, mock_llm_client, fixed_config, mocker):
        """Sleep must be called max_retries-1 times — never after the last failed attempt."""
        from llm.exceptions import TransientNetworkError
        mock_sleep = mocker.patch("llm.retry_policy.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = TransientNetworkError("always fails")

        with pytest.raises(TransientNetworkError):
            await retry_policy_fixed.query(MagicMock())

        assert mock_sleep.call_count == fixed_config.max_retries - 1


# ---------------------------------------------------------------------------
# Tests: EXPONENTIAL backoff — delay doubles on each retry
# ---------------------------------------------------------------------------

class TestExponentialBackoff:

    @pytest.mark.asyncio
    async def test_delay_grows_between_retries(self, mock_llm_client, exponential_config, mocker):
        """Each successive sleep delay must be strictly greater than the previous one."""
        from llm.retry_policy import RetryPolicy
        from llm.exceptions import TransientNetworkError

        mock_sleep = mocker.patch("llm.retry_policy.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = TransientNetworkError("always fails")

        policy = RetryPolicy(llm_client=mock_llm_client, config=exponential_config)

        with pytest.raises(TransientNetworkError):
            await policy.query(MagicMock())

        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert len(delays) >= 2, "Need at least 2 retries to verify growth"
        for i in range(1, len(delays)):
            assert delays[i] > delays[i - 1], (
                f"Expected delay[{i}]={delays[i]} > delay[{i-1}]={delays[i-1]}"
            )

    @pytest.mark.asyncio
    async def test_first_delay_equals_base(self, mock_llm_client, exponential_config, mocker):
        """The first sleep delay must equal delay_between_retries (the base value)."""
        from llm.retry_policy import RetryPolicy
        from llm.exceptions import TransientNetworkError

        mock_sleep = mocker.patch("llm.retry_policy.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = [
            TransientNetworkError("fail"),
            MagicMock()  # success on second attempt
        ]

        policy = RetryPolicy(llm_client=mock_llm_client, config=exponential_config)
        await policy.query(MagicMock())

        first_delay = mock_sleep.call_args_list[0].args[0]
        assert first_delay == exponential_config.delay_between_retries


# ---------------------------------------------------------------------------
# Tests: JITTER backoff — delay is randomised but within valid bounds
# ---------------------------------------------------------------------------

class TestJitterBackoff:

    @pytest.mark.asyncio
    async def test_delay_is_non_negative(self, mock_llm_client, jitter_config, mocker):
        """Every jitter sleep delay must be >= 0."""
        from llm.retry_policy import RetryPolicy
        from llm.exceptions import TransientNetworkError

        mock_sleep = mocker.patch("llm.retry_policy.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = TransientNetworkError("always fails")

        policy = RetryPolicy(llm_client=mock_llm_client, config=jitter_config)

        with pytest.raises(TransientNetworkError):
            await policy.query(MagicMock())

        for call in mock_sleep.call_args_list:
            assert call.args[0] >= 0

    @pytest.mark.asyncio
    async def test_delay_does_not_exceed_upper_bound(self, mock_llm_client, jitter_config, mocker):
        """
        Every jitter delay must be <= delay_between_retries * 2 (a reasonable upper bound).
        Adjust the multiplier if your implementation uses a different cap.
        """
        from llm.retry_policy import RetryPolicy
        from llm.exceptions import TransientNetworkError

        mock_sleep = mocker.patch("llm.retry_policy.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = TransientNetworkError("always fails")

        policy = RetryPolicy(llm_client=mock_llm_client, config=jitter_config)
        upper_bound = jitter_config.delay_between_retries * 2

        with pytest.raises(TransientNetworkError):
            await policy.query(MagicMock())

        for call in mock_sleep.call_args_list:
            assert call.args[0] <= upper_bound


# ---------------------------------------------------------------------------
# Tests: retry count boundary
# ---------------------------------------------------------------------------

class TestRetryCountBoundary:

    @pytest.mark.asyncio
    async def test_max_retries_one_raises_immediately(self, mock_llm_client, mocker):
        """With max_retries=1, a single failure must raise with no sleep at all."""
        from llm.retry_policy import RetryPolicy
        from llm.config import RetryPolicyConfig, BackoffStrategy
        from llm.exceptions import TransientNetworkError

        config = RetryPolicyConfig(max_retries=1, delay_between_retries=1, backoff_strategy=BackoffStrategy.FIXED)
        mock_sleep = mocker.patch("llm.retry_policy.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = TransientNetworkError("fail")

        policy = RetryPolicy(llm_client=mock_llm_client, config=config)

        with pytest.raises(TransientNetworkError):
            await policy.query(MagicMock())

        assert mock_llm_client.query.call_count == 1
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_succeeds_on_last_possible_attempt(self, mock_llm_client, mock_response, mocker):
        """With max_retries=3, the third attempt succeeding must return the response."""
        from llm.retry_policy import RetryPolicy
        from llm.config import RetryPolicyConfig, BackoffStrategy
        from llm.exceptions import TransientNetworkError

        config = RetryPolicyConfig(max_retries=3, delay_between_retries=0, backoff_strategy=BackoffStrategy.FIXED)
        mocker.patch("llm.retry_policy.sleep", new=AsyncMock())
        mock_llm_client.query.side_effect = [
            TransientNetworkError("fail 1"),
            TransientNetworkError("fail 2"),
            mock_response
        ]

        policy = RetryPolicy(llm_client=mock_llm_client, config=config)
        result = await policy.query(MagicMock())

        assert result is mock_response
        assert mock_llm_client.query.call_count == 3


# ---------------------------------------------------------------------------
# Tests: payload forwarding
# ---------------------------------------------------------------------------

class TestPayloadForwarding:

    @pytest.mark.asyncio
    async def test_payload_passed_unchanged_to_client(self, retry_policy_fixed, mock_llm_client, mock_response):
        """The exact payload object must be forwarded to the underlying LLM client."""
        payload = MagicMock()
        mock_llm_client.query.return_value = mock_response

        await retry_policy_fixed.query(payload)

        mock_llm_client.query.assert_called_with(payload)
