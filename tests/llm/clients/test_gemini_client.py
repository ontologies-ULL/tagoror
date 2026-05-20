"""
Unit tests for GeminiClient
============================
Covers:
  - Happy path (full _query flow)
  - Domain response mapping (_map_to_domain_response)
  - Token extraction with and without usage_metadata
  - Cost calculation
  - Model and config construction (json_mode ON/OFF)
  - Gemini safety block (StopCandidateException)
  - Generic network failure
  - Logger calls on every branch
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from google.genai import types

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_logger(mocker):
    """ITelemetryLogger mocked with pytest-mock."""
    logger = mocker.MagicMock()
    logger.log_event = mocker.MagicMock()
    return logger


@pytest.fixture
def mock_payload():
    """Minimal valid LLMPayload."""
    payload = MagicMock()
    payload.model_name = "gemini-1.5-flash"
    payload.temperature = 0.7
    payload.json_mode = False
    payload.user_prompt = "Hello, how are you?"
    return payload


@pytest.fixture
def mock_payload_json(mock_payload):
    """LLMPayload with json_mode=True."""
    mock_payload.json_mode = True
    return mock_payload


@pytest.fixture
def mock_raw_response():
    """Simulated response from the Google SDK."""
    response = MagicMock()
    response.text = '{"answer": "42"}'
    response.usage_metadata = MagicMock()
    response.usage_metadata.total_token_count = 150
    return response


@pytest.fixture
def gemini_client(mock_logger, mocker):
    """
    GeminiClient instance with genai.Client patched
    so no real API key is required in any test.
    """
    mocker.patch("google.genai.Client")
    from llm.clients.gemini import GeminiClient
    return GeminiClient(api_key="fake-api-key", logger=mock_logger)


# ---------------------------------------------------------------------------
# Tests: __init__
# ---------------------------------------------------------------------------

class TestGeminiClientInit:

    def test_configure_is_called_with_api_key(self, mock_logger, mocker):
        """genai.Client must receive exactly the provided api_key."""
        mock_client = mocker.patch("google.genai.Client")
        from llm.clients.gemini import GeminiClient

        GeminiClient(api_key="my-secret-key", logger=mock_logger)

        mock_client.assert_called_once_with(api_key="my-secret-key")

    def test_logger_is_stored(self, gemini_client, mock_logger):
        """The injected logger must be accessible internally."""
        assert gemini_client._logger is mock_logger


# ---------------------------------------------------------------------------
# Tests: _build_model
# ---------------------------------------------------------------------------

class TestBuildModel:

    def test_returns_generative_model_with_correct_model_name(self, gemini_client, mock_payload, mocker):
        """_build_model must return the payload model name."""
        result = gemini_client._build_model(mock_payload)
        assert result == mock_payload.model_name


# ---------------------------------------------------------------------------
# Tests: _build_generation_config
# ---------------------------------------------------------------------------

class TestBuildGenerationConfig:

    def test_plain_text_mode(self, gemini_client, mock_payload, mocker):
        """With json_mode=False, the MIME type must be text/plain."""
        mock_config_cls = mocker.patch("google.genai.types.GenerateContentConfig")

        gemini_client._build_generation_config(mock_payload)

        mock_config_cls.assert_called_once_with(
            temperature=mock_payload.temperature,
            response_mime_type="text/plain"
        )

    def test_json_mode(self, gemini_client, mock_payload_json, mocker):
        """With json_mode=True, the MIME type must be application/json."""
        mock_config_cls = mocker.patch("google.genai.types.GenerateContentConfig")

        gemini_client._build_generation_config(mock_payload_json)

        mock_config_cls.assert_called_once_with(
            temperature=mock_payload_json.temperature,
            response_mime_type="application/json"
        )


# ---------------------------------------------------------------------------
# Tests: _extract_token_usage
# ---------------------------------------------------------------------------

class TestExtractTokenUsage:

    def test_returns_token_count_when_metadata_present(self, gemini_client, mock_raw_response):
        """Must return total_token_count when usage_metadata is present."""
        result = gemini_client._extract_token_usage(mock_raw_response)
        assert result == 150

    def test_returns_zero_when_no_metadata(self, gemini_client):
        """Must return 0 if usage_metadata is None."""
        response = MagicMock()
        response.usage_metadata = None
        result = gemini_client._extract_token_usage(response)
        assert result == 0


# ---------------------------------------------------------------------------
# Tests: _calculate_cost
# ---------------------------------------------------------------------------

class TestCalculateCost:

    def test_cost_calculation_is_correct(self, gemini_client):
        """Cost must equal tokens / 1_000_000 * _COST_PER_MILLION_TOKENS."""
        tokens = 500_000
        expected = (500_000 / 1_000_000) * gemini_client._COST_PER_MILLION_TOKENS
        assert gemini_client._calculate_cost(tokens) == pytest.approx(expected)

    def test_zero_tokens_gives_zero_cost(self, gemini_client):
        """With 0 tokens the cost must be exactly 0.0."""
        assert gemini_client._calculate_cost(0) == 0.0

    def test_one_million_tokens_equals_rate(self, gemini_client):
        """1M tokens must cost exactly _COST_PER_MILLION_TOKENS."""
        assert gemini_client._calculate_cost(1_000_000) == pytest.approx(
            gemini_client._COST_PER_MILLION_TOKENS
        )


# ---------------------------------------------------------------------------
# Tests: _map_to_domain_response
# ---------------------------------------------------------------------------

class TestMapToDomainResponse:

    def test_raw_content_is_mapped(self, gemini_client, mock_raw_response):
        """LLMResponse.raw_content must match response.text."""
        result = gemini_client._map_to_domain_response(mock_raw_response, time.time())
        assert result.raw_content == mock_raw_response.text

    def test_tokens_consumed_is_mapped(self, gemini_client, mock_raw_response):
        """tokens_consumed must match the usage_metadata value."""
        result = gemini_client._map_to_domain_response(mock_raw_response, time.time())
        assert result.tokens_consumed == 150

    def test_duration_ms_is_positive(self, gemini_client, mock_raw_response):
        """duration_ms must be a non-negative integer."""
        start = time.time() - 0.5  # Simulate 500ms elapsed
        result = gemini_client._map_to_domain_response(mock_raw_response, start)
        assert result.duration_ms >= 0
        assert isinstance(result.duration_ms, int)

    def test_cost_is_calculated(self, gemini_client, mock_raw_response):
        """cost must not be None or negative."""
        result = gemini_client._map_to_domain_response(mock_raw_response, time.time())
        assert result.cost >= 0.0

    def test_success_log_is_called(self, gemini_client, mock_raw_response, mock_logger):
        """An INFO event must be logged on successful mapping."""
        gemini_client._map_to_domain_response(mock_raw_response, time.time())
        # Verify at least one call was at INFO level
        calls = [c.args[0] for c in mock_logger.log_event.call_args_list]
        assert "INFO" in calls


# ---------------------------------------------------------------------------
# Tests: _handle_security_block
# ---------------------------------------------------------------------------

class TestHandleSecurityBlock:

    def test_raises_exception_with_message(self, gemini_client):
        """Must raise an Exception containing 'safety policies'."""
        original_error = Exception("safety filter triggered")
        with pytest.raises(Exception, match="safety policies"):
            gemini_client._handle_security_block(original_error)

    def test_logs_error_event(self, gemini_client, mock_logger):
        """Must log an ERROR event before raising."""
        with pytest.raises(Exception):
            gemini_client._handle_security_block(Exception("blocked"))
        mock_logger.log_event.assert_called_once()
        assert mock_logger.log_event.call_args[0][0] == "ERROR"


# ---------------------------------------------------------------------------
# Tests: _handle_network_failure
# ---------------------------------------------------------------------------

class TestHandleNetworkFailure:

    def test_reraises_original_exception(self, gemini_client):
        """Must re-raise the original exception without wrapping it."""
        original = ConnectionError("timeout")
        with pytest.raises(ConnectionError):
            gemini_client._handle_network_failure(original)

    def test_logs_error_event(self, gemini_client, mock_logger):
        """Must log an ERROR event before re-raising."""
        with pytest.raises(Exception):
            gemini_client._handle_network_failure(Exception("boom"))
        mock_logger.log_event.assert_called_once()
        assert mock_logger.log_event.call_args[0][0] == "ERROR"


# ---------------------------------------------------------------------------
# Tests: _query (full flow — internal method integration)
# ---------------------------------------------------------------------------

class TestQuery:

    @pytest.mark.asyncio
    async def test_happy_path_returns_llm_response(self, gemini_client, mock_payload, mock_raw_response, mocker):
        """Full flow without errors: must return a valid LLMResponse."""
        mocker.patch.object(gemini_client, "_build_model", return_value=MagicMock())
        mocker.patch.object(gemini_client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(gemini_client, "_execute_network_call", new=AsyncMock(return_value=mock_raw_response))

        result = await gemini_client._query(mock_payload)

        assert result is not None
        assert result.raw_content == mock_raw_response.text

    @pytest.mark.asyncio
    async def test_happy_path_logs_start_event(self, gemini_client, mock_payload, mock_raw_response, mock_logger, mocker):
        """Must log an INFO event at the start of the request."""
        mocker.patch.object(gemini_client, "_build_model", return_value=MagicMock())
        mocker.patch.object(gemini_client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(gemini_client, "_execute_network_call", new=AsyncMock(return_value=mock_raw_response))

        await gemini_client._query(mock_payload)

        first_call = mock_logger.log_event.call_args_list[0]
        assert first_call.args[0] == "INFO"
        assert mock_payload.model_name in first_call.args[1]

    @pytest.mark.asyncio
    async def test_security_block_raises_exception(self, gemini_client, mock_payload, mocker):
        """StopCandidateException must be converted to an Exception containing 'safety policies'."""
        mocker.patch.object(gemini_client, "_build_model", return_value=MagicMock())
        mocker.patch.object(gemini_client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(
            gemini_client,
            "_execute_network_call",
            new=AsyncMock(side_effect=types.StopCandidateException("blocked"))
        )

        with pytest.raises(Exception, match="safety policies"):
            await gemini_client._query(mock_payload)

    @pytest.mark.asyncio
    async def test_network_failure_reraises_original(self, gemini_client, mock_payload, mocker):
        """A generic network error must propagate unchanged."""
        original_error = ConnectionError("DNS failure")
        mocker.patch.object(gemini_client, "_build_model", return_value=MagicMock())
        mocker.patch.object(gemini_client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(
            gemini_client,
            "_execute_network_call",
            new=AsyncMock(side_effect=original_error)
        )

        with pytest.raises(ConnectionError, match="DNS failure"):
            await gemini_client._query(mock_payload)

    @pytest.mark.asyncio
    async def test_network_failure_logs_error(self, gemini_client, mock_payload, mock_logger, mocker):
        """A network failure must log an ERROR event."""
        mocker.patch.object(gemini_client, "_build_model", return_value=MagicMock())
        mocker.patch.object(gemini_client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(
            gemini_client,
            "_execute_network_call",
            new=AsyncMock(side_effect=Exception("boom"))
        )

        with pytest.raises(Exception):
            await gemini_client._query(mock_payload)

        error_calls = [c for c in mock_logger.log_event.call_args_list if c.args[0] == "ERROR"]
        assert len(error_calls) >= 1
