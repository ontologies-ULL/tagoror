"""
Unit tests for GeminiClient (google-genai SDK + OpenTelemetry)
===============================================================
Covers:
  - genai.Client created with the correct api_key
  - OTel primitives created on __init__ (tracer, meter, 4 instruments, logger)
  - _build_generation_config: text/plain vs application/json
  - _extract_token_usage: metadata present vs None
  - _calculate_cost: formula, zero tokens, 1M tokens
  - _map_to_domain_response: all LLMResponse fields, span attributes, metrics
  - _handle_api_error: error re-raised, span ERROR, span.record_exception, log
  - _handle_network_failure: original exception re-raised, span ERROR, log
  - _query happy path: correct response, span name, counter incremented
  - _query API error: errors.APIError propagates unchanged
  - _query network failure: original exception propagates unchanged
  - _query span attributes set at the start (llm.model, llm.json_mode)

Testing strategy note:
  Because OTel primitives and genai.Client are created inside __init__ (not
  injected), we patch them before each instantiation via an autouse fixture.
  This keeps tests hermetic without requiring a real SDK pipeline.
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Shared patch fixture — applied to every test automatically
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_all(mocker):
    """
    Patches genai.Client, and all three OTel entry points so no real network
    call or SDK pipeline is needed. Returns a namespace for assertions.
    """
    # --- OTel span ---
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__  = MagicMock(return_value=False)

    # --- OTel tracer ---
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    # --- OTel meter + instruments ---
    mock_counter   = MagicMock()
    mock_histogram = MagicMock()
    mock_meter     = MagicMock()
    mock_meter.create_counter.return_value   = mock_counter
    mock_meter.create_histogram.return_value = mock_histogram

    # --- OTel logger ---
    mock_otel_logger = MagicMock()

    # --- google-genai Client ---
    mock_aio_models = MagicMock()
    mock_aio_models.generate_content = AsyncMock()
    mock_aio = MagicMock()
    mock_aio.models = mock_aio_models
    mock_genai_client = MagicMock()
    mock_genai_client.aio = mock_aio
    mock_client_cls = mocker.patch("google.genai.Client", return_value=mock_genai_client)

    mocker.patch("opentelemetry.trace.get_tracer",  return_value=mock_tracer)
    mocker.patch("opentelemetry.metrics.get_meter", return_value=mock_meter)
    mocker.patch("opentelemetry._logs.get_logger",  return_value=mock_otel_logger)

    ns = MagicMock()
    ns.span          = mock_span
    ns.tracer        = mock_tracer
    ns.meter         = mock_meter
    ns.counter       = mock_counter
    ns.histogram     = mock_histogram
    ns.otel_logger   = mock_otel_logger
    ns.client_cls    = mock_client_cls
    ns.genai_client  = mock_genai_client
    ns.aio_models    = mock_aio_models
    return ns


@pytest.fixture
def client(patch_all):
    """GeminiClient with all external dependencies patched."""
    from llm.clients.gemini import GeminiClient
    return GeminiClient(api_key="fake-key")


@pytest.fixture
def mock_payload():
    p = MagicMock()
    p.model_name  = "gemini-2.0-flash"
    p.temperature = 0.7
    p.json_mode   = False
    p.user_prompt = "Hello"
    return p


@pytest.fixture
def mock_payload_json(mock_payload):
    mock_payload.json_mode = True
    return mock_payload


@pytest.fixture
def mock_raw_response():
    r = MagicMock()
    r.text = '{"answer": "42"}'
    r.usage_metadata = MagicMock()
    r.usage_metadata.total_token_count = 200
    return r


# ---------------------------------------------------------------------------
# Tests: __init__
# ---------------------------------------------------------------------------

class TestInit:

    def test_genai_client_created_with_api_key(self, patch_all):
        """genai.Client must be instantiated with the provided api_key."""
        from llm.clients.gemini import GeminiClient
        GeminiClient(api_key="my-real-key")
        patch_all.client_cls.assert_called_once_with(api_key="my-real-key")

    def test_three_metric_counters_created(self, patch_all):
        """The meter must create exactly 3 counters (requests, tokens, cost)."""
        from llm.clients.gemini import GeminiClient
        GeminiClient(api_key="x")
        assert patch_all.meter.create_counter.call_count == 3

    def test_one_histogram_created(self, patch_all):
        """The meter must create exactly 1 histogram (duration)."""
        from llm.clients.gemini import GeminiClient
        GeminiClient(api_key="x")
        assert patch_all.meter.create_histogram.call_count == 1

    def test_tracer_created(self, patch_all):
        """get_tracer must be called during __init__."""
        from opentelemetry import trace
        from llm.clients.gemini import GeminiClient
        GeminiClient(api_key="x")
        assert trace.get_tracer.called


# ---------------------------------------------------------------------------
# Tests: _build_generation_config
# ---------------------------------------------------------------------------

class TestBuildGenerationConfig:

    def test_text_plain_when_json_mode_false(self, client, mock_payload, mocker):
        """MIME type must be text/plain when json_mode is False."""
        mock_cls = mocker.patch("google.genai.types.GenerateContentConfig")
        client._build_generation_config(mock_payload)
        mock_cls.assert_called_once_with(
            temperature=mock_payload.temperature,
            response_mime_type="text/plain",
        )

    def test_application_json_when_json_mode_true(self, client, mock_payload_json, mocker):
        """MIME type must be application/json when json_mode is True."""
        mock_cls = mocker.patch("google.genai.types.GenerateContentConfig")
        client._build_generation_config(mock_payload_json)
        mock_cls.assert_called_once_with(
            temperature=mock_payload_json.temperature,
            response_mime_type="application/json",
        )


# ---------------------------------------------------------------------------
# Tests: _extract_token_usage
# ---------------------------------------------------------------------------

class TestExtractTokenUsage:

    def test_returns_total_token_count_when_metadata_present(self, client, mock_raw_response):
        """Must return total_token_count when usage_metadata is present."""
        assert client._extract_token_usage(mock_raw_response) == 200

    def test_returns_zero_when_metadata_is_none(self, client):
        """Must return 0 when usage_metadata is None."""
        r = MagicMock()
        r.usage_metadata = None
        assert client._extract_token_usage(r) == 0


# ---------------------------------------------------------------------------
# Tests: _calculate_cost
# ---------------------------------------------------------------------------

class TestCalculateCost:

    def test_correct_formula(self, client):
        """Cost must equal tokens / 1_000_000 * _COST_PER_MILLION_TOKENS."""
        expected = (500_000 / 1_000_000) * client._COST_PER_MILLION_TOKENS
        assert client._calculate_cost(500_000) == pytest.approx(expected)

    def test_zero_tokens_returns_zero(self, client):
        assert client._calculate_cost(0) == 0.0

    def test_one_million_tokens_equals_full_rate(self, client):
        assert client._calculate_cost(1_000_000) == pytest.approx(client._COST_PER_MILLION_TOKENS)


# ---------------------------------------------------------------------------
# Tests: _map_to_domain_response
# ---------------------------------------------------------------------------

class TestMapToDomainResponse:

    def test_raw_content_mapped(self, client, mock_raw_response, patch_all):
        result = client._map_to_domain_response(mock_raw_response, time.time(), patch_all.span, "gemini-2.0-flash")
        assert result.raw_content == mock_raw_response.text

    def test_tokens_consumed_mapped(self, client, mock_raw_response, patch_all):
        result = client._map_to_domain_response(mock_raw_response, time.time(), patch_all.span, "gemini-2.0-flash")
        assert result.tokens_consumed == 200

    def test_duration_ms_is_non_negative_int(self, client, mock_raw_response, patch_all):
        result = client._map_to_domain_response(mock_raw_response, time.time() - 0.3, patch_all.span, "gemini-2.0-flash")
        assert isinstance(result.duration_ms, int)
        assert result.duration_ms >= 0

    def test_cost_is_non_negative(self, client, mock_raw_response, patch_all):
        result = client._map_to_domain_response(mock_raw_response, time.time(), patch_all.span, "gemini-2.0-flash")
        assert result.cost >= 0.0

    def test_span_status_set_to_ok(self, client, mock_raw_response, patch_all):
        """The span status must be OK after a successful mapping."""
        from opentelemetry.trace import StatusCode
        client._map_to_domain_response(mock_raw_response, time.time(), patch_all.span, "gemini-2.0-flash")
        status_arg = patch_all.span.set_status.call_args[0][0]
        assert status_arg.status_code == StatusCode.OK

    def test_span_attributes_tokens_duration_cost(self, client, mock_raw_response, patch_all):
        """span.set_attribute must be called for tokens, duration and cost."""
        client._map_to_domain_response(mock_raw_response, time.time(), patch_all.span, "gemini-2.0-flash")
        keys = [c.args[0] for c in patch_all.span.set_attribute.call_args_list]
        assert "llm.tokens_consumed" in keys
        assert "llm.duration_ms"     in keys
        assert "llm.cost_usd"        in keys

    def test_histogram_recorded_once(self, client, mock_raw_response, patch_all):
        """Duration histogram must be recorded exactly once."""
        client._map_to_domain_response(mock_raw_response, time.time(), patch_all.span, "gemini-2.0-flash")
        patch_all.histogram.record.assert_called_once()

    def test_info_log_emitted(self, client, mock_raw_response, patch_all):
        """An INFO log must be emitted on successful mapping."""
        client._map_to_domain_response(mock_raw_response, time.time(), patch_all.span, "gemini-2.0-flash")
        patch_all.otel_logger.emit.assert_called()


# ---------------------------------------------------------------------------
# Tests: _handle_api_error
# ---------------------------------------------------------------------------

class TestHandleApiError:

    def test_reraises_api_error(self, client, patch_all):
        """errors.APIError must be re-raised as-is."""
        from google.genai import errors
        api_err = MagicMock(spec=errors.APIError)
        api_err.code    = 429
        api_err.message = "quota exceeded"
        with pytest.raises(type(api_err)):
            client._handle_api_error(api_err, patch_all.span)

    def test_span_status_set_to_error(self, client, patch_all):
        """The span status must be ERROR after an API error."""
        from opentelemetry.trace import StatusCode
        from google.genai import errors
        api_err = MagicMock(spec=errors.APIError)
        api_err.code    = 500
        api_err.message = "internal error"
        with pytest.raises(Exception):
            client._handle_api_error(api_err, patch_all.span)
        status_arg = patch_all.span.set_status.call_args[0][0]
        assert status_arg.status_code == StatusCode.ERROR

    def test_span_records_exception(self, client, patch_all):
        """span.record_exception must be called with the original error."""
        from google.genai import errors
        api_err = MagicMock(spec=errors.APIError)
        api_err.code    = 403
        api_err.message = "forbidden"
        with pytest.raises(Exception):
            client._handle_api_error(api_err, patch_all.span)
        patch_all.span.record_exception.assert_called_once_with(api_err)

    def test_error_log_emitted(self, client, patch_all):
        """An ERROR log must be emitted before re-raising."""
        from google.genai import errors
        api_err = MagicMock(spec=errors.APIError)
        api_err.code    = 400
        api_err.message = "bad request"
        with pytest.raises(Exception):
            client._handle_api_error(api_err, patch_all.span)
        patch_all.otel_logger.emit.assert_called()


# ---------------------------------------------------------------------------
# Tests: _handle_network_failure
# ---------------------------------------------------------------------------

class TestHandleNetworkFailure:

    def test_reraises_original_exception_type(self, client, patch_all):
        """The original exception type must be preserved when re-raising."""
        with pytest.raises(ConnectionError, match="DNS failure"):
            client._handle_network_failure(ConnectionError("DNS failure"), patch_all.span)

    def test_span_status_set_to_error(self, client, patch_all):
        """The span status must be ERROR after a network failure."""
        from opentelemetry.trace import StatusCode
        with pytest.raises(Exception):
            client._handle_network_failure(Exception("boom"), patch_all.span)
        status_arg = patch_all.span.set_status.call_args[0][0]
        assert status_arg.status_code == StatusCode.ERROR

    def test_span_records_exception(self, client, patch_all):
        """span.record_exception must be called with the original error."""
        original = RuntimeError("timeout")
        with pytest.raises(RuntimeError):
            client._handle_network_failure(original, patch_all.span)
        patch_all.span.record_exception.assert_called_once_with(original)

    def test_error_log_emitted(self, client, patch_all):
        """An ERROR log must be emitted before re-raising."""
        with pytest.raises(Exception):
            client._handle_network_failure(Exception("boom"), patch_all.span)
        patch_all.otel_logger.emit.assert_called()


# ---------------------------------------------------------------------------
# Tests: _query (full flow)
# ---------------------------------------------------------------------------

class TestQuery:

    @pytest.mark.asyncio
    async def test_happy_path_returns_llm_response(self, client, mock_payload, mock_raw_response, mocker):
        """Full flow must return a valid LLMResponse with the correct content."""
        mocker.patch.object(client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(client, "_execute_network_call", new=AsyncMock(return_value=mock_raw_response))

        result = await client._query(mock_payload)

        assert result is not None
        assert result.raw_content == mock_raw_response.text

    @pytest.mark.asyncio
    async def test_span_created_with_correct_name(self, client, mock_payload, mock_raw_response, patch_all, mocker):
        """A span named 'gemini.query' must be started for every request."""
        mocker.patch.object(client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(client, "_execute_network_call", new=AsyncMock(return_value=mock_raw_response))

        await client._query(mock_payload)

        patch_all.tracer.start_as_current_span.assert_called_once_with("gemini.query")

    @pytest.mark.asyncio
    async def test_model_and_json_mode_set_as_span_attributes(self, client, mock_payload, mock_raw_response, patch_all, mocker):
        """llm.model and llm.json_mode must be set as span attributes at the start."""
        mocker.patch.object(client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(client, "_execute_network_call", new=AsyncMock(return_value=mock_raw_response))

        await client._query(mock_payload)

        keys = [c.args[0] for c in patch_all.span.set_attribute.call_args_list]
        assert "llm.model"     in keys
        assert "llm.json_mode" in keys

    @pytest.mark.asyncio
    async def test_request_counter_incremented(self, client, mock_payload, mock_raw_response, patch_all, mocker):
        """The request counter must be incremented once per call."""
        mocker.patch.object(client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(client, "_execute_network_call", new=AsyncMock(return_value=mock_raw_response))

        await client._query(mock_payload)

        patch_all.counter.add.assert_called()

    @pytest.mark.asyncio
    async def test_api_error_propagates(self, client, mock_payload, mocker):
        """errors.APIError raised during the call must propagate unchanged."""
        from google.genai import errors
        api_err = MagicMock(spec=errors.APIError)
        api_err.code    = 429
        api_err.message = "rate limit"
        mocker.patch.object(client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(client, "_execute_network_call", new=AsyncMock(side_effect=api_err))

        with pytest.raises(type(api_err)):
            await client._query(mock_payload)

    @pytest.mark.asyncio
    async def test_network_failure_propagates_with_original_type(self, client, mock_payload, mocker):
        """A generic network error must propagate with its original type and message."""
        mocker.patch.object(client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(client, "_execute_network_call",
            new=AsyncMock(side_effect=ConnectionError("DNS failure")))

        with pytest.raises(ConnectionError, match="DNS failure"):
            await client._query(mock_payload)
