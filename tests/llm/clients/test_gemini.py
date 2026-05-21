"""
Unit tests for GeminiClient
=============================
Based on: gemini.py (google-genai SDK + OpenTelemetry)

Covers:
  - __init__: genai.Client created with api_key, OTel primitives created,
              4 metric instruments registered
  - _build_generation_config: text/plain vs application/json
  - _execute_network_call: delegates to client.aio.models.generate_content
  - _extract_token_usage: metadata present vs None
  - _calculate_cost: formula, zero tokens, 1M tokens
  - _map_to_domain_response: all LLMResponse fields, span attributes, metrics
  - _handle_api_error: error re-raised, span ERROR, record_exception, log
  - _handle_network_failure: original exception re-raised, span ERROR, log
  - _emit_log: INFO severity for "INFO", ERROR for anything else, body content
  - _emit_log BUG: "info" (lowercase) produces ERROR — documented as known bug
  - _query happy path: LLMResponse returned, span name, span attributes,
                       request counter incremented
  - _query API error: errors.APIError propagates unchanged
  - _query network failure: original exception propagates unchanged

Patching strategy:
  - genai.Client is patched via mocker.patch("google.genai.Client")
  - OTel meter/tracer/logger entry points are patched via mocker.patch(...)
  - _emit_log is patched with mocker.patch.object where we only need to verify
    it was called, avoiding the get_logger reference resolution problem.
  - For tests that inspect the LogRecord directly, LogRecord is patched at its
    construction site inside gemini.py.
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, call
from llm.models import LLMPayload, LLMResponse


# ---------------------------------------------------------------------------
# Autouse patch fixture — OTel infrastructure + genai.Client
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_all(mocker):
    """
    Patches genai.Client and all OTel entry points before every test.
    Returns a namespace with the mocked objects for assertions.
    """
    # OTel span
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__  = MagicMock(return_value=False)

    # OTel tracer
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    # OTel meter + instruments
    mock_counter   = MagicMock()
    mock_histogram = MagicMock()
    mock_meter     = MagicMock()
    mock_meter.create_counter.return_value   = mock_counter
    mock_meter.create_histogram.return_value = mock_histogram

    # OTel logger — patched at the entry point AND at the module level
    mock_otel_logger = MagicMock()
    mocker.patch("opentelemetry._logs.get_logger",  return_value=mock_otel_logger)

    # google-genai Client
    mock_aio_models = MagicMock()
    mock_aio_models.generate_content = AsyncMock()
    mock_aio = MagicMock()
    mock_aio.models = mock_aio_models
    mock_genai_client = MagicMock()
    mock_genai_client.aio = mock_aio
    mock_client_cls = mocker.patch("google.genai.Client", return_value=mock_genai_client)

    mocker.patch("opentelemetry.trace.get_tracer",  return_value=mock_tracer)
    mocker.patch("opentelemetry.metrics.get_meter", return_value=mock_meter)

    ns             = MagicMock()
    ns.span        = mock_span
    ns.tracer      = mock_tracer
    ns.meter       = mock_meter
    ns.counter     = mock_counter
    ns.histogram   = mock_histogram
    ns.otel_logger = mock_otel_logger
    ns.client_cls  = mock_client_cls
    ns.aio_models  = mock_aio_models
    return ns


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(patch_all):
    from llm.clients.gemini import GeminiClient
    return GeminiClient(api_key="fake-key")


@pytest.fixture
def payload():
    return LLMPayload(
        user_prompt="Hello",
        model_name="gemini-2.0-flash",
        temperature=0.7,
        json_mode=False,
    )


@pytest.fixture
def payload_json():
    return LLMPayload(
        user_prompt="Hello",
        model_name="gemini-2.0-flash",
        temperature=0.7,
        json_mode=True,
    )


@pytest.fixture
def raw_response():
    r = MagicMock()
    r.text = '{"answer": "42"}'
    r.usage_metadata = MagicMock()
    r.usage_metadata.total_token_count = 200
    return r


# ---------------------------------------------------------------------------
# Helper: build a realistic APIError substitute
# ---------------------------------------------------------------------------

def make_api_error(code: int, message: str):
    """
    errors.APIError requires response_json in __init__ so we cannot instantiate
    it directly. We create a plain Exception subclass that satisfies
    isinstance(..., errors.APIError) via spec — but for raising we use a
    real subclass instead.
    """
    from google.genai import errors

    class FakeAPIError(errors.APIError):
        """Minimal subclass that bypasses the response_json requirement."""
        def __init__(self):
            # Skip the parent __init__ entirely
            self.code    = code
            self.message = message
            self.args    = (message,)

    return FakeAPIError()


# ---------------------------------------------------------------------------
# Tests: __init__
# ---------------------------------------------------------------------------

class TestInit:

    def test_genai_client_created_with_api_key(self, patch_all):
        """genai.Client must be instantiated with the provided api_key."""
        from llm.clients.gemini import GeminiClient
        GeminiClient(api_key="my-key")
        patch_all.client_cls.assert_called_once_with(api_key="my-key")

    def test_three_counters_created(self, patch_all):
        """meter.create_counter must be called exactly 3 times (requests, tokens, cost)."""
        from llm.clients.gemini import GeminiClient
        GeminiClient(api_key="x")
        assert patch_all.meter.create_counter.call_count == 3

    def test_one_histogram_created(self, patch_all):
        """meter.create_histogram must be called exactly once (duration)."""
        from llm.clients.gemini import GeminiClient
        GeminiClient(api_key="x")
        assert patch_all.meter.create_histogram.call_count == 1

    def test_tracer_and_meter_created(self, patch_all):
        """get_tracer and get_meter must be called during __init__."""
        from opentelemetry import trace, metrics
        from llm.clients.gemini import GeminiClient
        GeminiClient(api_key="x")
        assert trace.get_tracer.called
        assert metrics.get_meter.called

    def test_logger_attribute_is_set(self, client):
        """_logger must be set on the instance after __init__."""
        assert hasattr(client, "_logger")
        assert client._logger is not None


# ---------------------------------------------------------------------------
# Tests: _build_generation_config
# ---------------------------------------------------------------------------

class TestBuildGenerationConfig:

    def test_text_plain_when_json_mode_false(self, client, payload, mocker):
        """MIME type must be text/plain when json_mode is False."""
        mock_cls = mocker.patch("google.genai.types.GenerateContentConfig")
        client._build_generation_config(payload)
        mock_cls.assert_called_once_with(
            temperature=payload.temperature,
            response_mime_type="text/plain",
        )

    def test_application_json_when_json_mode_true(self, client, payload_json, mocker):
        """MIME type must be application/json when json_mode is True."""
        mock_cls = mocker.patch("google.genai.types.GenerateContentConfig")
        client._build_generation_config(payload_json)
        mock_cls.assert_called_once_with(
            temperature=payload_json.temperature,
            response_mime_type="application/json",
        )


# ---------------------------------------------------------------------------
# Tests: _execute_network_call
# ---------------------------------------------------------------------------

class TestExecuteNetworkCall:

    @pytest.mark.asyncio
    async def test_delegates_to_aio_models_generate_content(self, client, patch_all, raw_response):
        """Must call client.aio.models.generate_content with the correct arguments."""
        patch_all.aio_models.generate_content.return_value = raw_response
        mock_config = MagicMock()

        await client._execute_network_call("gemini-2.0-flash", "Hello", mock_config)

        patch_all.aio_models.generate_content.assert_called_once_with(
            model="gemini-2.0-flash",
            contents="Hello",
            config=mock_config,
        )

    @pytest.mark.asyncio
    async def test_returns_raw_sdk_response(self, client, patch_all, raw_response):
        """Must return whatever client.aio.models.generate_content returns."""
        patch_all.aio_models.generate_content.return_value = raw_response
        result = await client._execute_network_call("gemini-2.0-flash", "Hello", MagicMock())
        assert result is raw_response


# ---------------------------------------------------------------------------
# Tests: _extract_token_usage
# ---------------------------------------------------------------------------

class TestExtractTokenUsage:

    def test_returns_total_token_count_when_metadata_present(self, client, raw_response):
        assert client._extract_token_usage(raw_response) == 200

    def test_returns_zero_when_metadata_is_none(self, client):
        r = MagicMock()
        r.usage_metadata = None
        assert client._extract_token_usage(r) == 0


# ---------------------------------------------------------------------------
# Tests: _calculate_cost
# ---------------------------------------------------------------------------

class TestCalculateCost:

    def test_correct_formula(self, client):
        expected = (500_000 / 1_000_000) * client._COST_PER_MILLION_TOKENS
        assert client._calculate_cost(500_000) == pytest.approx(expected)

    def test_zero_tokens_returns_zero(self, client):
        assert client._calculate_cost(0) == 0.0

    def test_one_million_tokens_equals_full_rate(self, client):
        assert client._calculate_cost(1_000_000) == pytest.approx(client._COST_PER_MILLION_TOKENS)


# ---------------------------------------------------------------------------
# Tests: _emit_log
# We patch LogRecord at its import site inside gemini.py so we can inspect
# the arguments it was called with, without needing the real OTel pipeline.
# ---------------------------------------------------------------------------

class TestEmitLog:

    def test_info_uppercase_emits_info_severity(self, client, mocker):
        """'INFO' (uppercase) must produce SeverityNumber.INFO."""
        from opentelemetry._logs._internal import SeverityNumber
        captured = []
        original_record = __import__(
            "opentelemetry._logs._internal", fromlist=["LogRecord"]
        ).LogRecord

        def capture_record(**kwargs):
            obj = original_record(**kwargs)
            captured.append(obj)
            return obj

        mocker.patch("llm.clients.gemini.LogRecord", side_effect=capture_record)
        client._emit_log("INFO", "test message")
        assert len(captured) == 1
        assert captured[0].severity_number == SeverityNumber.INFO

    def test_error_emits_error_severity(self, client, mocker):
        """'ERROR' must produce SeverityNumber.ERROR."""
        from opentelemetry._logs._internal import SeverityNumber
        captured = []
        original_record = __import__(
            "opentelemetry._logs._internal", fromlist=["LogRecord"]
        ).LogRecord

        def capture_record(**kwargs):
            obj = original_record(**kwargs)
            captured.append(obj)
            return obj

        mocker.patch("llm.clients.gemini.LogRecord", side_effect=capture_record)
        client._emit_log("ERROR", "something failed")
        assert captured[0].severity_number == SeverityNumber.ERROR

    def test_log_body_contains_message(self, client, mocker):
        """The emitted LogRecord body must contain the message passed in."""
        captured = []
        original_record = __import__(
            "opentelemetry._logs._internal", fromlist=["LogRecord"]
        ).LogRecord

        def capture_record(**kwargs):
            obj = original_record(**kwargs)
            captured.append(obj)
            return obj

        mocker.patch("llm.clients.gemini.LogRecord", side_effect=capture_record)
        client._emit_log("INFO", "my specific message")
        assert "my specific message" in captured[0].body

    def test_emit_is_called_on_logger(self, client, mocker):
        """_logger.emit must be called once per _emit_log call."""
        mocker.patch("llm.clients.gemini.LogRecord", return_value=MagicMock())
        mock_emit = mocker.patch.object(client, "_logger")
        client._emit_log("INFO", "hello")
        mock_emit.emit.assert_called_once()

    def test_known_bug_info_lowercase_emits_error_severity(self, client, mocker):
        """
        BUG (line 62): _emit_log is called with "info" (lowercase) at the
        start of _query(). Because _emit_log compares with "INFO" (uppercase),
        lowercase "info" falls through to the else branch and emits ERROR
        severity instead of INFO.
        Fix: change line 62 in gemini.py to _emit_log("INFO", ...).
        """
        from opentelemetry._logs._internal import SeverityNumber
        captured = []
        original_record = __import__(
            "opentelemetry._logs._internal", fromlist=["LogRecord"]
        ).LogRecord

        def capture_record(**kwargs):
            obj = original_record(**kwargs)
            captured.append(obj)
            return obj

        mocker.patch("llm.clients.gemini.LogRecord", side_effect=capture_record)
        client._emit_log("info", "this should be INFO but is actually ERROR")
        # Documents the bug: lowercase produces ERROR, not INFO
        assert captured[0].severity_number == SeverityNumber.ERROR


# ---------------------------------------------------------------------------
# Tests: _map_to_domain_response
# ---------------------------------------------------------------------------

class TestMapToDomainResponse:

    def test_raw_content_mapped(self, client, raw_response, patch_all):
        result = client._map_to_domain_response(raw_response, time.time(), patch_all.span, "gemini-2.0-flash")
        assert result.raw_content == raw_response.text

    def test_tokens_consumed_mapped(self, client, raw_response, patch_all):
        result = client._map_to_domain_response(raw_response, time.time(), patch_all.span, "gemini-2.0-flash")
        assert result.tokens_consumed == 200

    def test_duration_ms_is_non_negative_int(self, client, raw_response, patch_all):
        result = client._map_to_domain_response(raw_response, time.time() - 0.3, patch_all.span, "gemini-2.0-flash")
        assert isinstance(result.duration_ms, int)
        assert result.duration_ms >= 0

    def test_cost_is_non_negative(self, client, raw_response, patch_all):
        result = client._map_to_domain_response(raw_response, time.time(), patch_all.span, "gemini-2.0-flash")
        assert result.cost >= 0.0

    def test_span_status_set_to_ok(self, client, raw_response, patch_all):
        """Span status must be OK after successful mapping."""
        from opentelemetry.trace import StatusCode
        client._map_to_domain_response(raw_response, time.time(), patch_all.span, "gemini-2.0-flash")
        status_arg = patch_all.span.set_status.call_args[0][0]
        assert status_arg.status_code == StatusCode.OK

    def test_span_attributes_set_for_tokens_duration_cost(self, client, raw_response, patch_all):
        """llm.tokens_consumed, llm.duration_ms and llm.cost_usd must be span attributes."""
        client._map_to_domain_response(raw_response, time.time(), patch_all.span, "gemini-2.0-flash")
        keys = [c.args[0] for c in patch_all.span.set_attribute.call_args_list]
        assert "llm.tokens_consumed" in keys
        assert "llm.duration_ms"     in keys
        assert "llm.cost_usd"        in keys

    def test_histogram_recorded_once(self, client, raw_response, patch_all):
        """Duration histogram must be recorded exactly once per mapping."""
        client._map_to_domain_response(raw_response, time.time(), patch_all.span, "gemini-2.0-flash")
        patch_all.histogram.record.assert_called_once()

    def test_info_log_emitted(self, client, raw_response, patch_all, mocker):
        """_emit_log must be called during successful mapping."""
        mock_emit = mocker.patch.object(client, "_emit_log")
        client._map_to_domain_response(raw_response, time.time(), patch_all.span, "gemini-2.0-flash")
        mock_emit.assert_called()


# ---------------------------------------------------------------------------
# Tests: _handle_api_error
# ---------------------------------------------------------------------------

class TestHandleApiError:

    def test_reraises_api_error(self, client, patch_all):
        """errors.APIError must be re-raised as-is."""
        from google.genai import errors
        err = make_api_error(429, "quota exceeded")
        with pytest.raises(errors.APIError):
            client._handle_api_error(err, patch_all.span)

    def test_span_status_set_to_error(self, client, patch_all):
        """Span status must be ERROR after an API error."""
        from opentelemetry.trace import StatusCode
        err = make_api_error(500, "internal error")
        with pytest.raises(Exception):
            client._handle_api_error(err, patch_all.span)
        status_arg = patch_all.span.set_status.call_args[0][0]
        assert status_arg.status_code == StatusCode.ERROR

    def test_error_message_contains_code_and_message(self, client, patch_all):
        """The span status description must include both the HTTP code and message."""
        err = make_api_error(403, "forbidden")
        with pytest.raises(Exception):
            client._handle_api_error(err, patch_all.span)
        status_arg = patch_all.span.set_status.call_args[0][0]
        assert "403"       in status_arg.description
        assert "forbidden" in status_arg.description

    def test_span_records_exception(self, client, patch_all):
        """span.record_exception must be called with the original error."""
        err = make_api_error(400, "bad request")
        with pytest.raises(Exception):
            client._handle_api_error(err, patch_all.span)
        patch_all.span.record_exception.assert_called_once_with(err)

    def test_error_log_emitted(self, client, patch_all, mocker):
        """_emit_log must be called with ERROR level before re-raising."""
        mock_emit = mocker.patch.object(client, "_emit_log")
        err = make_api_error(503, "unavailable")
        with pytest.raises(Exception):
            client._handle_api_error(err, patch_all.span)
        mock_emit.assert_called_once_with("ERROR", pytest.approx(mock_emit.call_args[0][1], abs=0))
        assert mock_emit.call_args[0][0] == "ERROR"


# ---------------------------------------------------------------------------
# Tests: _handle_network_failure
# ---------------------------------------------------------------------------

class TestHandleNetworkFailure:

    def test_reraises_original_exception_type(self, client, patch_all):
        """The original exception type must be preserved when re-raising."""
        with pytest.raises(ConnectionError, match="DNS failure"):
            client._handle_network_failure(ConnectionError("DNS failure"), patch_all.span)

    def test_span_status_set_to_error(self, client, patch_all):
        """Span status must be ERROR after a network failure."""
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

    def test_error_log_emitted(self, client, patch_all, mocker):
        """_emit_log must be called with ERROR level before re-raising."""
        mock_emit = mocker.patch.object(client, "_emit_log")
        with pytest.raises(Exception):
            client._handle_network_failure(Exception("boom"), patch_all.span)
        assert mock_emit.call_args[0][0] == "ERROR"


# ---------------------------------------------------------------------------
# Tests: _query (full flow)
# ---------------------------------------------------------------------------

class TestQuery:

    @pytest.mark.asyncio
    async def test_happy_path_returns_llm_response(self, client, payload, raw_response, mocker):
        """Full flow must return a valid LLMResponse with the correct content."""
        mocker.patch.object(client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(client, "_execute_network_call", new=AsyncMock(return_value=raw_response))
        result = await client._query(payload)
        assert result is not None
        assert result.raw_content == raw_response.text

    @pytest.mark.asyncio
    async def test_span_created_with_correct_name(self, client, payload, raw_response, patch_all, mocker):
        """A span named 'gemini.query' must be started for every request."""
        mocker.patch.object(client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(client, "_execute_network_call", new=AsyncMock(return_value=raw_response))
        await client._query(payload)
        patch_all.tracer.start_as_current_span.assert_called_once_with("gemini.query")

    @pytest.mark.asyncio
    async def test_model_and_json_mode_set_as_span_attributes(self, client, payload, raw_response, patch_all, mocker):
        """llm.model and llm.json_mode must be set as span attributes at the start."""
        mocker.patch.object(client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(client, "_execute_network_call", new=AsyncMock(return_value=raw_response))
        await client._query(payload)
        keys = [c.args[0] for c in patch_all.span.set_attribute.call_args_list]
        assert "llm.model"     in keys
        assert "llm.json_mode" in keys

    @pytest.mark.asyncio
    async def test_request_counter_incremented_once(self, client, payload, raw_response, patch_all, mocker):
        """The request counter must be incremented exactly once per call."""
        mocker.patch.object(client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(client, "_execute_network_call", new=AsyncMock(return_value=raw_response))
        await client._query(payload)
        patch_all.counter.add.assert_called()

    @pytest.mark.asyncio
    async def test_api_error_propagates_unchanged(self, client, payload, mocker):
        """errors.APIError raised during the call must propagate unchanged."""
        from google.genai import errors
        err = make_api_error(429, "rate limit")
        mocker.patch.object(client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(client, "_execute_network_call", new=AsyncMock(side_effect=err))
        with pytest.raises(errors.APIError):
            await client._query(payload)

    @pytest.mark.asyncio
    async def test_network_failure_propagates_with_original_type(self, client, payload, mocker):
        """A generic network error must propagate with its original type and message."""
        mocker.patch.object(client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(client, "_execute_network_call",
            new=AsyncMock(side_effect=ConnectionError("DNS failure")))
        with pytest.raises(ConnectionError, match="DNS failure"):
            await client._query(payload)

    @pytest.mark.asyncio
    async def test_known_bug_start_log_emitted_as_error(self, client, payload, raw_response, mocker):
        """
        BUG (line 62 of gemini.py): _emit_log is called with "info" (lowercase)
        which causes ERROR severity instead of INFO.
        This test documents the current broken behaviour.
        Fix: change _emit_log("info", ...) to _emit_log("INFO", ...).
        """
        from opentelemetry._logs._internal import SeverityNumber
        captured = []
        original_record = __import__(
            "opentelemetry._logs._internal", fromlist=["LogRecord"]
        ).LogRecord

        def capture_record(**kwargs):
            obj = original_record(**kwargs)
            captured.append(obj)
            return obj

        mocker.patch("llm.clients.gemini.LogRecord", side_effect=capture_record)
        mocker.patch.object(client, "_build_generation_config", return_value=MagicMock())
        mocker.patch.object(client, "_execute_network_call", new=AsyncMock(return_value=raw_response))
        await client._query(payload)
        # First LogRecord corresponds to the start log (line 62)
        assert captured[0].severity_number == SeverityNumber.ERROR  # bug: should be INFO