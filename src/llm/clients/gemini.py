"""

"""

import time
import google.genai as genai
from google.genai import types, errors

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry._logs import get_logger
from opentelemetry._logs._internal import LogRecord, SeverityNumber

from llm.base_llm_client import BaseLLMClient
from llm.models import LLMPayload, LLMResponse

class GeminiClient(BaseLLMClient):
    """
    Concrete adapter for the Google Gemini API.
    Implements the Facade pattern to hide the complexity of the Google SDK.
    """
    
    # TODO: This is a placeholder. You should replace it with the actual cost from Google's pricing page.
    _COST_PER_MILLION_TOKENS = 0.075 

    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)
 
        self._tracer = trace.get_tracer(__name__)
        self._meter  = metrics.get_meter(__name__)
        self._logger = get_logger(__name__)
 
        self._request_counter = self._meter.create_counter(
            name="llm.gemini.requests",
            description="Total number of Gemini API requests",
            unit="1",
        )
        self._duration_histogram = self._meter.create_histogram(
            name="llm.gemini.duration",
            description="Duration of Gemini API requests",
            unit="ms",
        )
        self._token_counter = self._meter.create_counter(
            name="llm.gemini.tokens",
            description="Total tokens consumed by Gemini API requests",
            unit="1",
        )
        self._cost_counter = self._meter.create_counter(
            name="llm.gemini.cost",
            description="Estimated cost of Gemini API requests",
            unit="USD",
        )


    async def _query(self, payload: LLMPayload) -> LLMResponse:
        """main orchestrator — one otel span wraps the full request lifecycle."""
        with self._tracer.start_as_current_span("gemini.query") as span:
            span.set_attribute("llm.model", payload.model_name)
            span.set_attribute("llm.json_mode", payload.json_mode)
            start_time = time.time()
 
            self._emit_log("info", f"starting gemini request. model: {payload.model_name}")
            self._request_counter.add(1, {"model": payload.model_name})
 
            try:
                config = self._build_generation_config(payload)
                raw    = await self._execute_network_call(payload.model_name, payload.user_prompt, config)
                return self._map_to_domain_response(raw, start_time, span, payload.model_name)
 
            except errors.APIError as e:
                self._handle_api_error(e, span)
            except Exception as e:
                self._handle_network_failure(e, span)

    # --- setup methods ---
    
    def _build_generation_config(self, payload: LLMPayload) -> types.GenerateContentConfig:
        """builds the sdk config object from our domain payload."""
        return types.GenerateContentConfig(
            temperature=payload.temperature,
            response_mime_type="application/json" if payload.json_mode else "text/plain",
        )

    # --- execution & mapping methods ---

    async def _execute_network_call(self, model_name: str, prompt: str, config: types.GenerateContentConfig):
        """
        strictly encapsulates the asynchronous i/o call.
        """
        return await self._client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

    def _map_to_domain_response(self, raw_response, start_time, span, model_name: str) -> LLMResponse:
        """
        Acts as an Anti-Corruption Layer (ACL).
        Translates the proprietary Google object into our pure domain entity.
        """
        duration_ms = int((time.time() - start_time) * 1000)
        tokens       = self._extract_token_usage(raw_response)
        cost         = self._calculate_cost(tokens)
 
        # Record metrics
        self._duration_histogram.record(duration_ms, {"model": model_name})
        self._token_counter.add(tokens, {"model": model_name})
        self._cost_counter.add(cost, {"model": model_name})
 
        # Enrich span
        span.set_attribute("llm.tokens_consumed", tokens)
        span.set_attribute("llm.duration_ms", duration_ms)
        span.set_attribute("llm.cost_usd", cost)
        span.set_status(Status(StatusCode.OK))
 
        self._emit_log("INFO", f"Request successful. Duration: {duration_ms}ms, Tokens: {tokens}")
 
        return LLMResponse(
            raw_content=raw_response.text,
            tokens_consumed=tokens,
            duration_ms=duration_ms,
            cost=cost,
        )


    # --- Telemetry & Pure Calculation Methods ---

    def _extract_token_usage(self, raw_response) -> int:
        if raw_response.usage_metadata:
            return raw_response.usage_metadata.total_token_count
        return 0

    def _calculate_cost(self, tokens: int) -> float:
        return (tokens / 1_000_000) * self._COST_PER_MILLION_TOKENS

    # --- Error Handling Methods ---

    def _handle_api_error(self, error: errors.APIError, span):
        """
        Handles errors raised by the Gemini API (e.g. safety blocks, quota exceeded).
        errors.APIError exposes .code (HTTP status) and .message.
        """
        msg = f"Gemini API error (code={error.code}): {error.message}"
        self._emit_log("ERROR", msg)
        span.set_status(Status(StatusCode.ERROR, msg))
        span.record_exception(error)
        raise error
 
    def _handle_network_failure(self, error: Exception, span):
        """Handles generic I/O and transport-level failures."""
        msg = "Critical communication failure with Gemini API"
        self._emit_log("ERROR", msg)
        span.set_status(Status(StatusCode.ERROR, msg))
        span.record_exception(error)
        raise error

    # --- Logging Helper ---

    def _emit_log(self, level: str, message: str):
        """Emits a structured OTel log record at INFO or ERROR severity."""
        severity = SeverityNumber.INFO if level == "INFO" else SeverityNumber.ERROR
        self._logger.emit(LogRecord(severity_number=severity, body=message))
