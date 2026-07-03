"""
Concrete adapter for the Google Gemini API.
Implements the Facade pattern to hide the complexity of the Google SDK.
"""

import time
import google.genai as genai
from google.genai import types, errors

from opentelemetry import trace, metrics
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry._logs import get_logger
from opentelemetry._logs._internal import LogRecord, SeverityNumber
from pydantic import ValidationError

from llm.base_llm_client import BaseLLMClient
from llm.models import LLMPayload, LLMResponse

class GeminiClient(BaseLLMClient):
    """
    Concrete adapter for the Google Gemini API.
    Implements the Facade pattern to hide the complexity of the Google SDK.
    """
    
    _COST_PER_MILLION_INPUT_TOKENS = 0.25 
    _COST_PER_MILLION_OUTPUT_TOKENS = 1.50 

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


    async def query(self, payload: LLMPayload) -> LLMResponse:
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
 
            except errors.APIError as error:
                self._handle_error(error, span, "Gemini API error")
            except ValidationError as error:
                self._handle_error(error, span, "Response validation error")
            except Exception as error:
                self._handle_error(error, span, "Unexpected error")

    # --- setup methods ---
    
    def _build_generation_config(self, payload: LLMPayload) -> types.GenerateContentConfig:
        """builds the sdk config object from our domain payload."""
        config_args = {
            "temperature": payload.temperature,
            "system_instruction": payload.system_prompt
        }
        
        if payload.json_mode:
            config_args["response_mime_type"] = "application/json"
            config_args["response_schema"] = {
                "type": "OBJECT",
                "properties": {
                    "reasoning": {"type": "STRING", "description": "Step-by-step explanation of the validation. Look for possible hallucinations."},
                    "status": {
                        "type": "STRING", 
                        "enum": ["SUCCESS", "FAILURE"],
                        "description": "The final verdict. Must be strictly SUCCESS or FAILURE."
                    },
                    "findings": {
                        "type": "ARRAY", 
                        "items": {"type": "STRING"},
                        "description": "List of specific errors found."
                    }
                },
                "required": ["reasoning", "status", "findings"]
            }
        else:
            config_args["response_mime_type"] = "text/plain"
            
        return types.GenerateContentConfig(**config_args)


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

    def _map_to_domain_response(self, raw_response: str, start_time: float, span: Span, model_name: str) -> LLMResponse:
        """
        Acts as an Anti-Corruption Layer (ACL).
        Translates the proprietary Google object into our pure domain entity.
        """
        duration_ms = int((time.time() - start_time) * 1000)
        input_tokens, output_tokens = self._extract_token_usage(raw_response)
        total_tokens = input_tokens + output_tokens 
        output_cost        = self._calculate_cost(output_tokens, self._COST_PER_MILLION_OUTPUT_TOKENS)
        input_cost         = self._calculate_cost(input_tokens, self._COST_PER_MILLION_INPUT_TOKENS)
        total_cost = output_cost + input_cost

        # Record metrics
        self._duration_histogram.record(duration_ms, {"model": model_name})
        self._token_counter.add(total_tokens, {"model": model_name})
        self._cost_counter.add(total_cost, {"model": model_name})
 
        # Enrich span
        span.set_attribute("llm.tokens_consumed", total_tokens)
        span.set_attribute("llm.duration_ms", duration_ms)
        span.set_attribute("llm.cost_usd", total_cost)
        span.set_status(Status(StatusCode.OK))
 
        self._emit_log("INFO", f"Request successful. Duration: {duration_ms}ms, Tokens: {total_tokens}")
 
        return LLMResponse(
            raw_content=raw_response.text,
            tokens_consumed=total_tokens,
            duration_ms=duration_ms,
            cost=total_cost,
        )


    # --- Telemetry & Pure Calculation Methods ---

    def _extract_token_usage(self, raw_response) -> tuple[int, int]:
        if raw_response.usage_metadata:
            return (
                raw_response.usage_metadata.prompt_token_count,
                raw_response.usage_metadata.candidates_token_count
            )
        return (0, 0)

    def _calculate_cost(self, tokens: int, cost_per_million_tokens: float) -> float:
        return (tokens / 1_000_000) * cost_per_million_tokens

    # --- Error Handling Methods ---

    def _handle_error(self, error: Exception, span, error_msg: str):
        """
        Handles generic errors.
        """
        msg = f"{error_msg} (code={getattr(error, 'code', 'N/A')}): {str(error)}"
        self._emit_log("ERROR", msg)
        span.set_status(Status(StatusCode.ERROR, msg))
        span.record_exception(error)
        raise error
 
    # --- Logging Helper ---

    def _emit_log(self, level: str, message: str):
        """Emits a structured OTel log record at INFO or ERROR severity."""
        severity = SeverityNumber.INFO if level == "INFO" else SeverityNumber.ERROR
        self._logger.emit(LogRecord(severity_number=severity, body=message))
