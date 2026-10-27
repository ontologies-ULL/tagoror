import time
from abc import ABC, abstractmethod
from typing import Any

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry._logs import get_logger
from opentelemetry._logs._internal import LogRecord, SeverityNumber
from pydantic import ValidationError

from tagoror.llm.models import LLMPayload, LLMResponse


class BaseLLMClient(ABC):
    """
        Base interface for all LLM client
    """

    _PROMPT_TOKEN_KEYS = ("prompt_tokens", "input_tokens", "prompt_token_count")
    _COMPLETION_TOKEN_KEYS = ("completion_tokens", "output_tokens", "candidates_token_count")

    def __init__(self, provider_name: str, cost_per_m_in: float, cost_per_m_out: float):
        self.provider_name = provider_name
        self.cost_per_m_in = cost_per_m_in
        self.cost_per_m_out = cost_per_m_out

        self._tracer = trace.get_tracer("tagoror.llm")
        self._meter = metrics.get_meter("tagoror.llm")
        self._logger = get_logger("tagoror.llm")

        self._request_counter = self._meter.create_counter(
            name="llm.client.requests",
            description="Total number of LLM API requests",
            unit="1",
        )
        self._duration_histogram = self._meter.create_histogram(
            name="llm.client.duration",
            description="Duration of LLM API requests",
            unit="ms",
        )
        self._token_counter = self._meter.create_counter(
            name="llm.client.tokens",
            description="Total tokens consumed by LLM API requests",
            unit="1",
        )
        self._cost_counter = self._meter.create_counter(
            name="llm.client.cost",
            description="Estimated cost of LLM API requests",
            unit="USD",
        )

    @property
    @abstractmethod
    def client(self) -> Any:
        """
            Return the initialized Instructor async client.
        """
        pass

    async def query(self, payload: LLMPayload) -> LLMResponse:
        """
            Execute the query lifecycle, record metrics, and trace the operation.
        """
        with self._tracer.start_as_current_span(f"{self.provider_name}.query") as span:
            span.set_attribute("llm.model", payload.model_name)
            start_time = time.time()
            self._request_counter.add(1, {"model": payload.model_name})

            messages = []
            if payload.system_prompt:
                messages.append({"role": "system", "content": payload.system_prompt})
            messages.append({"role": "user", "content": payload.user_prompt})

            kwargs: dict[str, Any] = {
                "model": payload.model_name,
                "messages": messages,
                "temperature": payload.temperature,
            }
            if payload.response_model:
                kwargs["response_model"] = payload.response_model
                kwargs["max_retries"] = 2

            try:
                parsed, raw_response = await self.client.chat.completions.create_with_completion(**kwargs)
                return self._build_domain_response(raw_response, parsed, start_time, span, payload.model_name)
            except ValidationError as error:
                self._handle_error(error, span, "Instructor validation error")
            except Exception as error:
                self._handle_error(error, span, f"{self.provider_name} API error")

    def _extract_tokens(self, raw_response: Any) -> tuple[int, int]:
        """
            Extract prompt and candidate token counts iterating over potential vendor fields.
        """
        usage = getattr(raw_response, "usage", None) or getattr(raw_response, "usage_metadata", None)
        if not usage:
            return 0, 0

        prompt_tokens = 0
        for key in self._PROMPT_TOKEN_KEYS:
            val = getattr(usage, key, None)
            if isinstance(val, int):
                prompt_tokens = val
                break

        completion_tokens = 0
        for key in self._COMPLETION_TOKEN_KEYS:
            val = getattr(usage, key, None)
            if isinstance(val, int):
                completion_tokens = val
                break

        return prompt_tokens, completion_tokens

    def _build_domain_response(
        self, raw_response: Any, parsed: Any, start_time: float, span, model_name: str
    ) -> LLMResponse:
        """
            Map raw and parsed outputs into a domain LLMResponse with telemetry data.
        """
        duration_ms = int((time.time() - start_time) * 1000)
        in_tokens, out_tokens = self._extract_tokens(raw_response)
        total_tokens = in_tokens + out_tokens

        cost = ((in_tokens / 1_000_000) * self.cost_per_m_in) + ((out_tokens / 1_000_000) * self.cost_per_m_out)

        self._duration_histogram.record(duration_ms, {"model": model_name})
        self._token_counter.add(total_tokens, {"model": model_name})
        self._cost_counter.add(cost, {"model": model_name})

        span.set_attribute("llm.tokens_consumed", total_tokens)
        span.set_attribute("llm.cost_usd", cost)
        span.set_status(Status(StatusCode.OK))

        raw_text = getattr(raw_response, "text", None) or str(raw_response)

        return LLMResponse(
            raw_content=raw_text,
            parsed=parsed,
            tokens_consumed=total_tokens,
            duration_ms=duration_ms,
            cost=cost,
        )

    def _handle_error(self, error: Exception, span, error_msg: str):
        """Emit telemetry error logs and re-raise the exception."""
        msg = f"{error_msg} (code={getattr(error, 'code', 'N/A')}): {str(error)}"
        self._emit_log("ERROR", msg)
        span.set_status(Status(StatusCode.ERROR, msg))
        span.record_exception(error)
        raise error

    def _emit_log(self, level: str, message: str):
        """Emit a structured OpenTelemetry log record."""
        severity = SeverityNumber.INFO if level == "INFO" else SeverityNumber.ERROR
        self._logger.emit(LogRecord(severity_number=severity, body=message))
