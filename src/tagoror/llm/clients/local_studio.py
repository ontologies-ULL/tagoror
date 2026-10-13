from opentelemetry import trace, metrics
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry._logs import get_logger
from opentelemetry._logs._internal import LogRecord, SeverityNumber

from tagoror.llm.base_llm_client import BaseLLMClient
from tagoror.llm.models import LLMPayload, LLMResponse

class LocalStudioClient(BaseLLMClient):
    """
    Concrete adapter for LLm Studio API.
    """

    def __init__(self, target_url: str, api_key: str | None = None):
        self._target_url = target_url
        self._api_key = api_key
        super().__init__(provider_name="local_studio") 
        
    def query(self, payload: LLMPayload) -> LLMResponse:
        """
        Query the LLM Studio API with a given payload and return the response.
        """
        with self._tracer.start_as_current_span("local_studio.query") as span:
            span.set_attribute("llm.model", payload.model_name)

