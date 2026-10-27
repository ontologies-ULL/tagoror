from typing import Any
from .base_llm_client import BaseLLMClient
from .config import RetryPolicyConfig
from .models import LLMPayload, LLMResponse
from tagoror.exceptions import TransientNetworkException, LLMParseException

class RetryableLLMClient(BaseLLMClient):
    """
    Proxy class that implements the BaseLLMClient interface and adds transparent
    resilience (retries) to any underlying LLM client.
    """
    
    def __init__(self, llm_client: BaseLLMClient, config: RetryPolicyConfig = RetryPolicyConfig):
        self.llm_client = llm_client
        self.config = config
        super().__init__(
            provider_name=getattr(llm_client, "provider_name", "retryable_proxy"),
            cost_per_m_in=getattr(llm_client, "cost_per_m_in", 0.0),
            cost_per_m_out=getattr(llm_client, "cost_per_m_out", 0.0),
        )

    @property
    def client(self) -> Any:
        """Forward access to the underlying instructor/async client."""
        return getattr(self.llm_client, "client", None)

    async def query(self, payload: LLMPayload) -> LLMResponse:
        max_attempts = self.config.max_retries
        for attempt in range(1, max_attempts + 1):
            try:
                return await self.llm_client.query(payload)
            except (TransientNetworkException, LLMParseException) as error:
                if attempt == max_attempts:
                    raise error
                delay = self._calculate_delay(attempt)
                await sleep(delay)
                
        raise RuntimeError("The retry loop terminated unexpectedly.")
