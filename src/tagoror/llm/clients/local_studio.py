from typing import Any
import instructor
from openai import AsyncOpenAI
from tagoror.llm.base_llm_client import BaseLLMClient


class LocalStudioClient(BaseLLMClient):
    """
        LM Studio adapter exposing an OpenAI-compatible interface wrapped with Instructor.
    """

    def __init__(self, target_url: str = "http://localhost:1234/v1", api_key: str = "not-needed"):
        super().__init__(
            provider_name="local_studio",
            cost_per_m_in=0.0,
            cost_per_m_out=0.0,
        )
        raw_client = AsyncOpenAI(base_url=target_url, api_key=api_key)
        self._instructor_client = instructor.from_openai(
            client=raw_client,
            mode=instructor.Mode.JSON,
        )

    @property
    def client(self) -> Any:
        return self._instructor_client
