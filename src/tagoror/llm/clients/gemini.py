import instructor
from google import genai
from tagoror.llm.base_llm_client import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """
        Google Gemini adapter wrapping Google GenAI SDK.
    """

    def __init__(self, api_key: str):
        super().__init__(
            provider_name="gemini",
            cost_per_m_in=0.25,
            cost_per_m_out=1.50
        )
        raw_client = genai.Client(api_key=api_key)
        self._instructor_client = instructor.from_genai(
            client=raw_client.aio,
            mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
        )

    @property
    def client(self):
        return self._instructor_client
