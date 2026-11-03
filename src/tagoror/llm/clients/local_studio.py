from typing import Any
import httpx
import instructor
from openai import AsyncOpenAI
from tagoror.llm.base_llm_client import BaseLLMClient


class LocalStudioClient(BaseLLMClient):
    """
    LM Studio adapter supporting dynamic model switching (JIT) and explicit load/unload.
    """

    def __init__(
        self, 
        base_host: str = "http://localhost:1234", 
        api_key: str = "lm-studio",
    ):
        super().__init__(
            provider_name="local_studio",
            cost_per_m_in=0.0,
            cost_per_m_out=0.0,
        )
        self._base_host = base_host.rstrip("/")
        self._api_key = api_key

        raw_client = AsyncOpenAI(
            base_url=f"{self._base_host}/v1", 
            api_key=self._api_key
        )
        self._instructor_client = instructor.from_openai(
            client=raw_client,
            mode=instructor.Mode.JSON,
        )

    @property
    def client(self) -> Any:
        return self._instructor_client

    async def load_model(self, model_name: str, context_length: int | None = None, ttl: int = 300) -> dict[str, Any]:
        """
        Explicitly load a model into VRAM via LM Studio native REST API v1.
        """
        payload: dict[str, Any] = {"model": model_name, "ttl": ttl}
        if context_length:
            payload["context_length"] = context_length

        async with httpx.AsyncClient(timeout=120.0) as http_client:
            response = await http_client.post(
                f"{self._base_host}/api/v1/models/load",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            return response.json()

    async def unload_model(self, instance_id: str) -> dict[str, Any]:
        """
        Explicitly unload a model from VRAM to free memory immediately.
        """
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.post(
                f"{self._base_host}/api/v1/models/unload",
                json={"instance_id": instance_id},
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            return response.json()
