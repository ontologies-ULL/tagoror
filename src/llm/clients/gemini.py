"""

"""

import time
import google.genai as genai
from google.genai import types

from llm.base_llm_client import BaseLLMClient
from llm.models import LLMPayload, LLMResponse
from observability.logger import ITelemetryLogger

class GeminiClient(BaseLLMClient):
    """
    Concrete adapter for the Google Gemini API.
    Implements the Facade pattern to hide the complexity of the Google SDK.
    """
    
    # TODO: This is a placeholder. You should replace it with the actual cost from Google's pricing page.
    _COST_PER_MILLION_TOKENS = 0.075 

    def __init__(self, api_key: str, logger: ITelemetryLogger):
        self._logger = logger
        self._client = genai.Client(api_key=api_key)

    async def _query(self, payload: LLMPayload) -> LLMResponse:
        """
        Main orchestrator for the adapter.
        """
        start_time = time.time()
        self._logger.log_event("INFO", f"Starting Gemini API request. Model: {payload.model_name}", "")

        try:
            model = self._build_model(payload)
            config = self._build_generation_config(payload)
            
            raw_response = await self._execute_network_call(model, payload.user_prompt, config)
            
            return self._map_to_domain_response(raw_response, start_time)

        except types.StopCandidateException as e:
            self._handle_security_block(e)
        except Exception as e:
            self._handle_network_failure(e)

    # --- Setup Methods ---

    def _build_model(self, payload: LLMPayload) -> str:
        """
        Isolates the model selection logic.
        """
        return payload.model_name

    def _build_generation_config(self, payload: LLMPayload) -> types.GenerateContentConfig:
        """
        Isolates hyperparameter and serialization configuration.
        """
        return types.GenerateContentConfig(
            temperature=payload.temperature,
            response_mime_type="application/json" if payload.json_mode else "text/plain"
        )

    # --- Execution & Mapping Methods ---

    async def _execute_network_call(self, model: str, prompt: str, config: types.GenerateContentConfig):
        """
        Strictly encapsulates the asynchronous I/O call.
        """
        return await self._client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )

    def _map_to_domain_response(self, raw_response, start_time: float) -> LLMResponse:
        """
        Acts as an Anti-Corruption Layer (ACL).
        Translates the proprietary Google object into our pure domain entity.
        """
        duration_ms = int((time.time() - start_time) * 1000)
        tokens = self._extract_token_usage(raw_response)
        estimated_cost = self._calculate_cost(tokens)

        self._logger.log_event("INFO", f"Request successful. Duration: {duration_ms}ms", "")

        return LLMResponse(
            raw_content=raw_response.text,
            tokens_consumed=tokens,
            duration_ms=duration_ms,
            cost=estimated_cost
        )

    # --- Telemetry & Pure Calculation Methods ---

    def _extract_token_usage(self, raw_response) -> int:
        if raw_response.usage_metadata:
            return raw_response.usage_metadata.total_token_count
        return 0

    def _calculate_cost(self, tokens: int) -> float:
        return (tokens / 1_000_000) * self._COST_PER_MILLION_TOKENS

    # --- Error Handling Methods ---

    def _handle_security_block(self, error: Exception):
        """
        Handles Google's specific safety block exceptions.
        """
        error_msg = f"Gemini blocked the response due to safety policies: {str(error)}"
        self._logger.log_event("ERROR", error_msg, str(error))
        raise Exception(error_msg) 

    def _handle_network_failure(self, error: Exception):
        """
        Handles generic I/O and network failures.
        """
        self._logger.log_event("ERROR", "Critical communication failure with Gemini API", str(error))
        raise error
