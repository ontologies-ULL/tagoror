from dataclasses import dataclass

@dataclass(frozen=True)
class LLMPayload:
    """
    Represents the payload for an LLM request, including the prompt and any additional parameters.
    """
    prompt: str
    parameters: dict

@dataclass(frozen=True)
class LLMResponse:
    """
    Represents the response from an LLM, including the generated text and any additional metadata.
    """
    generated_text: str
    metadata: dict
