"""
What the user picks is a *name* ("gemini/gemini-2.5-flash"); everything else
(price, rate limit, context window) is looked up here, so the user never types it.

Key format follows LiteLLM's convention: "<provider>/<model>".
  gemini/gemini-2.5-pro
  lm_studio/qwen/qwen3.6-35b-a3b      (local: the price is always 0)
"""

import difflib
import json
import urllib.request
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

PRICING_URL = (
  "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)


class Provider(str, Enum):
    GEMINI = "gemini"
    LM_STUDIO = "lm_studio"


class UnknownModelError(ValueError):
    """The requested model is not in the catalog."""


@dataclass(frozen=True)
class ModelProfile:
    """Everything the system needs to know about ONE concrete model. Price is on USD per million tokens."""
    provider: Provider
    model_name: str
    price_in_per_m: float 
    price_out_per_m: float
    requests_per_minute: float | None
    context_window: int | None 
    deprecation_date: str | None 
    pricing_source: str

    @property
    def key(self) -> str:
        return f"{self.provider.value}/{self.model_name}"


class ModelCatalog:
    """Resolves a model name into a ModelProfile using a saved copy of the price table."""

    def __init__(self, snapshot_path: Path) -> None:
        self._table: dict = json.loads(snapshot_path.read_text(encoding="utf-8"))
        saved_on = date.fromtimestamp(snapshot_path.stat().st_mtime).isoformat()
        self._source = f"litellm:{snapshot_path.name}@{saved_on}"

    def resolve(self, key: str) -> ModelProfile:
        provider_name, _, model_name = key.partition("/")
        if not model_name:
            raise UnknownModelError(
                f"'{key}' must look like '<provider>/<model>', "
                f"e.g. 'gemini/gemini-2.5-flash'. Providers: {[p.value for p in Provider]}"
            )
        try:
            provider = Provider(provider_name)
        except ValueError:
            raise UnknownModelError(
                f"Unknown provider '{provider_name}'. Valid: {[p.value for p in Provider]}"
            ) from None

        if provider is Provider.LM_STUDIO:
            return ModelProfile(provider, model_name, 0.0, 0.0, None, None, None, "local")
        return self._resolve_gemini(model_name)

    def _resolve_gemini(self, model_name: str) -> ModelProfile:
        entry = self._table.get(f"gemini/{model_name}")
        if entry is None:
            known = [k.removeprefix("gemini/") for k in self._table if k.startswith("gemini/")]
            close = difflib.get_close_matches(model_name, known, n=3)
            hint = f" Did you mean: {close}?" if close else ""
            raise UnknownModelError(f"'gemini/{model_name}' is not in the price table.{hint}")

        price_in = entry.get("input_cost_per_token")
        price_out = entry.get("output_cost_per_token")
        if price_in is None or price_out is None:
            # Never default to 0: a silent zero would corrupt the cost comparison.
            raise UnknownModelError(f"'gemini/{model_name}' has no token prices in the table.")

        return ModelProfile(
            provider=Provider.GEMINI,
            model_name=model_name,
            price_in_per_m=price_in * 1_000_000,
            price_out_per_m=price_out * 1_000_000,
            requests_per_minute=entry.get("rpm"),
            context_window=entry.get("max_input_tokens"),
            deprecation_date=entry.get("deprecation_date"),
            pricing_source=self._source,
        )


def download_snapshot(destination: Path) -> None:
    """Saves a fresh copy of the price table. Run it when you want to refresh prices."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(PRICING_URL, timeout=60) as response:
        data = response.read()
    json.loads(data) 
    destination.write_bytes(data)