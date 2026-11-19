"""
What the user picks is a *name* ("gemini/gemini-2.5-flash"); everything else
(price, rate limit, context window) is looked up here.

Key format follows LiteLLM's convention: "<provider>/<model>".
"""

import difflib
import json
import urllib.request
from datetime import date
from pathlib import Path

from tagoror.exceptions import UnknownModelException
from tagoror.llm.models import ModelProfile, Provider

PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
TOKENS_PER_MILLION = 1_000_000


class ModelCatalog:
    """Resolves a model name into a ModelProfile using a saved copy of the LiteLLM price table."""

    def __init__(self, snapshot_path: Path) -> None:
        self._table: dict = json.loads(snapshot_path.read_text(encoding="utf-8"))
        saved_on = date.fromtimestamp(snapshot_path.stat().st_mtime).isoformat()
        self._source = f"litellm:{snapshot_path.name}@{saved_on}"
        self._resolvers = {
            Provider.GEMINI: self._resolve_gemini,
            Provider.LM_STUDIO: self._resolve_lm_studio,
        }

    def resolve(self, key: str) -> ModelProfile:
        provider, model_name = self._split(key)
        return self._resolvers[provider](model_name)

    @staticmethod
    def download_snapshot(destination: Path) -> None:
        """Saves a fresh copy of the price table. Run it when you want to refresh prices."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(PRICING_URL, timeout=60) as response:
            data = response.read()
        json.loads(data)
        temp = destination.with_suffix(".tmp")
        temp.write_bytes(data)
        temp.replace(destination)

    @staticmethod
    def _split(key: str) -> tuple[Provider, str]:
        provider_name, _, model_name = key.partition("/")
        valid = [p.value for p in Provider]
        if not model_name:
            raise UnknownModelException(
                f"'{key}' must look like '<provider>/<model>', e.g. 'gemini/gemini-2.5-flash'. Providers: {valid}"
            )
        try:
            return Provider(provider_name), model_name
        except ValueError:
            raise UnknownModelException(f"Unknown provider '{provider_name}'. Valid: {valid}") from None

    def _resolve_lm_studio(self, model_name: str) -> ModelProfile:
        return ModelProfile(
            provider=Provider.LM_STUDIO,
            model_name=model_name,
            price_in_per_m=0.0,
            price_out_per_m=0.0,
            pricing_source="local",
        )

    def _resolve_gemini(self, model_name: str) -> ModelProfile:
        entry = self._table.get(f"gemini/{model_name}")
        if entry is None:
            raise UnknownModelException(self._not_found_message(model_name))

        price_in = entry.get("input_cost_per_token")
        price_out = entry.get("output_cost_per_token")
        if price_in is None or price_out is None:
            raise UnknownModelException(f"'gemini/{model_name}' has no token prices in the table.")

        return ModelProfile(
            provider=Provider.GEMINI,
            model_name=model_name,
            price_in_per_m=price_in * TOKENS_PER_MILLION,
            price_out_per_m=price_out * TOKENS_PER_MILLION,
            requests_per_minute=entry.get("rpm"),
            context_window=entry.get("max_input_tokens"),
            deprecation_date=entry.get("deprecation_date"),
            pricing_source=self._source,
        )

    def _not_found_message(self, model_name: str) -> str:
        known = [k.removeprefix("gemini/") for k in self._table if k.startswith("gemini/")]
        close = difflib.get_close_matches(model_name, known, n=3)
        hint = f" Did you mean: {close}?" if close else ""
        return f"'gemini/{model_name}' is not in the price table.{hint}"
