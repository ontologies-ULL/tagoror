from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RunConfig(BaseModel):
    """
    Exactly what the end user decides. Replaces the loose `config` dict of main.py.
    Prices and rate limits are NOT here: they come from the ModelCatalog.

    Pydantic gives us, for free: a str from the CLI becomes a Path, and an empty
    model list is rejected before anything runs.
    """
    model_config = ConfigDict(frozen=True)

    model_keys: tuple[str, ...] = Field(min_length=1)
    target_dir: Path
    user_input: str
    results_dir: Path = Path("results")