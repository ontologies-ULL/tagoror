"""
Writes the results so that two models (or two runs) never overwrite each other.

  results/<run_id>/<provider>_<model>/<rdf_stem>.json
  results/<run_id>/run_index.json

Infrastructure errors (no connection, model failed to load) are stored as
status="error", NOT as a verdict: otherwise a network failure would look like
"the model failed to detect the hallucination".
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.tagoror.core.audit_models import EntityAuditResult
from src.tagoror.llm.model_catalog import ModelProfile


class ResultStore:

    def __init__(self, results_dir: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = results_dir / self.run_id
        self._index: list[dict] = []

    def save_file_result(self, profile: ModelProfile, rdf_path: Path, results: list[EntityAuditResult]) -> Path:
        return self._write(profile, rdf_path, "ok", {"results": [r.model_dump(mode="json") for r in results]})

    def save_file_error(self, profile: ModelProfile, rdf_path: Path, error: Exception) -> Path:
        return self._write(profile, rdf_path, "error", {"error": f"{type(error).__name__}: {error}"})

    def save_model_error(self, profile: ModelProfile, error: Exception) -> None:
        """The model could not even be loaded: no file was audited with it."""
        self._index.append(
            {"model": profile.key, "rdf": None, "status": "error", "error": f"{type(error).__name__}: {error}"}
        )

    def write_index(self) -> Path:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        path = self._run_dir / "run_index.json"
        path.write_text(json.dumps(self._index, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _write(self, profile: ModelProfile, rdf_path: Path, status: str, body: dict) -> Path:
        model_dir = self._run_dir / profile.key.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{rdf_path.stem}.json"
        document = {
            "model": profile.key,
            "rdf": rdf_path.name,
            "status": status,
            "executed_at": datetime.now().isoformat(),
            # Reproducibility: which prices were used for the cost figures.
            "profile": asdict(profile),
            **body,
        }
        path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        self._index.append(
            {"model": profile.key, "rdf": rdf_path.name, "status": status, "path": str(path.relative_to(self._run_dir))}
        )
        return path