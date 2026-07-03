import json
from pathlib import Path
from datetime import datetime


class ResultsManager:
    """
    Manages the storage and organization of validation results for each ontology.
    Writes a per-ontology JSON file into `results/` containing metadata (ontology name,
    model used and timestamp) and the raw results payload.
    """

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.results_dir = self.base_path / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _safe(self, s: str) -> str:
        return s.replace(" ", "_").replace("/", "_")

    def write(self, ontology_filename: str, model_name: str, results) -> str:
        name = Path(ontology_filename).stem
        ts = self._timestamp()
        model_safe = self._safe(model_name)
        file_name = f"{name}_{model_safe}_{ts}.json"
        file_path = self.results_dir / file_name

        payload = {
            "ontology": name,
            "model": model_name,
            "executed_at": datetime.now().isoformat(),
            "results": results,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        return str(file_path)
