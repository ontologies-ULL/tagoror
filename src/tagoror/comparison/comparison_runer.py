"""
The loop that did not exist: for each chosen model, audit every RDF file.

    for model in chosen models:          <- outer loop: the model (VRAM!)
        load it
        for rdf in files:                <- inner loop: what main.py already does
            audit(rdf)
        unload it
"""

from pathlib import Path
from typing import TYPE_CHECKING

from src.tagoror.llm.model_catalog import ModelCatalog, ModelProfile
from .result_store import ResultStore
from .run_config import RunConfig

if TYPE_CHECKING:
    from .audit_service_factory import AuditServiceFactory


class RunListener:
    """
    What the UI (CLI spinner, web, ...) implements to follow the progress.
    All methods do nothing by default, so the runner never prints anything itself.
    """
    def model_started(self, profile: ModelProfile) -> None: ...
    def model_failed(self, profile: ModelProfile, error: Exception) -> None: ...
    def file_started(self, profile: ModelProfile, rdf_path: Path) -> None: ...
    def file_finished(self, profile: ModelProfile, rdf_path: Path, results: list) -> None: ...
    def file_failed(self, profile: ModelProfile, rdf_path: Path, error: Exception) -> None: ...


class ModelComparisonRunner:

    def __init__(
        self,
        catalog: ModelCatalog,
        factory: "AuditServiceFactory",
        store: ResultStore,
        listener: RunListener | None = None,
    ) -> None:
        self._catalog = catalog
        self._factory = factory
        self._store = store
        self._listener = listener or RunListener()

    async def run(self, config: RunConfig) -> Path:
        # Resolve ALL names first: a typo in the third model must fail now,
        # not after the first two models have already been paid for.
        profiles = [self._catalog.resolve(key) for key in config.model_keys]
        rdf_files = sorted(config.target_dir.glob("*.rdf"))
        if not rdf_files:
            raise FileNotFoundError(f"No .rdf files found in {config.target_dir}")

        for profile in profiles:
            await self._run_model(profile, rdf_files)
        return self._store.write_index()

    async def _run_model(self, profile: ModelProfile, rdf_files: list[Path]) -> None:
        self._listener.model_started(profile)
        try:
            client = self._factory.create_client(profile)
            async with client:                       # Gemini: nothing. LM Studio: load / unload.
                service = self._factory.create_service(client, profile)
                for rdf_path in rdf_files:
                    await self._audit_file(service, profile, rdf_path)
        except Exception as error:                   # e.g. the model failed to load
            self._store.save_model_error(profile, error)
            self._listener.model_failed(profile, error)

    async def _audit_file(self, service, profile: ModelProfile, rdf_path: Path) -> None:
        self._listener.file_started(profile, rdf_path)
        try:
            results = await service.audit(rdf_path)
        except Exception as error:                   # one bad file never stops the run
            self._store.save_file_error(profile, rdf_path, error)
            self._listener.file_failed(profile, rdf_path, error)
            return
        self._store.save_file_result(profile, rdf_path, results)
        self._listener.file_finished(profile, rdf_path, results)