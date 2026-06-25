from .extractor import OntologyExtractor
from .orchestrator import EntityOrchestrator
from core.models import ExecutionSummary

class ValidationPipeline:

    def __init__(self, orchestrator: EntityOrchestrator) -> None:
        self._orchestrator = orchestrator

    async def run(self, file_path: str) -> list[ExecutionSummary]:
        individuals = OntologyExtractor.extract(file_path)
        return await self._orchestrator.process(individuals)