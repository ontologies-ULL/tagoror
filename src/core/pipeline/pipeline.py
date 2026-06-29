from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .extractor import OntologyExtractor
from .orchestrator import EntityOrchestrator
from core.models import ExecutionSummary


class Pipeline:
    
    def __init__(self, orchestrator: EntityOrchestrator) -> None:
        self._orchestrator = orchestrator
        self._tracer = trace.get_tracer(__name__)

    async def execute(self, file_path: str) -> list[ExecutionSummary]:
        with self._tracer.start_as_current_span("pipeline.execute") as span:
            span.set_attribute("ontology.file_path", file_path)
            
            try:
                individuals = OntologyExtractor.extract(file_path)
                base_ontology = OntologyExtractor.get_base_ontology()
                span.set_attribute("ontology.individuals_extracted", len(individuals))
                span.set_attribute("ontology.base_ontology", base_ontology)

                if not individuals:
                    span.set_status(Status(StatusCode.OK, "No individuals found to process"))
                    return []
                if not base_ontology:
                    span.set_status(Status(StatusCode.OK, "No base ontology found to process"))
                    return []

                results = await self._orchestrator.process(individuals, base_ontology)
                
                span.set_status(Status(StatusCode.OK))
                return results

            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, f"Pipeline catastrophically failed: {str(error)}"))
                span.record_exception(error)
                raise