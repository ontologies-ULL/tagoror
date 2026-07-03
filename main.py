import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# Carga de variables de entorno
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[ERROR] Falta la librería 'python-dotenv'.")
    sys.exit(1)

# Dependencias de OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Dependencias de Dominio (Asegúrate de que tu entorno 'uv' está bien instalado)
from core.pipeline.pipeline import Pipeline
from core.pipeline.orchestrator import EntityOrchestrator
from core.pipeline.evaluation.strategies.llm_auditor import LLMEntityAuditor
from core.pipeline.evaluation.strategies.majority_vote import ConsensusResolver
from core.prompt_manager import PromptManager
from serialization.serializers.turtle_serializer import TurtleSerializer
from llm.clients.gemini import GeminiClient 
from llm.retry import RetryableLLMClient
from llm.config import RetryPolicyConfig, BackoffStrategy
from aiolimiter import AsyncLimiter


def patch_owlready2():
    """Aplica parches de compatibilidad a owlready2 de forma aislada y controlada."""
    from owlready2 import Thing, World
    if not hasattr(Thing, "individual_id"):
        type.__setattr__(Thing, "individual_id", property(lambda self: self.name))
    if not hasattr(World.as_rdflib_graph, "_is_patched"):
        original_as_rdflib_graph = World.as_rdflib_graph
        def patched_as_rdflib_graph(self, *args, **kwargs):
            return original_as_rdflib_graph(self)
        patched_as_rdflib_graph._is_patched = True
        World.as_rdflib_graph = patched_as_rdflib_graph


def setup_telemetry(logs_dir: Path):
    """Configura OpenTelemetry centralizado en una sola función pura."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    provider = TracerProvider()
    
    # Exportador JSON detallado
    detailed_file = open(logs_dir / f"telemetry_detailed_{ts}.json", "w", encoding="utf-8")
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=detailed_file)))
    
    # Exportador humano
    def _human_formatter(span: ReadableSpan) -> str:
        dur = ((span.end_time or 0) - (span.start_time or 0)) / 1e6
        status = "🛑" if span.status.status_code.name == "ERROR" else "✅"
        return f"[{span.status.status_code.name}] {status} Span: '{span.name}' | Duración: {dur:.2f}ms\n"
        
    readable_file = open(logs_dir / f"telemetry_readable_{ts}.log", "w", encoding="utf-8")
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=readable_file, formatter=_human_formatter)))
    
    trace.set_tracer_provider(provider)


def save_results(ontology_name: str, model_name: str, results: list, out_dir: Path):
    """Escribe los resultados en disco sin necesidad de instanciar clases inútiles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace(" ", "_").replace("/", "_")
    
    file_path = out_dir / f"{ontology_name}_{safe_model}_{ts}.json"
    payload = {
        "ontology": ontology_name,
        "model": model_name,
        "executed_at": datetime.now().isoformat(),
        "results": results,
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    
    return file_path


async def run_batch(target_dir: Path, prompts_path: Path, model_name: str, rate_limit: float, user_input: str):
    """Bucle principal de ejecución asíncrona (Sustituye al PipelineExecutor)."""
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: La variable GEMINI_API_KEY no está definida en tu .env")
        sys.exit(1)

    if not target_dir.exists():
        print(f"❌ ERROR: El directorio objetivo '{target_dir}' no existe.")
        sys.exit(1)

    rdf_files = list(target_dir.glob("*.rdf"))
    if not rdf_files:
        print(f"⚠️  No se encontraron archivos .rdf en {target_dir}")
        sys.exit(0)

    # 1. Inyección de Dependencias Lineal y Limpia
    print("Iniciando dependencias...")
    base_llm = GeminiClient(api_key=api_key)
    retry_config = RetryPolicyConfig(max_retries=3, delay_between_retries=1.0, backoff_strategy=BackoffStrategy.EXPONENTIAL)
    llm_client = RetryableLLMClient(llm_client=base_llm, config=retry_config)
    
    prompt_manager = PromptManager(file_path=str(prompts_path))
    tracer = trace.get_tracer(__name__)
    serializer = TurtleSerializer(tracer)
    
    auditor = LLMEntityAuditor(
        model=llm_client,
        prompt_manager=prompt_manager,
        serializer=serializer,
        consensus_resolver=ConsensusResolver(),
        rate_limiter=AsyncLimiter(max_rate=rate_limit, time_period=60),
        user_input=user_input,
        suite_name="owl_validations",
        model_name=model_name
    )
    
    pipeline = Pipeline(orchestrator=EntityOrchestrator(strategy=auditor))
    results_dir = Path("results")
    aggregated_results = {}

    # 2. Bucle de ejecución
    print(f"\n🚀 Iniciando validación por lotes de {len(rdf_files)} ontologías...")
    print("-" * 60)
    
    for idx, rdf_file in enumerate(rdf_files, 1):
        print(f"⏳ [{idx}/{len(rdf_files)}] Procesando: {rdf_file.name}...")
        
        try:
            # Ejecución pura, sin acoplamiento a Spinners de consola
            execution_summaries = await pipeline.execute(str(rdf_file))
            
            file_results_dump = [s.model_dump(mode='json') for s in execution_summaries]
            aggregated_results[rdf_file.name] = file_results_dump
            
            saved_path = save_results(rdf_file.stem, model_name, file_results_dump, results_dir)
            
            exitos = sum(1 for s in execution_summaries if s.is_successful())
            fallos = len(execution_summaries) - exitos
            print(f"   ✅ Éxitos: {exitos} | ❌ Fallos: {fallos} -> Guardado en: {saved_path.name}")
            
        except Exception as e:
            print(f"   🔥 ERROR CRÍTICO procesando {rdf_file.name}: {str(e)}")
            aggregated_results[rdf_file.name] = {"error": str(e)}

    # Guardado final aglomerado
    with open("validation_results.json", "w", encoding="utf-8") as f:
        json.dump(aggregated_results, f, indent=4, ensure_ascii=False)
    
    print("\n✅ Ejecución finalizada. Revisa la carpeta 'results/'.")


def main():
    parser = argparse.ArgumentParser(description="Validación de Ontologías mediante LLM-as-a-Judge")
    parser.add_argument("--target-dir", default="target_ontologies", help="Directorio con archivos .rdf")
    parser.add_argument("--rate-limit", type=float, default=15.0, help="Límite de peticiones por minuto")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Modelo LLM a utilizar")
    parser.add_argument("--prompts", default="src/config/prompts.yaml", help="Ruta al archivo prompts.yaml")
    parser.add_argument("--user-input", default="Validate this healthcare ontology strictly.", help="Contexto de usuario")
    
    args = parser.parse_args()

    project_root = Path(__file__).parent
    
    patch_owlready2()
    setup_telemetry(project_root / "logs")
    
    try:
        asyncio.run(run_batch(
            target_dir=project_root / args.target_dir,
            prompts_path=project_root / args.prompts,
            model_name=args.model,
            rate_limit=args.rate_limit,
            user_input=args.user_input
        ))
    except KeyboardInterrupt:
        print("\n\n⚠️ Ejecución cancelada por el usuario.")
        sys.exit(0)

if __name__ == "__main__":
    main()