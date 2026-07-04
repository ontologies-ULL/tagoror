"""
Main entry point for Ontology Validation.
Includes interactive CLI mode and real-time visual feedback (Async Spinner).
"""

import os
import sys
import json
import asyncio
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[ERROR] Missing 'python-dotenv' library.")
    sys.exit(1)

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from tagoror.core.pipeline.pipeline import Pipeline
from tagoror.core.pipeline.orchestrator import EntityOrchestrator
from tagoror.core.pipeline.evaluation.strategies.llm_auditor import LLMEntityAuditor
from tagoror.core.pipeline.evaluation.strategies.majority_vote import ConsensusResolver
from tagoror.core.prompt_manager import PromptManager
from tagoror.serialization.serializers.turtle_serializer import TurtleSerializer 
from tagoror.llm.clients.gemini import GeminiClient 
from tagoror.llm.retry import RetryableLLMClient
from tagoror.llm.config import RetryPolicyConfig, BackoffStrategy
from aiolimiter import AsyncLimiter

# =====================================================================
# 1. VISUAL COMPONENTS & STATE TRACKING (UI & Tracking)
# =====================================================================

class AsyncSpinner:
    """Asynchronous terminal animation that does not block validation."""
    def __init__(self, filename: str):
        self.filename = filename
        self._task = None
        self._start_time = 0.0

    async def _spin(self):
        chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i = 0
        self._start_time = time.time()
        try:
            while True:
                elapsed = time.time() - self._start_time
                msg = f"\r   {chars[i % len(chars)]} Processing '{self.filename}' [{elapsed:.1f}s]"
                sys.stdout.write(msg)
                sys.stdout.flush()
                i += 1
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            sys.stdout.write("\r" + " " * 120 + "\r")
            sys.stdout.flush()

    def start(self):
        self._task = asyncio.create_task(self._spin())

    async def stop(self):
        if self._task:
            self._task.cancel()
            await self._task


# =====================================================================
# 2. CONFIGURATION & INTERACTIVE MENUS
# =====================================================================

def run_interactive_menu() -> dict:
    print("\n" + "=" * 60)
    print(" 🚀 ONTOLOGY VALIDATION - INTERACTIVE SETUP")
    print("=" * 60 + "\n")

    def_dir = "target_ontologies"
    t_dir = input(f"📁 Ontology directory (Enter for '{def_dir}'): ").strip()
    target_dir = Path(t_dir or def_dir)

    def_rate = "10"
    r_limit = input(f"🚦 Rate limit [requests/min] (Enter for {def_rate}): ").strip()
    rate_limit = float(r_limit) if r_limit else float(def_rate)

    models = ["gemini-3.1-flash-lite", "gemini-3.1-flash", "gemini-2.5-flash"]
    print("\n🤖 Available Models:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")
    m_choice = input("Select a number (Default 1): ").strip()
    idx = int(m_choice) - 1 if m_choice.isdigit() and 0 < int(m_choice) <= len(models) else 0
    model_name = models[idx]

    def_input = "Validate this healthcare ontology strictly."
    user_input = input(f"\n💭 Contextual Hallucination Prompt (Enter for default):\n> ").strip()
    user_input = user_input or def_input

    print("\n" + "-" * 60)
    print("📋 SUMMARY:")
    print(f"  Target: {target_dir} | Rate: {rate_limit}/min | Model: {model_name}")
    print("-" * 60)
    
    if input("Proceed with execution? [Y/n]: ").strip().lower() in ["n", "no"]:
        print("Cancelled by user.")
        sys.exit(0)

    return {
        "target_dir": target_dir,
        "rate_limit": rate_limit,
        "model_name": model_name,
        "user_input": user_input,
        "prompts_path": Path("src/tagoror/config/prompts.yaml")
    }


def patch_owlready2():
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
    readable_dir = logs_dir / "readable"
    detailed_dir = logs_dir / "detailed"
    readable_dir.mkdir(parents=True, exist_ok=True)
    detailed_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    provider = TracerProvider()
    
    detailed_file = open(detailed_dir / f"telemetry_detailed_{ts}.json", "w", encoding="utf-8")
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=detailed_file)))
    
    # ---------------------------------------------------------
    # EL NUEVO FORMATEADOR: EXTRAE Y MUESTRA LA EXCEPCIÓN
    # ---------------------------------------------------------
    def _human_formatter(span: ReadableSpan) -> str:
        dur = ((span.end_time or 0) - (span.start_time or 0)) / 1e6
        status = "🛑" if span.status.status_code.name == "ERROR" else "✅"
        out = f"[{span.status.status_code.name}] {status} Span: '{span.name}' | Duration: {dur:.2f}ms\n"
        
        if span.status.status_code.name == "ERROR":
            if span.status.description:
                out += f"    👉 Description: {span.status.description}\n"
            if span.events:
                for event in span.events:
                    if event.attributes and "exception.type" in event.attributes:
                        exc_type = event.attributes.get("exception.type", "UnknownException")
                        exc_msg = event.attributes.get("exception.message", "No message provided")
                        stack = event.attributes.get("exception.stacktrace", "")
                        out += f"    🔥 Exception ({exc_type}): {exc_msg}\n"
                        if stack:
                            out += f"    📜 StackTrace:\n{stack}\n"
        return out
        
    readable_file = open(readable_dir / f"telemetry_readable_{ts}.log", "w", encoding="utf-8")
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=readable_file, formatter=_human_formatter)))
    trace.set_tracer_provider(provider)


def save_results(ontology_name: str, model_name: str, results: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace(" ", "_").replace("/", "_")
    file_path = out_dir / f"{ontology_name}_{safe_model}_{ts}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"ontology": ontology_name, "model": model_name, "executed_at": datetime.now().isoformat(), "results": results}, f, indent=4, ensure_ascii=False)
    return file_path


# =====================================================================
# 3. EXECUTION CORE (Pipeline)
# =====================================================================

async def run_batch(config: dict):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY environment variable is missing in .env")
        sys.exit(1)

    target_dir = config["target_dir"]
    if not target_dir.exists():
        print(f"❌ ERROR: Target directory '{target_dir}' does not exist.")
        sys.exit(1)

    rdf_files = list(target_dir.glob("*.rdf"))
    if not rdf_files:
        print(f"⚠️  No .rdf files found in {target_dir}")
        sys.exit(0)

    base_llm = GeminiClient(api_key=api_key)
    llm_client = RetryableLLMClient(base_llm, RetryPolicyConfig(max_retries=3, delay_between_retries=1.0, backoff_strategy=BackoffStrategy.EXPONENTIAL))
    
    try:
        serializer_instance = TurtleSerializer(tracer=trace.get_tracer(__name__))
    except TypeError:
        serializer_instance = TurtleSerializer()

    auditor = LLMEntityAuditor(
        model=llm_client,
        prompt_manager=PromptManager(file_path=str(config["prompts_path"])),
        serializer=serializer_instance,
        consensus_resolver=ConsensusResolver(),
        rate_limiter=AsyncLimiter(max_rate=config["rate_limit"], time_period=60),
        user_input=config["user_input"],
        suite_name="owl_validations",
        model_name=config["model_name"]
    )
    
    pipeline = Pipeline(orchestrator=EntityOrchestrator(strategy=auditor))
    aggregated_results = {}
    results_dir = Path("results")

    print(f"\n⚙️  Processing {len(rdf_files)} RDF file(s)...")
    print("-" * 60)
    
    for rdf_file in rdf_files:
        spinner = AsyncSpinner(rdf_file.name)
        spinner.start()
        
        try:
            execution_summaries = await pipeline.execute(str(rdf_file))
            
            file_results_dump = [s.model_dump(mode='json') for s in execution_summaries]
            aggregated_results[rdf_file.name] = file_results_dump
            saved_path = save_results(rdf_file.stem, config["model_name"], file_results_dump, results_dir)
            
            successes = sum(1 for s in execution_summaries if s.is_successful())
            failures = len(execution_summaries) - successes
            
            await spinner.stop()
            print(f"✅ {rdf_file.name} | Passed: {successes} | Failed: {failures} -> {saved_path.name}")
            
        except Exception as e:
            await spinner.stop()
            print(f"🔥 ERROR in {rdf_file.name}: {str(e)}")
            aggregated_results[rdf_file.name] = {"error": str(e)}

    aggregated_path = results_dir / "validation_results_aggregated.json"
    with open(aggregated_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n🎉 Execution completed. Check the '{results_dir}' folder.")


def main():
    parser = argparse.ArgumentParser(description="Ontology Validation Client")
    parser.add_argument("--interactive", action="store_true", help="Force interactive menu")
    parser.add_argument("--target-dir", default="target_ontologies", help="Directory with .rdf files")
    parser.add_argument("--rate-limit", type=float, default=15.0, help="Rate limit [requests/min]")
    parser.add_argument("--model", default="gemini-2.5-flash", help="LLM Model")
    parser.add_argument("--user-input", default="Validate this healthcare ontology strictly.", help="User context input")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    patch_owlready2()
    setup_telemetry(project_root / "logs")

    use_interactive = args.interactive or len(sys.argv) == 1

    try:
        if use_interactive:
            config = run_interactive_menu()
        else:
            config = {
                "target_dir": project_root / args.target_dir,
                "rate_limit": args.rate_limit,
                "model_name": args.model,
                "user_input": args.user_input,
                "prompts_path": project_root / "src" / "tagoror" / "config" / "prompts.yaml"
            }
            
        asyncio.run(run_batch(config))
    except KeyboardInterrupt:
        print("\n\n⚠️ Execution cancelled by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()