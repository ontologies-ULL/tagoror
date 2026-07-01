"""
Main entry point for the Ontology Validation application.

This script demonstrates how an end-user or external system would initialize
the architecture, configure the AI models, and run the pipeline against a
real OWL ontology file.

Usage:
    uv run main.py [path_to_ontology_file]
"""

import asyncio
import os
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv
except ImportError:
    print("[ERROR] Missing 'python-dotenv' library.")
    print("Please install it using: pip install python-dotenv")
    sys.exit(1)

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from core.pipeline.pipeline import Pipeline
from core.pipeline.orchestrator import EntityOrchestrator
from core.pipeline.evaluation.strategies.llm_auditor import LLMEntityAuditor
from core.pipeline.evaluation.strategies.majority_vote import ConsensusResolver
from core.prompt_manager import PromptManager

from serialization.serializers.turtle_serializer import TurtleSerializer
from llm.clients.gemini import GeminiClient 
from llm.retry import RetryableLLMClient
from llm.config import RetryPolicyConfig, BackoffStrategy

from owlready2 import Thing, World

from aiolimiter import AsyncLimiter


def human_readable_formatter(span: ReadableSpan) -> str:
    """
    Translates raw OpenTelemetry JSON spans into a clean, human-readable text format.
    """
    status_name = span.status.status_code.name
    icon = "🛑" if status_name == "ERROR" else "✅"
    
    # Calculate duration in milliseconds
    duration_ms = 0.0
    if span.end_time and span.start_time:
        duration_ms = (span.end_time - span.start_time) / 1e6

    output = f"[{status_name}] {icon} Span: '{span.name}' | Duration: {duration_ms:.2f}ms\n"
    
    if span.attributes:
        output += "  Attributes:\n"
        for key, value in span.attributes.items():
            output += f"    - {key}: {value}\n"
            
    if span.events:
        output += "  Events/Errors:\n"
        for event in span.events:
            # Format exceptions cleanly
            if event.attributes and "exception.type" in event.attributes:
                exc_type = event.attributes.get("exception.type")
                exc_msg = event.attributes.get("exception.message")
                output += f"    -> {exc_type}: {exc_msg}\n"
            else:
                output += f"    -> Event: {event.name}\n"
                
    output += "-" * 60 + "\n"
    return output

def setup_telemetry(readable_log_path: str, detailed_json_path: str):
    """
    Configures OpenTelemetry to export all traces and spans to a local file
    instead of the standard output or an external server.
    """
    provider = TracerProvider()
    
    readable_file = open(readable_log_path, "w", encoding="utf-8")
    readable_exporter = ConsoleSpanExporter(out=readable_file, formatter=human_readable_formatter)
    provider.add_span_processor(SimpleSpanProcessor(readable_exporter))

    detailed_file = open(detailed_json_path, "w", encoding="utf-8")
    detailed_exporter = ConsoleSpanExporter(out=detailed_file)
    provider.add_span_processor(SimpleSpanProcessor(detailed_exporter))
    
    trace.set_tracer_provider(provider)


async def show_spinner(task: asyncio.Task, filename: str):
    """
    Displays a visual spinner and elapsed time in the console while an async task is running.
    This prevents the user from thinking the script has frozen during long API waits.
    """
    start_time = time.time()
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    
    while not task.done():
        elapsed = time.time() - start_time
        # Use '\r' (carriage return) to overwrite the current line continuously
        sys.stdout.write(f"\r   {spinner_chars[i % len(spinner_chars)]} Processing '{filename}'... [{elapsed:.1f}s elapsed]")
        sys.stdout.flush()
        i += 1
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            break
            
    # Clear the spinner line completely once the task finishes
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


async def main():
    # 2. Setup Telemetry File Output
    # -------------------------------------------------------------------------
    telemetry_readable = PROJECT_ROOT / "telemetry_readable.log"
    telemetry_detailed = PROJECT_ROOT / "telemetry_detailed.json"
    setup_telemetry(str(telemetry_readable), str(telemetry_detailed))

    target_dir = PROJECT_ROOT / "target_ontologies"
    output_json_file = PROJECT_ROOT / "validation_results.json"
    
    if not target_dir.exists():
        print(f"[ERROR] Target directory not found: {target_dir}")
        print("Please create a folder named 'target_ontologies' in the project root and add .rdf files.")
        sys.exit(1)
 
    # 3. Load Environment Variables from the .env file
    # -------------------------------------------------------------------------
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY variable not found.")
        print("Please create a file named '.env' in the project root")
        print("and add the following line:\nGEMINI_API_KEY=your_key_here")
        sys.exit(1)
    try:
        requests_per_minute = float(os.environ.get("RATE_LIMIT_RPM", 15.0))
    except ValueError:
        requests_per_minute = 10.0

    # 4. Command Line Arguments Handling
    # -------------------------------------------------------------------------
    rdf_files = list(target_dir.glob("*.rdf"))
    
    if not rdf_files:
        print(f"[WARNING] No .rdf files found in directory: {target_dir}")
        sys.exit(0)

    print(f"🚀 Starting Batch Ontology Validation Pipeline...")
    print(f"📂 Found {len(rdf_files)} .rdf file(s) in '{target_dir.name}'")
    print(f"🚦 Rate Limit Configured: {requests_per_minute} requests/minute")
    print("-" * 60)

    # 5. Dependency Injection and Initialization
    # -------------------------------------------------------------------------
    base_llm = GeminiClient(api_key=api_key)
    
    retry_config = RetryPolicyConfig(
        max_retries=3, 
        delay_between_retries=1.0, 
        backoff_strategy=BackoffStrategy.EXPONENTIAL
    )
    resilient_llm = RetryableLLMClient(llm_client=base_llm, config=retry_config)

    prompts_path = SRC_DIR / "config" / "prompts.yaml" 
    if not prompts_path.exists():
        print(f"[WARNING] prompts.yaml file not found at {prompts_path.absolute()}")
        print("Ensure your prompts.yaml file is located inside the 'src/config' directory.")
        sys.exit(1)
        
    prompt_manager = PromptManager(file_path=str(prompts_path))
    tracer = trace.get_tracer(__name__)
    serializer = TurtleSerializer(tracer)
    consensus_resolver = ConsensusResolver()
    global_rate_limiter = AsyncLimiter(max_rate=requests_per_minute, time_period=60)

    auditor = LLMEntityAuditor(
        model=resilient_llm,
        prompt_manager=prompt_manager,
        serializer=serializer,
        consensus_resolver=consensus_resolver,
        rate_limiter=global_rate_limiter,
        user_input="Please, validate this healthcare ontology strictly.",
        suite_name="owl_validations",
        model_name="gemini-3.1-flash-lite"
    )

    orchestrator = EntityOrchestrator(strategy=auditor)
    pipeline = Pipeline(orchestrator=orchestrator)

    # 6. Pipeline Execution
    # -------------------------------------------------------------------------
    aggregated_results = {}
    
    global_cost = 0.0
    global_tokens = 0
    global_successful = 0
    global_failed = 0

    if not hasattr(Thing, "individual_id"):
        type.__setattr__(Thing, "individual_id", property(lambda self: self.name))
    
    if not hasattr(World.as_rdflib_graph, "_is_patched"):
        original_as_rdflib_graph = World.as_rdflib_graph
        def patched_as_rdflib_graph(self, *args, **kwargs):
            return original_as_rdflib_graph(self)
        patched_as_rdflib_graph._is_patched = True
        World.as_rdflib_graph = patched_as_rdflib_graph

    for rdf_file in rdf_files:
        print(f"\n⏳ Processing file: {rdf_file.name}...")
        try:
            pipeline_task = asyncio.create_task(pipeline.execute(str(rdf_file)))
            spinner_task = asyncio.create_task(show_spinner(pipeline_task, rdf_file.name))
            execution_summaries = await pipeline_task
            await spinner_task
            
            file_results_dump = [summary.model_dump(mode='json') for summary in execution_summaries]
            aggregated_results[rdf_file.name] = file_results_dump
            
            file_success = sum(1 for s in execution_summaries if s.is_successful())
            file_failed = len(execution_summaries) - file_success
            
            print(f"   ✅ Done! [{file_success} Passed | {file_failed} Failed]")
            
            global_successful += file_success
            global_failed += file_failed
            for summary in execution_summaries:
                global_cost += summary.total_metrics.cost
                global_tokens += summary.total_metrics.tokens_consumed
                
        except Exception as e:
            print(f"   [ERROR] Failed to process {rdf_file.name}: {str(e)}")
            aggregated_results[rdf_file.name] = {"error": str(e)}

    # 7. Write Final JSON Export
    # -------------------------------------------------------------------------
    try:
        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(aggregated_results, f, indent=4, ensure_ascii=False)
        print(f"\n💾 Successfully exported full analysis to: {output_json_file.name}")
    except Exception as e:
        print(f"\n[ERROR] Failed to write JSON output file: {str(e)}")

    # 8. Global Telemetry/Metrics Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total Files Processed    : {len(rdf_files)}")
    print(f"Total Entities Validated : {global_successful + global_failed}")
    print(f"Successful Validations   : {global_successful}")
    print(f"Failed Validations       : {global_failed}")
    print(f"Total Tokens Consumed    : {global_tokens:,}")
    print(f"Estimated Total Cost     : ${global_cost:.6f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Pipeline execution interrupted by the user.")
        sys.exit(0)