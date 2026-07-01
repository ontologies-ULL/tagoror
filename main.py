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
from opentelemetry.sdk.trace import TracerProvider
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


def setup_telemetry(log_file_path: str):
    """
    Configures OpenTelemetry to export all traces and spans to a local file
    instead of the standard output or an external server.
    """
    provider = TracerProvider()
    
    trace_file = open(log_file_path, "a", encoding="utf-8")
    file_exporter = ConsoleSpanExporter(out=trace_file)
    processor = SimpleSpanProcessor(file_exporter)
    provider.add_span_processor(processor)
    
    trace.set_tracer_provider(provider)


async def main():
    # 2. Setup Telemetry File Output
    # -------------------------------------------------------------------------
    telemetry_file = PROJECT_ROOT / "telemetry_traces.json"
    setup_telemetry(str(telemetry_file))
    
    # 3. Load Environment Variables from the .env file
    # -------------------------------------------------------------------------
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY variable not found.")
        print("Please create a file named '.env' in the project root")
        print("and add the following line:\nGEMINI_API_KEY=your_key_here")
        sys.exit(1)

    # 4. Command Line Arguments Handling
    # -------------------------------------------------------------------------
    if len(sys.argv) > 1:
        ontology_path = sys.argv[1]
    else:
        ontology_path = "tests/ontologies/osdi_CU1_P1_S1_M1.rdf"
        
    if not Path(ontology_path).exists():
        print(f"[ERROR] Ontology file not found at: {ontology_path}")
        sys.exit(1)

    print(f"🚀 Starting Ontology Validation Pipeline...")
    print(f"📂 Target File: {ontology_path}")
    print(f"📡 Telemetry traces will be saved to: {telemetry_file.name}")
    print("-" * 60)

    # 5. Dependency Injection and Initialization
    # -------------------------------------------------------------------------
    base_llm = GeminiClient(api_key=api_key)
    
    retry_config = RetryPolicyConfig(
        max_retries=3, 
        delay_between_retries=1.0, 
        backoff_strategy=BackoffStrategy.EXPONENTIAL
    )
    resilient_llm = RetryableLLMClient(client=base_llm, config=retry_config)

    prompts_path = SRC_DIR / "prompts.yaml" 
    if not prompts_path.exists():
        print(f"[WARNING] prompts.yaml file not found at {prompts_path.absolute()}")
        print("Ensure your prompts.yaml file is located inside the 'src' directory.")
        sys.exit(1)
        
    prompt_manager = PromptManager(file_path=str(prompts_path))
    serializer = TurtleSerializer()
    consensus_resolver = ConsensusResolver()

    auditor = LLMEntityAuditor(
        model=resilient_llm,
        prompt_manager=prompt_manager,
        serializer=serializer,
        consensus_resolver=consensus_resolver,
        user_input="Please, validate this healthcare ontology strictly.",
        suite_name="owl_validations",
        model_name="gemini-1.5-pro"
    )

    orchestrator = EntityOrchestrator(strategy=auditor)
    pipeline = Pipeline(orchestrator=orchestrator)

    # 6. Pipeline Execution
    # -------------------------------------------------------------------------
    try:
        print("⏳ Processing entities concurrently. Please wait...")
        execution_summaries = await pipeline.execute(ontology_path)
    except Exception as e:
        print(f"\n[FATAL ERROR] The pipeline collapsed unexpectedly: {str(e)}")
        sys.exit(1)

    # 7. Output Formatting and Results Display
    # -------------------------------------------------------------------------
    print("\n✅ Pipeline Execution Completed!\n")
    print("=" * 60)
    print(" " * 20 + "EXECUTION RESULTS" + " " * 20)
    print("=" * 60)
    
    total_cost = 0.0
    total_tokens = 0
    successful_entities = 0
    failed_entities = 0

    for summary in execution_summaries:
        status_icon = "🟢" if summary.is_successful() else "🔴"
        
        print(f"\n{status_icon} Entity: {summary.individual_id}")
        print(f"   System Message: {summary.system_summary}")
        
        for result in summary.results:
            task_icon = "✔️" if result.status.value == "success" else "❌"
            print(f"   {task_icon} Task [{result.task_id}]: {result.status.value.upper()}")
            
            if result.findings and result.status.value != "success":
                for finding in result.findings:
                    print(f"       - {finding}")

        total_cost += summary.total_metrics.cost
        total_tokens += summary.total_metrics.tokens_consumed
        
        if summary.is_successful():
            successful_entities += 1
        else:
            failed_entities += 1

    # 8. Global Telemetry/Metrics Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 GLOBAL METRICS SUMMARY")
    print("=" * 60)
    print(f"Total Entities Evaluated : {len(execution_summaries)}")
    print(f"Successful Validations   : {successful_entities}")
    print(f"Failed Validations       : {failed_entities}")
    print(f"Total Tokens Consumed    : {total_tokens:,}")
    print(f"Estimated Total Cost     : ${total_cost:.6f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Pipeline execution interrupted by the user.")
        sys.exit(0)