"""
Main entry point for the Ontology Validation application.

Provides interactive and CLI modes for ontology validation with
hallucination detection using multiple LLM models.

Usage:
    uv run main.py              # Interactive mode (recommended)
    uv run main.py --help       # Show help
"""

import asyncio
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    print("[ERROR] Missing 'python-dotenv' library.")
    sys.exit(1)

from storage import TelemetryManager, ResultsManager
from ui.ui_manager import UIManager
from cli.interactive_cli import InteractiveCLI
from pipeline import (
    PipelineConfigurator,
    PipelineExecutor,
    ResultsWriter,
    EnvironmentManager,
)


async def execute_pipeline(
    target_dir: Path,
    rate_limit: float,
    model_name: str,
    prompts_path: Path,
    user_input: str,
    verbose: bool = True,
) -> None:
    """
    Execute the ontology validation pipeline with given configuration.
    
    Delegates to specialized components for environment validation,
    pipeline configuration, and execution.
    """
    # Initialize managers
    telemetry_mgr = TelemetryManager(PROJECT_ROOT)
    telemetry_mgr.setup()
    
    ui = UIManager(verbose=verbose)
    env_mgr = EnvironmentManager(ui)
    results_mgr = ResultsManager(PROJECT_ROOT)
    results_writer = ResultsWriter(ui, PROJECT_ROOT)
    
    try:
        # Validate environment
        env_mgr.load_env()
        api_key = env_mgr.validate_api_key()
        env_mgr.validate_target_dir(target_dir)
        env_mgr.validate_prompts_file(prompts_path)
        env_mgr.setup_owlready2()
        
        rdf_files = env_mgr.validate_target_dir(target_dir)

        # Show startup info
        ui.info("🚀 Starting Batch Ontology Validation Pipeline...")
        ui.info(f"📂 Found {len(rdf_files)} .rdf file(s)")
        ui.info(f"🚦 Rate Limit: {rate_limit} requests/minute")
        ui.info(f"🤖 Model: {model_name}")
        ui.info(f"💭 Custom Input: {user_input[:50]}...")
        ui.info("-" * 70)

        # Configure pipeline
        configurator = PipelineConfigurator(api_key)
        llm = configurator.build_llm()
        pipeline, status_reporter = configurator.build_pipeline(
            llm_client=llm,
            prompts_path=prompts_path,
            rate_limit=rate_limit,
            model_name=model_name,
            user_input=user_input,
        )

        # Execute pipeline
        executor = PipelineExecutor(
            ui=ui,
            results_mgr=results_mgr,
            pipeline=pipeline,
            status_reporter=status_reporter,
            model_name=model_name,
        )
        
        aggregated_results = await executor.process_files(rdf_files)
        
        # Write results and show summary
        results_writer.write_aggregated_results(aggregated_results)
        executor.print_summary()
        ui.show_completion_message(PROJECT_ROOT / "results")
        
    finally:
        try:
            telemetry_mgr.close()
        except Exception:
            pass



def main():
    """
    Main entry point with CLI argument parsing and mode selection.
    Delegates to interactive or CLI execution path.
    """
    parser = argparse.ArgumentParser(
        description="Ontology Validation Pipeline with Hallucination Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py                              # Interactive mode
  uv run main.py --user-input "Custom prompt" # CLI mode
  uv run main.py --help                       # Show this help
        """
    )
    
    parser.add_argument("--target-dir", dest="target_dir", help="Directory with .rdf files")
    parser.add_argument("--rate-limit", dest="rate_limit", type=float, help="Requests/min")
    parser.add_argument("--model", dest="model_name", help="LLM model name")
    parser.add_argument("--prompts", dest="prompts_path", help="Path to prompts.yaml")
    parser.add_argument("--user-input", dest="user_input", help="Custom hallucination prompt")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--interactive", action="store_true", help="Force interactive mode")

    args = parser.parse_args()
    
    # Determine execution mode
    use_interactive = args.interactive or (
        not args.target_dir and not args.rate_limit and 
        not args.model_name and not args.user_input
    )
    
    try:
        if use_interactive:
            _run_interactive_mode(args.quiet)
        else:
            _run_cli_mode(args)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


def _run_interactive_mode(quiet: bool) -> None:
    """Run the interactive setup and execution flow."""
    cli = InteractiveCLI(PROJECT_ROOT)
    config = cli.run_interactive_setup()
    
    asyncio.run(
        execute_pipeline(
            target_dir=config["target_dir"],
            rate_limit=config["rate_limit"],
            model_name=config["model"],
            prompts_path=SRC_DIR / "config" / "prompts.yaml",
            user_input=config["user_input"],
            verbose=not quiet,
        )
    )


def _run_cli_mode(args: argparse.Namespace) -> None:
    """Run the CLI mode with provided arguments."""
    target_dir = Path(args.target_dir) if args.target_dir else (PROJECT_ROOT / "target_ontologies")
    rate_limit = args.rate_limit or 15.0
    model_name = args.model_name or "gemini-2.0-flash-lite"
    prompts_path = Path(args.prompts_path) if args.prompts_path else (SRC_DIR / "config" / "prompts.yaml")
    user_input = args.user_input or "Please, validate this healthcare ontology strictly."
    
    asyncio.run(
        execute_pipeline(
            target_dir=target_dir,
            rate_limit=rate_limit,
            model_name=model_name,
            prompts_path=prompts_path,
            user_input=user_input,
            verbose=not args.quiet,
        )
    )



if __name__ == "__main__":
    main()