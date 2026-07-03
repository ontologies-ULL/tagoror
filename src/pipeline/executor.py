"""
Pipeline execution and results aggregation.
Handles the main processing loop and statistics tracking.
"""

import asyncio
import json
from pathlib import Path
from dataclasses import dataclass
from core.pipeline.pipeline import Pipeline
from ui.status_reporter import StatusReporter
from ui.ui_manager import UIManager
from storage import ResultsManager


@dataclass
class ExecutionStats:
    """Tracks execution statistics."""
    total_files: int = 0
    total_entities: int = 0
    successful: int = 0
    failed: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0


class PipelineExecutor:
    """
    Executes the validation pipeline over a batch of RDF files.
    Handles the main processing loop and result aggregation.
    """

    def __init__(
        self,
        ui: UIManager,
        results_mgr: ResultsManager,
        pipeline: Pipeline,
        status_reporter: StatusReporter,
        model_name: str,
    ):
        self.ui = ui
        self.results_mgr = results_mgr
        self.pipeline = pipeline
        self.status_reporter = status_reporter
        self.model_name = model_name
        self.stats = ExecutionStats()

    async def process_files(self, rdf_files: list[Path]) -> dict:
        """
        Process all RDF files through the pipeline.
        
        Returns:
            Dictionary with aggregated results
        """
        self.stats.total_files = len(rdf_files)
        aggregated_results = {}

        for rdf_file in rdf_files:
            self.ui.info(f"\n⏳ Processing: {rdf_file.name}...")
            
            try:
                # Execute pipeline for this file
                pipeline_task = asyncio.create_task(self.pipeline.execute(str(rdf_file)))
                spinner_task = self.ui.start_spinner(
                    pipeline_task, rdf_file.name, self.status_reporter
                )
                execution_summaries = await pipeline_task
                await self.ui.stop_spinner()
                
                # Process results
                file_results_dump = [
                    summary.model_dump(mode='json') for summary in execution_summaries
                ]
                aggregated_results[rdf_file.name] = file_results_dump
                
                # Save individual result
                self._save_individual_result(rdf_file.name, file_results_dump)
                
                # Update statistics
                self._update_stats(execution_summaries)
                
                # Print file summary
                file_success = sum(1 for s in execution_summaries if s.is_successful())
                file_failed = len(execution_summaries) - file_success
                self.ui.info(f"   ✅ [{file_success} Passed | {file_failed} Failed]")
                
            except Exception as e:
                self.ui.error(f"   Failed: {str(e)}")
                aggregated_results[rdf_file.name] = {"error": str(e)}

        return aggregated_results

    def _save_individual_result(self, filename: str, results: list) -> None:
        """Save individual ontology result to file."""
        try:
            saved_path = self.results_mgr.write(filename, self.model_name, results)
            self.ui.info(f"   💾 Result: {saved_path}")
        except Exception as e:
            self.ui.warn(f"Could not write result: {e}")

    def _update_stats(self, execution_summaries: list) -> None:
        """Update execution statistics."""
        for summary in execution_summaries:
            self.stats.total_entities += 1
            if summary.is_successful():
                self.stats.successful += 1
            else:
                self.stats.failed += 1
            
            self.stats.total_cost += summary.total_metrics.cost
            self.stats.total_tokens += summary.total_metrics.tokens_consumed

    def print_summary(self) -> None:
        """Print execution summary."""
        self.ui.info("\n" + "=" * 70)
        self.ui.info("📊 BATCH PROCESSING SUMMARY")
        self.ui.info("=" * 70)
        self.ui.info(f"Total Files Processed    : {self.stats.total_files}")
        self.ui.info(f"Total Entities Validated : {self.stats.total_entities}")
        self.ui.info(f"Successful Validations   : {self.stats.successful}")
        self.ui.info(f"Failed Validations       : {self.stats.failed}")
        self.ui.info(f"Total Tokens Consumed    : {self.stats.total_tokens:,}")
        self.ui.info(f"Estimated Total Cost     : ${self.stats.total_cost:.6f}")
        self.ui.info("=" * 70 + "\n")


class ResultsWriter:
    """
    Handles writing aggregated and individual results to files.
    """

    def __init__(self, ui: UIManager, project_root: Path):
        self.ui = ui
        self.project_root = project_root

    def write_aggregated_results(self, results: dict) -> None:
        """Write aggregated results to a single JSON file."""
        output_file = self.project_root / "validation_results.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            self.ui.info(f"💾 Aggregated results: {output_file.name}")
        except Exception as e:
            self.ui.error(f"Failed to write results: {str(e)}")
