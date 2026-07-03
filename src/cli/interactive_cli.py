"""
Interactive CLI for Ontology Validation Pipeline.

Provides a user-friendly interface to configure and run the validation pipeline,
including setting custom prompts and hallucination parameters.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


class InteractiveCLI:
    """
    Manages interactive CLI experience for pipeline configuration and execution.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.config_dir = self.src_dir / "config"
        self.prompts_file = self.config_dir / "prompts.yaml"

    def _print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70 + "\n")

    def _print_section(self, title: str):
        """Print a section header."""
        print(f"\n📋 {title}")
        print("-" * 70)

    def _prompt_with_default(self, message: str, default: str | None = None) -> str:
        """
        Prompt user with optional default value.
        
        If default is provided, show it in brackets. Press Enter to use default.
        """
        if default:
            prompt_text = f"{message} [{default}]: "
        else:
            prompt_text = f"{message}: "
        
        user_input = input(prompt_text).strip()
        return user_input if user_input else (default or "")

    def _prompt_yes_no(self, message: str, default: bool = True) -> bool:
        """Prompt for yes/no confirmation."""
        default_str = "Y/n" if default else "y/N"
        response = input(f"{message} [{default_str}]: ").strip().lower()
        
        if not response:
            return default
        return response in ("y", "yes")

    def _prompt_choice(self, message: str, options: list[str], default_idx: int = 0) -> str:
        """
        Prompt user to choose from a list of options.
        """
        self._print_section(message)
        for idx, option in enumerate(options):
            marker = "→" if idx == default_idx else " "
            print(f"  {marker} ({idx + 1}) {option}")
        
        while True:
            choice = input(f"\nSelect option (default: {default_idx + 1}): ").strip()
            if not choice:
                return options[default_idx]
            
            try:
                chosen_idx = int(choice) - 1
                if 0 <= chosen_idx < len(options):
                    return options[chosen_idx]
                print(f"  ❌ Invalid selection. Please choose 1-{len(options)}")
            except ValueError:
                print(f"  ❌ Invalid input. Please enter a number.")

    def _prompt_path(self, message: str, must_exist: bool = True) -> Path:
        """
        Prompt for a file/directory path.
        """
        while True:
            path_str = self._prompt_with_default(message)
            path = Path(path_str).expanduser()
            
            if must_exist and not path.exists():
                print(f"  ❌ Path does not exist: {path}")
                continue
            
            return path

    def _prompt_float(self, message: str, default: float = 15.0, min_val: float = 1.0) -> float:
        """
        Prompt for a floating-point number.
        """
        while True:
            try:
                value_str = self._prompt_with_default(message, str(default))
                value = float(value_str)
                
                if value < min_val:
                    print(f"  ❌ Value must be >= {min_val}")
                    continue
                
                return value
            except ValueError:
                print(f"  ❌ Invalid number. Please enter a valid float.")

    def _show_available_models(self) -> list[str]:
        """
        List available LLM models.
        """
        return [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gpt-4-turbo",
            "gpt-4o",
        ]

    def _edit_prompts_file(self):
        """
        Open prompts.yaml in the user's default editor.
        """
        if not self.prompts_file.exists():
            print(f"  ⚠️  Prompts file not found: {self.prompts_file}")
            return
        
        editor = os.environ.get("EDITOR", "nano")
        try:
            subprocess.run([editor, str(self.prompts_file)], check=True)
            print(f"  ✅ Prompts file updated: {self.prompts_file}")
        except Exception as e:
            print(f"  ❌ Failed to open editor: {e}")

    def configure_target_ontology(self) -> Path:
        """
        Interactive configuration for target ontology directory.
        """
        self._print_section("Target Ontology Directory")
        
        default_target = self.project_root / "target_ontologies"
        target_dir = self._prompt_path(
            f"Enter path to ontology directory (default: {default_target})",
            must_exist=True
        )
        
        if not target_dir.exists():
            print(f"  ❌ Directory does not exist: {target_dir}")
            return default_target
        
        rdf_files = list(target_dir.glob("*.rdf"))
        print(f"  ℹ️  Found {len(rdf_files)} .rdf file(s)")
        
        return target_dir

    def configure_rate_limit(self) -> float:
        """
        Interactive configuration for API rate limit.
        """
        self._print_section("API Rate Limit Configuration")
        
        try:
            env_rate = float(os.environ.get("RATE_LIMIT_RPM", 15.0))
        except ValueError:
            env_rate = 15.0
        
        return self._prompt_float(
            "Enter requests per minute (RPM)",
            default=env_rate,
            min_val=1.0
        )

    def configure_model(self) -> str:
        """
        Interactive configuration for LLM model selection.
        """
        self._print_section("LLM Model Selection")
        
        available_models = self._show_available_models()
        default_model = "gemini-2.0-flash-lite"
        
        model = self._prompt_choice(
            "Choose an LLM model",
            available_models,
            default_idx=available_models.index(default_model) if default_model in available_models else 0
        )
        
        return model

    def configure_user_input(self) -> str:
        """
        Interactive configuration for user input (contextual hallucination prompt).
        """
        self._print_section("Custom User Input for Contextual Hallucination Detection")
        
        default_input = "Please, validate this healthcare ontology strictly."
        
        print("This input will be used to evaluate if the ontology response")
        print("meets the original specifications (Contextual Hallucination).\n")
        
        user_input = self._prompt_with_default(
            "Enter your custom prompt",
            default=default_input
        )
        
        return user_input

    def configure_prompts(self) -> bool:
        """
        Ask if user wants to edit prompts.yaml.
        
        Returns True if edited, False otherwise.
        """
        self._print_section("Edit Prompts Configuration")
        
        if self._prompt_yes_no(
            "Do you want to edit the prompts configuration file?",
            default=False
        ):
            self._edit_prompts_file()
            return True
        
        return False

    def show_configuration_summary(self, config: dict):
        """
        Show a summary of configuration before execution.
        """
        self._print_header("Configuration Summary")
        
        print(f"  Target Directory:      {config.get('target_dir', 'N/A')}")
        print(f"  Rate Limit (RPM):      {config.get('rate_limit', 'N/A')}")
        print(f"  LLM Model:             {config.get('model', 'N/A')}")
        print(f"  Custom User Input:     {config.get('user_input', 'N/A')[:50]}...")
        
        print()

    def run_interactive_setup(self) -> dict:
        """
        Run the complete interactive setup flow.
        
        Returns dict with all configuration.
        """
        self._print_header("🚀 Ontology Validation Pipeline - Interactive Setup")
        
        print("Welcome! This CLI will help you configure and run the pipeline.")
        print("Press Ctrl+C at any time to cancel.\n")
        
        # Step 1: Target directory
        target_dir = self.configure_target_ontology()
        
        # Step 2: Rate limit
        rate_limit = self.configure_rate_limit()
        
        # Step 3: Model selection
        model = self.configure_model()
        
        # Step 4: User input
        user_input = self.configure_user_input()
        
        # Step 5: Optional prompt editing
        self.configure_prompts()
        
        # Step 6: Show summary and confirm
        config = {
            "target_dir": target_dir,
            "rate_limit": rate_limit,
            "model": model,
            "user_input": user_input,
        }
        
        self.show_configuration_summary(config)
        
        if self._prompt_yes_no("Proceed with pipeline execution?", default=True):
            return config
        else:
            print("\n❌ Pipeline execution cancelled by user.")
            sys.exit(0)

    def show_completion_message(self, results_dir: Path):
        """
        Show completion message with results location.
        """
        self._print_header("✅ Pipeline Execution Completed")
        
        print(f"Results saved to: {results_dir}")
        print("\nNext steps:")
        print(f"  1. Review results in: {results_dir}")
        print("  2. Check logs in: ./logs/")
        print("  3. Inspect telemetry: ./logs/telemetry_*.log")
        print()

