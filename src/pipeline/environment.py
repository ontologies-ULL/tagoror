"""
Environment and configuration validation.
Handles all pre-flight checks and environment setup.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from ui.ui_manager import UIManager


class EnvironmentManager:
    """
    Validates environment and configuration before pipeline execution.
    """

    def __init__(self, ui: UIManager):
        self.ui = ui

    def load_env(self) -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    def validate_api_key(self) -> str:
        """
        Validate that GEMINI_API_KEY is set.
        
        Returns:
            API key if valid
            
        Raises:
            SystemExit if API key not found
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            self.ui.error("GEMINI_API_KEY variable not found.")
            self.ui.info("Create a .env file with: GEMINI_API_KEY=your_key_here")
            sys.exit(1)
        return api_key

    def validate_target_dir(self, target_dir: Path) -> list[Path]:
        """
        Validate target directory and find RDF files.
        
        Returns:
            List of RDF files found
            
        Raises:
            SystemExit if directory invalid or no files found
        """
        if not target_dir.exists():
            self.ui.error(f"Target directory not found: {target_dir}")
            self.ui.info("Please specify a directory containing .rdf ontology files.")
            sys.exit(1)

        rdf_files = list(target_dir.glob("*.rdf"))
        
        if not rdf_files:
            self.ui.warn(f"No .rdf files found in: {target_dir}")
            sys.exit(0)

        return rdf_files

    def validate_prompts_file(self, prompts_path: Path) -> None:
        """
        Validate that prompts.yaml exists.
        
        Raises:
            SystemExit if file not found
        """
        if not prompts_path.exists():
            self.ui.error(f"prompts.yaml not found: {prompts_path}")
            sys.exit(1)

    def setup_owlready2(self) -> None:
        """
        Configure owlready2 library patches for compatibility.
        """
        from owlready2 import Thing, World
        
        if not hasattr(Thing, "individual_id"):
            type.__setattr__(Thing, "individual_id", property(lambda self: self.name))
        
        if not hasattr(World.as_rdflib_graph, "_is_patched"):
            original_as_rdflib_graph = World.as_rdflib_graph
            def patched_as_rdflib_graph(self, *args, **kwargs):
                return original_as_rdflib_graph(self)
            patched_as_rdflib_graph._is_patched = True
            World.as_rdflib_graph = patched_as_rdflib_graph
