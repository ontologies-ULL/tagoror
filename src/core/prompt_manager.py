import yaml
from pathlib import Path

class PromptManager:
    _BASE_PROMPT_KEY = "base_generic"
    _TASK_SUITES_KEY = "evaluation_suites"
    _DEFAULT_PROMPT_FILE: Path = Path(__file__).parent.parent / "prompts.yaml" 
  
    def __init__(self, file_path: str = None):
        if file_path is None:
            file_path = str(self._DEFAULT_PROMPT_FILE)
        self._prompts = self._load_from_disk(file_path)

    def _load_from_disk(self, file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def get_assembled_system_prompt(self) -> str:
        """
        Assemble the base system prompt by joining all base sections.
        """
        base_sections = self._get_base_sections() 
        if not base_sections:
            raise KeyError("Missing base prompt sections in prompts file.")
        prompt_parts = []

        for section_name, content in base_sections.items():
            header_title = section_name.replace("_", " ").upper()
            formatted_section = f"### {header_title}\n{content.strip()}"
            prompt_parts.append(formatted_section)

        return "\n\n".join(prompt_parts)
 
    def get_evaluation_suite(self, suite_name: str = "owl_validations") -> dict:
        """
        Retrieve the evaluation suite by name from the prompts file.
        """
        suites_block = self._prompts.get(self._TASK_SUITES_KEY) or {}
        suite = suites_block.get(suite_name)
        if not suite:
            raise KeyError(f"Missing evaluation suite: {suite_name}")
        return suite
    
    def _get_base_sections(self) -> dict:
        """
        Retrieve the base prompt sections from the prompts file.
        """
        base_block = self._prompts.get(self._BASE_PROMPT_KEY) or {}
        if not base_block:
            raise KeyError("Missing base prompt block in prompts file.")
        return base_block