import yaml
from pathlib import Path

class PromptManager:
    _BASE_PROMPT_KEY = "base_generic"
    _TASK_SUITES_KEY = "evaluation_suites"
    _DEFAULT_PROMPT_FILE: Path = Path(__file__).parent.parent / "prompts.yaml" 
  
    def __init__(self, file_path: str = None):
        if file_path is None:
            file_path = str(self._DEFAULT_PROMPT_FILE)
        self._raw_config = self._load_from_disk(file_path)

    def _load_from_disk(self, file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def build_task_prompt(self, task_config: dict, context_data: dict, default_self_criticism: bool = False) -> str:
        """
        Constructs the user prompt for a specific task, incorporating context data and optional few-shot examples.
        """
        raw_prompt = task_config.get("prompt", "")
        if not raw_prompt:
            return None

        class SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"
        formatted_prompt = raw_prompt.format_map(SafeDict(context_data))

        examples = task_config.get("examples", [])
        few_shot_block = ""
        if examples:
            few_shot_block = "\n## Examples\n\n"
            for i, ex in enumerate(examples, 1):
                few_shot_block += f"### Example {i}\n"
                few_shot_block += f"**Input:**\n```json\n{ex.get('input', '').strip()}\n```\n\n"
                few_shot_block += f"**Output:**\n```json\n{ex.get('output', '').strip()}\n```\n\n"
                
        use_cot = task_config.get("enable_self_criticism", default_self_criticism)
        cot_instruction = ""
        if use_cot:
            cot_instruction = (
                "\n\n## Self-Criticism Required\n"
                "Before providing the final `status`, you **MUST** provide a detailed "
                "`reasoning` field in your JSON explaining your thought process step-by-step."
            )
        return few_shot_block + formatted_prompt + cot_instruction
    
    def get_suite_config(self, suite_name: str) -> dict:
        """
        Extracts the configuration block. 
        Supports both 'defaults' and 'configurations' keys to prevent mapping errors.
        """
        suite_data = self._raw_config.get("evaluation_suites", {}).get(suite_name, {})
        return suite_data.get("defaults", suite_data.get("configurations", {}))

    def get_suite_tasks(self, suite_name: str = "owl_validations") -> dict:
        """
        Extracts strictly the tasks dictionary.
        This proves the system DOES NOT ignore the 'tasks' YAML nesting.
        """
        suite_data = self._raw_config.get("evaluation_suites", {}).get(suite_name, {})
        tasks = suite_data.get("tasks", {})
        
        if not tasks:
            raise ValueError(f"CRITICAL: No 'tasks' key found under suite '{suite_name}'. Check YAML indentation.")
        return tasks 

    def get_assembled_system_prompt(self, base_ontology_text: str, with_base_ontology: bool = True) -> str:
        base_generic = self._raw_config.get("base_generic", {})
        role = base_generic.get("role", "")
        constraints = base_generic.get("constraints", "")
        
        class SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"
        formatted_role = role
        if with_base_ontology:
            formatted_role = role.format_map(SafeDict({"base_ontology": base_ontology_text}))
        return f"{formatted_role}\n\n## Constraints\n{constraints}"
 
    def get_evaluation_suite(self, suite_name: str = "owl_validations") -> dict:
        """
        Retrieve the evaluation suite by name from the prompts file.
        """
        suites_block = self._raw_config.get(self._TASK_SUITES_KEY) or {}
        suite = suites_block.get(suite_name)
        if not suite:
            raise KeyError(f"Missing evaluation suite: {suite_name}")
        return suite

    def _get_base_sections(self) -> dict:
        """
        Retrieve the base prompt sections from the prompts file.
        """
        base_block = self._raw_config.get(self._BASE_PROMPT_KEY) or {}
        if not base_block:
            raise KeyError("Missing base prompt block in prompts file.")
        return base_block