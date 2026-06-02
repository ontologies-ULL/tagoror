import yaml

class PromptManager:
  _BASE_PROMPT_KEY = "base_generic"
  _BASE_PROMPT_KEY_LEGACY = "base_generica"
  _TASK_CHAINS_KEY = "task_chains"
  _TASK_CHAINS_KEY_LEGACY = ""

  def __init__(self, file_path: str):
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

    return "\n\n".join(base_sections.values())

  def get_task_chain(self, chain_name: str) -> list:
    """
    Return the task chain for the given chain name.
    """
    task_chain = self._get_task_chains().get(chain_name)
    if not task_chain:
      raise KeyError(f"Missing task chain: {chain_name}")

    return task_chain

  def _get_base_sections(self) -> dict:
    return self._prompts.get(
      self._BASE_PROMPT_KEY,
      self._prompts.get(self._BASE_PROMPT_KEY_LEGACY, {}),
    )

  def _get_task_chains(self) -> dict:
    return self._prompts.get(
      self._TASK_CHAINS_KEY,
      self._prompts.get(self._TASK_CHAINS_KEY_LEGACY, {}),
    )