"""
Unit tests for PromptManager
==============================
Located at: src/core.

YAML structure (from the real prompts file):

  base_generic:
    role: |
      You are an expert ontology validator...
    constraints: |
      Do not invent information...
  task_chains:
    basic_validation:
      - id: check_class
        prompt: |
          Verify whether the entity is a valid OWL class.
      - id: check_properties
        prompt: |
          List relevant properties and check their consistency.
    extended_validation:
      - id: detect_inconsistencies
        prompt: |
          Point out logical inconsistencies if they exist.
      - id: suggest_improvements
        prompt: |
          Suggest OWL modeling improvements.

Covers:
  - _load_from_disk: reads a valid YAML file correctly
  - _load_from_disk: returns empty dict for an empty file
  - _load_from_disk: raises FileNotFoundError for a missing file
  - _load_from_disk: loaded content matches YAML exactly
  - get_assembled_system_prompt: joins role + constraints with double newline
  - get_assembled_system_prompt: role content present in result
  - get_assembled_system_prompt: constraints content present in result
  - get_assembled_system_prompt: single section returns value without separator
  - get_assembled_system_prompt: raises KeyError when base_generic missing
  - get_assembled_system_prompt: raises KeyError for empty YAML
  - get_assembled_system_prompt: falls back to 'base_generica' (legacy key)
  - get_assembled_system_prompt: returns a string
  - get_task_chain: returns list for known chain
  - get_task_chain: correct number of tasks per chain
  - get_task_chain: each task has 'id' and 'prompt' keys
  - get_task_chain: task ids match YAML content exactly
  - get_task_chain: task prompts match YAML content exactly
  - get_task_chain: different chains return different tasks
  - get_task_chain: raises KeyError for unknown chain name
  - get_task_chain: raises KeyError when task_chains section is absent
  - _get_base_sections: returns base_generic dict when present
  - _get_base_sections: falls back to base_generica when base_generic absent
  - _get_base_sections: returns empty dict when neither key exists
  - _get_task_chains: returns task_chains dict when present
  - _get_task_chains: returns empty dict when task_chains absent

Testing strategy:
  _load_from_disk opens a real file, so we use tmp_path (pytest built-in)
  to create temporary YAML files on disk. This validates real YAML parsing
  without mocking the filesystem, and does not depend on any file in
  src/core. being present at test time.
"""

import pytest
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# YAML fixtures matching the real file structure
# ---------------------------------------------------------------------------

REAL_YAML_CONTENT = {
    "base_generic": {
        "role": (
            "You are an expert ontology validator and health economist "
            "specializing in Health Technology Assessment (HTA) models."
        ),
        "constraints": (
            "Do not invent information. If context is missing, say so."
        ),
    },
    "task_chains": {
        "basic_validation": [
            {"id": "check_class",       "prompt": "Verify whether the entity is a valid OWL class."},
            {"id": "check_properties",  "prompt": "List relevant properties and check their consistency."},
        ],
        "extended_validation": [
            {"id": "detect_inconsistencies", "prompt": "Point out logical inconsistencies if they exist."},
            {"id": "suggest_improvements",   "prompt": "Suggest OWL modeling improvements."},
        ],
    },
}


def write_yaml(tmp_path: Path, content: dict, filename: str = "prompts.yaml") -> Path:
    """Write a dict as YAML to a temp file and return its path as string."""
    path = tmp_path / filename
    path.write_text(yaml.dump(content, allow_unicode=True), encoding="utf-8")
    return str(path)


def write_raw(tmp_path: Path, content: str, filename: str = "prompts.yaml") -> str:
    """Write a raw string to a temp file and return its path as string."""
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Tests: _load_from_disk
# ---------------------------------------------------------------------------

class TestLoadFromDisk:

    def test_reads_valid_yaml_file(self, tmp_path):
        """Must return a non-empty dict when the file is valid YAML."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        pm = PromptManager(path)
        assert isinstance(pm._prompts, dict)
        assert len(pm._prompts) > 0

    def test_loaded_content_contains_base_generic(self, tmp_path):
        """Loaded dict must contain the 'base_generic' key."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        pm = PromptManager(path)
        assert "base_generic" in pm._prompts

    def test_loaded_role_matches_yaml(self, tmp_path):
        """The role value must match exactly what was written to the YAML."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        pm = PromptManager(path)
        assert pm._prompts["base_generic"]["role"] == REAL_YAML_CONTENT["base_generic"]["role"]

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        """An empty YAML file must produce an empty dict, not None."""
        path = write_raw(tmp_path, "")
        from core.prompt_manager import PromptManager
        pm = PromptManager(path)
        assert pm._prompts == {}

    def test_raises_file_not_found_for_missing_file(self, tmp_path):
        """Must raise FileNotFoundError when the file path does not exist."""
        from core.prompt_manager import PromptManager
        with pytest.raises(FileNotFoundError):
            PromptManager(str(tmp_path / "nonexistent.yaml"))


# ---------------------------------------------------------------------------
# Tests: get_assembled_system_prompt
# ---------------------------------------------------------------------------

class TestGetAssembledSystemPrompt:

    def test_role_present_in_result(self, tmp_path):
        """The role section content must appear in the assembled prompt."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        result = PromptManager(path).get_assembled_system_prompt()
        assert REAL_YAML_CONTENT["base_generic"]["role"] in result

    def test_constraints_present_in_result(self, tmp_path):
        """The constraints section content must appear in the assembled prompt."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        result = PromptManager(path).get_assembled_system_prompt()
        assert REAL_YAML_CONTENT["base_generic"]["constraints"] in result

    def test_sections_joined_with_double_newline(self, tmp_path):
        """Role and constraints must be joined with '\\n\\n' as separator."""
        content = {
            "base_generic": {
                "role":        "ROLE_CONTENT",
                "constraints": "CONSTRAINTS_CONTENT",
            }
        }
        path = write_yaml(tmp_path, content)
        from core.prompt_manager import PromptManager
        result = PromptManager(path).get_assembled_system_prompt()
        assert result == "ROLE_CONTENT\n\nCONSTRAINTS_CONTENT"

    def test_single_section_no_separator(self, tmp_path):
        """A single section must return its value with no trailing separator."""
        content = {"base_generic": {"role": "Only role here"}}
        path = write_yaml(tmp_path, content)
        from core.prompt_manager import PromptManager
        result = PromptManager(path).get_assembled_system_prompt()
        assert result == "Only role here"

    def test_returns_string(self, tmp_path):
        """Return value must always be a plain string."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        result = PromptManager(path).get_assembled_system_prompt()
        assert isinstance(result, str)

    def test_raises_key_error_when_base_generic_missing(self, tmp_path):
        """Must raise KeyError with a descriptive message when base_generic is absent."""
        content = {"task_chains": {}}
        path = write_yaml(tmp_path, content)
        from core.prompt_manager import PromptManager
        with pytest.raises(KeyError, match="Missing base prompt sections"):
            PromptManager(path).get_assembled_system_prompt()

    def test_raises_key_error_for_empty_yaml(self, tmp_path):
        """Must raise KeyError when the YAML file is completely empty."""
        path = write_raw(tmp_path, "")
        from core.prompt_manager import PromptManager
        with pytest.raises(KeyError):
            PromptManager(path).get_assembled_system_prompt()

# ---------------------------------------------------------------------------
# Tests: get_task_chain
# ---------------------------------------------------------------------------

class TestGetTaskChain:

    def test_returns_list_for_basic_validation(self, tmp_path):
        """get_task_chain('basic_validation') must return a list."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        chain = PromptManager(path).get_task_chain("basic_validation")
        assert isinstance(chain, list)

    def test_basic_validation_has_two_tasks(self, tmp_path):
        """basic_validation must contain exactly 2 tasks."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        chain = PromptManager(path).get_task_chain("basic_validation")
        assert len(chain) == 2

    def test_extended_validation_has_two_tasks(self, tmp_path):
        """extended_validation must contain exactly 2 tasks."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        chain = PromptManager(path).get_task_chain("extended_validation")
        assert len(chain) == 2

    def test_each_task_has_id_key(self, tmp_path):
        """Every task dict must have an 'id' key."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        chain = PromptManager(path).get_task_chain("basic_validation")
        for task in chain:
            assert "id" in task

    def test_each_task_has_prompt_key(self, tmp_path):
        """Every task dict must have a 'prompt' key."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        chain = PromptManager(path).get_task_chain("basic_validation")
        for task in chain:
            assert "prompt" in task

    def test_basic_validation_task_ids_match_yaml(self, tmp_path):
        """basic_validation task ids must match exactly: check_class, check_properties."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        chain = PromptManager(path).get_task_chain("basic_validation")
        assert [t["id"] for t in chain] == ["check_class", "check_properties"]

    def test_extended_validation_task_ids_match_yaml(self, tmp_path):
        """extended_validation task ids must match: detect_inconsistencies, suggest_improvements."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        chain = PromptManager(path).get_task_chain("extended_validation")
        assert [t["id"] for t in chain] == ["detect_inconsistencies", "suggest_improvements"]

    def test_first_task_prompt_matches_yaml(self, tmp_path):
        """The first task prompt must match the YAML value exactly."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        chain = PromptManager(path).get_task_chain("basic_validation")
        assert chain[0]["prompt"] == "Verify whether the entity is a valid OWL class."

    def test_chains_are_independent(self, tmp_path):
        """basic_validation and extended_validation must return different task lists."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        pm     = PromptManager(path)
        basic  = [t["id"] for t in pm.get_task_chain("basic_validation")]
        extended = [t["id"] for t in pm.get_task_chain("extended_validation")]
        assert basic != extended

    def test_raises_key_error_for_unknown_chain(self, tmp_path):
        """Must raise KeyError containing the unknown chain name."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        with pytest.raises(KeyError, match="nonexistent_chain"):
            PromptManager(path).get_task_chain("nonexistent_chain")

    def test_raises_key_error_when_task_chains_section_missing(self, tmp_path):
        """Must raise KeyError when the task_chains section is absent entirely."""
        content = {"base_generic": {"role": "Only base, no chains."}}
        path = write_yaml(tmp_path, content)
        from core.prompt_manager import PromptManager
        with pytest.raises(KeyError):
            PromptManager(path).get_task_chain("basic_validation")


# ---------------------------------------------------------------------------
# Tests: _get_base_sections
# ---------------------------------------------------------------------------

class TestGetBaseSections:

    def test_returns_base_generic_dict(self, tmp_path):
        """Must return the full base_generic dict when the key is present."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        sections = PromptManager(path)._get_base_sections()
        assert sections == REAL_YAML_CONTENT["base_generic"]

    def test_returned_dict_has_role_and_constraints(self, tmp_path):
        """The returned dict must have both 'role' and 'constraints' keys."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        sections = PromptManager(path)._get_base_sections()
        assert "role"        in sections
        assert "constraints" in sections

    def test_returns_empty_dict_when_neither_key_exists(self, tmp_path):
        """Must return an empty dict when neither base_generic nor base_generica exist."""
        path = write_yaml(tmp_path, {"task_chains": {}})
        from core.prompt_manager import PromptManager
        assert PromptManager(path)._get_base_sections() == {}


# ---------------------------------------------------------------------------
# Tests: _get_task_chains
# ---------------------------------------------------------------------------

class TestGetTaskChains:

    def test_returns_task_chains_dict(self, tmp_path):
        """Must return the full task_chains dict when the section is present."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        chains = PromptManager(path)._get_task_chains()
        assert "basic_validation"    in chains
        assert "extended_validation" in chains

    def test_returns_empty_dict_when_section_missing(self, tmp_path):
        """Must return an empty dict when the task_chains section is absent."""
        content = {"base_generic": {"role": "Only base."}}
        path = write_yaml(tmp_path, content)
        from core.prompt_manager import PromptManager
        assert PromptManager(path)._get_task_chains() == {}