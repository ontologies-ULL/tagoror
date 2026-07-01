"""
Unit tests for PromptManager
==============================
Located at: src/core.

YAML structure (matches the real class attributes):

  base_generic:
    role: |
      You are an expert ontology validator...
    constraints: |
      Do not invent information...
  evaluation_suites:
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
    owl_validations:
      - id: default_check
        prompt: |
          Default suite used when no name is provided.

Covers (matches PromptManager in prompt_manager.py):
  - _load_from_disk: reads a valid YAML file correctly
  - _load_from_disk: returns empty dict for an empty file
  - _load_from_disk: raises FileNotFoundError for a missing file
  - _load_from_disk: loaded content matches YAML exactly
  - __init__: falls back to _DEFAULT_PROMPT_FILE when file_path is None
  - get_assembled_system_prompt: joins sections as "### TITLE\ncontent",
                                  separated by a blank line
  - get_assembled_system_prompt: role content present in result
  - get_assembled_system_prompt: constraints content present in result
  - get_assembled_system_prompt: single section returns value without separator
  - get_assembled_system_prompt: raises KeyError when base_generic missing
  - get_assembled_system_prompt: raises KeyError for empty YAML
  - get_assembled_system_prompt: returns a string
  - get_evaluation_suite: returns list for known suite
  - get_evaluation_suite: correct number of tasks per suite
  - get_evaluation_suite: each task has 'id' and 'prompt' keys
  - get_evaluation_suite: task ids/prompts match YAML content exactly
  - get_evaluation_suite: different suites return different tasks
  - get_evaluation_suite: default suite name is 'owl_validations'
  - get_evaluation_suite: raises KeyError for unknown suite name
  - get_evaluation_suite: raises KeyError when evaluation_suites section is absent
  - _get_base_sections: returns base_generic dict when present
  - _get_base_sections: raises KeyError when base_generic is absent or empty

Testing strategy:
  _load_from_disk opens a real file, so we use tmp_path (pytest built-in)
  to create temporary YAML files on disk. This validates real YAML parsing
  without mocking the filesystem, and does not depend on any file in
  src/core. being present at test time.

Note: the class exposes get_evaluation_suite() (not get_task_chain), reads
the "evaluation_suites" YAML key (not "task_chains"), has no legacy
"base_generica" fallback, and _get_base_sections() raises KeyError instead
of returning {} when the block is missing/empty.
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
    "evaluation_suites": {
        "basic_validation": [
            {"id": "check_class",      "prompt": "Verify whether the entity is a valid OWL class."},
            {"id": "check_properties", "prompt": "List relevant properties and check their consistency."},
        ],
        "extended_validation": [
            {"id": "detect_inconsistencies", "prompt": "Point out logical inconsistencies if they exist."},
            {"id": "suggest_improvements",   "prompt": "Suggest OWL modeling improvements."},
        ],
        "owl_validations": [
            {"id": "default_check", "prompt": "Default suite used when no name is provided."},
        ],
    },
}


def write_yaml(tmp_path: Path, content: dict, filename: str = "prompts.yaml") -> str:
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
# Tests: _load_from_disk / __init__
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

    def test_default_file_path_used_when_none_given(self, mocker):
        """When file_path is None, __init__ must fall back to _DEFAULT_PROMPT_FILE."""
        from core.prompt_manager import PromptManager
        mock_load = mocker.patch.object(
            PromptManager, "_load_from_disk", return_value={"base_generic": {"role": "x"}}
        )
        PromptManager()
        mock_load.assert_called_once_with(str(PromptManager._DEFAULT_PROMPT_FILE))


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
        """Sections must render as '### TITLE\\ncontent' joined by '\\n\\n'."""
        content = {
            "base_generic": {
                "role":        "ROLE_CONTENT",
                "constraints": "CONSTRAINTS_CONTENT",
            }
        }
        path = write_yaml(tmp_path, content)
        from core.prompt_manager import PromptManager
        result = PromptManager(path).get_assembled_system_prompt()
        expected_role = "### ROLE\nROLE_CONTENT"
        expected_constraints = "### CONSTRAINTS\nCONSTRAINTS_CONTENT"
        assert expected_role in result
        assert expected_constraints in result
        assert result.count("\n\n") == 1

    def test_single_section_no_separator(self, tmp_path):
        """A single section must return its value with no trailing separator."""
        content = {"base_generic": {"role": "Only role here"}}
        path = write_yaml(tmp_path, content)
        from core.prompt_manager import PromptManager
        result = PromptManager(path).get_assembled_system_prompt()
        assert result == "### ROLE\nOnly role here"

    def test_returns_string(self, tmp_path):
        """Return value must always be a plain string."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        result = PromptManager(path).get_assembled_system_prompt()
        assert isinstance(result, str)

    def test_raises_key_error_when_base_generic_missing(self, tmp_path):
        """Must raise KeyError with a descriptive message when base_generic is absent."""
        content = {"evaluation_suites": {}}
        path = write_yaml(tmp_path, content)
        from core.prompt_manager import PromptManager
        with pytest.raises(KeyError, match="Missing base prompt"):
            PromptManager(path).get_assembled_system_prompt()

    def test_raises_key_error_for_empty_yaml(self, tmp_path):
        """Must raise KeyError when the YAML file is completely empty."""
        path = write_raw(tmp_path, "")
        from core.prompt_manager import PromptManager
        with pytest.raises(KeyError):
            PromptManager(path).get_assembled_system_prompt()


# ---------------------------------------------------------------------------
# Tests: get_evaluation_suite
# ---------------------------------------------------------------------------

class TestGetEvaluationSuite:

    def test_returns_list_for_basic_validation(self, tmp_path):
        """get_evaluation_suite('basic_validation') must return a list."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        suite = PromptManager(path).get_evaluation_suite("basic_validation")
        assert isinstance(suite, list)

    def test_basic_validation_has_two_tasks(self, tmp_path):
        """basic_validation must contain exactly 2 tasks."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        suite = PromptManager(path).get_evaluation_suite("basic_validation")
        assert len(suite) == 2

    def test_extended_validation_has_two_tasks(self, tmp_path):
        """extended_validation must contain exactly 2 tasks."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        suite = PromptManager(path).get_evaluation_suite("extended_validation")
        assert len(suite) == 2

    def test_each_task_has_id_key(self, tmp_path):
        """Every task dict must have an 'id' key."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        suite = PromptManager(path).get_evaluation_suite("basic_validation")
        for task in suite:
            assert "id" in task

    def test_each_task_has_prompt_key(self, tmp_path):
        """Every task dict must have a 'prompt' key."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        suite = PromptManager(path).get_evaluation_suite("basic_validation")
        for task in suite:
            assert "prompt" in task

    def test_basic_validation_task_ids_match_yaml(self, tmp_path):
        """basic_validation task ids must match exactly: check_class, check_properties."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        suite = PromptManager(path).get_evaluation_suite("basic_validation")
        assert [t["id"] for t in suite] == ["check_class", "check_properties"]

    def test_extended_validation_task_ids_match_yaml(self, tmp_path):
        """extended_validation task ids must match: detect_inconsistencies, suggest_improvements."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        suite = PromptManager(path).get_evaluation_suite("extended_validation")
        assert [t["id"] for t in suite] == ["detect_inconsistencies", "suggest_improvements"]

    def test_first_task_prompt_matches_yaml(self, tmp_path):
        """The first task prompt must match the YAML value exactly."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        suite = PromptManager(path).get_evaluation_suite("basic_validation")
        assert suite[0]["prompt"] == "Verify whether the entity is a valid OWL class."

    def test_suites_are_independent(self, tmp_path):
        """basic_validation and extended_validation must return different task lists."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        pm = PromptManager(path)
        basic    = [t["id"] for t in pm.get_evaluation_suite("basic_validation")]
        extended = [t["id"] for t in pm.get_evaluation_suite("extended_validation")]
        assert basic != extended

    def test_default_suite_name_is_owl_validations(self, tmp_path):
        """Calling with no argument must default to the 'owl_validations' suite."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        suite = PromptManager(path).get_evaluation_suite()
        assert [t["id"] for t in suite] == ["default_check"]

    def test_raises_key_error_for_unknown_suite(self, tmp_path):
        """Must raise KeyError containing the unknown suite name."""
        path = write_yaml(tmp_path, REAL_YAML_CONTENT)
        from core.prompt_manager import PromptManager
        with pytest.raises(KeyError, match="nonexistent_suite"):
            PromptManager(path).get_evaluation_suite("nonexistent_suite")

    def test_raises_key_error_when_evaluation_suites_section_missing(self, tmp_path):
        """Must raise KeyError when the evaluation_suites section is absent entirely."""
        content = {"base_generic": {"role": "Only base, no suites."}}
        path = write_yaml(tmp_path, content)
        from core.prompt_manager import PromptManager
        with pytest.raises(KeyError):
            PromptManager(path).get_evaluation_suite("basic_validation")


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

    def test_raises_key_error_when_base_generic_key_missing(self, tmp_path):
        """Must raise KeyError when base_generic is not present at all."""
        path = write_yaml(tmp_path, {"evaluation_suites": {}})
        from core.prompt_manager import PromptManager
        with pytest.raises(KeyError, match="Missing base prompt block"):
            PromptManager(path)._get_base_sections()

    def test_raises_key_error_when_base_generic_is_empty(self, tmp_path):
        """Must raise KeyError when base_generic is present but empty."""
        path = write_yaml(tmp_path, {"base_generic": {}})
        from core.prompt_manager import PromptManager
        with pytest.raises(KeyError, match="Missing base prompt block"):
            PromptManager(path)._get_base_sections()