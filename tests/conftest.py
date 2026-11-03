"""
Global pytest configuration for live LLM integration tests.
"""
import pytest
from dotenv import load_dotenv

load_dotenv()

def pytest_addoption(parser):
    parser.addoption(
        "--live-models",
        action="store_true",
        default=False,
        help="Run real inference against local LLM models defined in .env to benchmark prompts."
    )

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live_llm: mark test to run only when --live-models flag is provided"
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live-models"):
        skip_live = pytest.mark.skip(reason="Pass --live-models to execute tests against live LLMs")
        for item in items:
            if "live_llm" in item.keywords:
                item.add_marker(skip_live)
