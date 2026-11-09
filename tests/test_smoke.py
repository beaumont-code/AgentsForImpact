"""Smoke tests — run before and after refactoring to verify nothing broke."""
import pytest
import sys
import os
from unittest.mock import patch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")


# --- 1. All modules import without error ---

def test_import_config():
    import config
    assert config.NVIDIA_API_KEY


def test_import_prompts():
    from prompts import SYSTEM_PROMPT, VISION_DESCRIBE_PROMPT, CONTINUOUS_MODE_PROMPT_TEMPLATE
    assert "Third Eye" in SYSTEM_PROMPT
    assert len(VISION_DESCRIBE_PROMPT) > 0


def test_import_agent():
    from agent import run_agentic_loop, create_initial_history, TOOL_DEFINITIONS


def test_import_vision():
    from vision import capture_and_describe, capture_frame_raw, read_text, release_camera


def test_import_speech():
    from speech import listen, speak, play_proximity_beep


def test_import_navigation():
    from navigation import get_directions


def test_import_orchestrator():
    from orchestrator import Orchestrator, AppState


# --- 2. Config constants exist and have sane values ---

def test_config_constants():
    import config
    assert config.NEMOTRON_MODEL
    assert config.VISION_MODEL
    assert config.CONTINUOUS_MODE_INTERVAL > 0
    assert config.FRAME_WIDTH > 0


# --- 3. Orchestrator wires up correctly ---

def test_orchestrator_init():
    with patch("orchestrator.create_initial_history", return_value=[]):
        from orchestrator import Orchestrator
        orch = Orchestrator()
        assert orch.state.value == "listening"
        assert "capture_and_describe" in orch.tool_handlers
        assert "read_text" in orch.tool_handlers
        assert "get_directions" in orch.tool_handlers
        assert "toggle_continuous_mode" in orch.tool_handlers


# --- 4. Proximity detection logic ---

def test_proximity_alert_close():
    with patch("orchestrator.create_initial_history", return_value=[]):
        from orchestrator import Orchestrator
        orch = Orchestrator()
        assert orch._is_proximity_alert("Person ahead, 2 feet")
        assert orch._is_proximity_alert("Wall, 3 ft away")
        assert orch._is_proximity_alert("object very close")


def test_proximity_alert_far():
    with patch("orchestrator.create_initial_history", return_value=[]):
        from orchestrator import Orchestrator
        orch = Orchestrator()
        assert not orch._is_proximity_alert("Door ahead, 10 feet")
        assert not orch._is_proximity_alert("Clear hallway")


# --- 5. Agent tool definitions are valid ---

def test_agent_tools_schema():
    from agent import TOOL_DEFINITIONS
    assert len(TOOL_DEFINITIONS) == 4
    tool_names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert tool_names == {"capture_and_describe", "read_text", "get_directions", "toggle_continuous_mode"}


# --- 6. Prompt templates have required placeholders ---

def test_continuous_prompt_template():
    from prompts import CONTINUOUS_MODE_PROMPT_TEMPLATE
    result = CONTINUOUS_MODE_PROMPT_TEMPLATE.format(
        seconds_ago=5, previous_description="old", current_description="new", current_nav_step="none"
    )
    assert "old" in result and "new" in result
