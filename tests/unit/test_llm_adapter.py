from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.llm import LocalPhiModel


def test_local_phi_model_parses_structured_json() -> None:
    with patch("langchain_community.llms.Ollama") as ollama:
        ollama.return_value.invoke.return_value = '{"escalate": true}'
        model = LocalPhiModel()
        assert model.invoke_json("Assess this alert") == {"escalate": True}


def test_local_phi_model_rejects_invalid_json() -> None:
    with patch("langchain_community.llms.Ollama") as ollama:
        ollama.return_value.invoke.return_value = "not json"
        model = LocalPhiModel()
        with pytest.raises(ValueError, match="invalid JSON"):
            model.invoke_json("Assess this alert")
