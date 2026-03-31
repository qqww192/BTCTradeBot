"""
Tests for analyse.py
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analyse import _parse_json_response, _get_client, AnalysisBudget


class TestParseJsonResponse:
    def test_json_in_code_block(self):
        text = '```json\n{"verdict": "BUY", "risk": "3"}\n```'
        result = _parse_json_response(text)
        assert result["verdict"] == "BUY"

    def test_raw_json(self):
        text = '{"verdict": "HOLD", "risk": "5"}'
        result = _parse_json_response(text)
        assert result["verdict"] == "HOLD"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON found"):
            _parse_json_response("No json here")


class TestGetClient:
    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
                _get_client()


class TestAnalysisBudget:
    def test_initial_state(self):
        budget = AnalysisBudget(max_requests=5)
        assert budget.remaining == 5
        assert not budget.exhausted

    def test_consume(self):
        budget = AnalysisBudget(max_requests=2)
        assert budget.consume() is True
        assert budget.remaining == 1
        assert budget.consume() is True
        assert budget.remaining == 0
        assert budget.exhausted
        assert budget.consume() is False
