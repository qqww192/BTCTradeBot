"""
Tests for fetch_portfolio.py
"""

import base64
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add src to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fetch_portfolio import _get_headers, t212_to_yfinance


class TestGetHeaders:
    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="T212_API_KEY"):
                _get_headers()

    def test_api_key_only_returns_direct_auth(self):
        with patch.dict(os.environ, {"T212_API_KEY": "key123"}, clear=True):
            headers = _get_headers()
            assert headers["Authorization"] == "key123"

    def test_both_keys_returns_basic_auth(self):
        with patch.dict(os.environ, {"T212_API_KEY": "key123", "T212_SECRET_KEY": "secret456"}, clear=True):
            headers = _get_headers()
            expected = base64.b64encode(b"key123:secret456").decode()
            assert headers["Authorization"] == f"Basic {expected}"


class TestT212ToYfinance:
    def test_us_stock(self):
        assert t212_to_yfinance("AAPL_US_EQ") == "AAPL"

    def test_us_stock_with_underscore(self):
        assert t212_to_yfinance("BRK_B_US_EQ") == "BRK-B"

    def test_london_stock(self):
        assert t212_to_yfinance("VUAGl_EQ") == "VUAG.L"

    def test_unknown_ticker(self):
        assert t212_to_yfinance("UNKNOWN") == "UNKNOWN"

    def test_empty(self):
        assert t212_to_yfinance("") == ""
