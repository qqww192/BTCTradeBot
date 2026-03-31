"""
conftest.py
Pre-mock libraries that depend on native C extensions or are not installed
in the test environment (e.g. CI sandboxes).
"""

import sys
from unittest.mock import MagicMock

# These modules may not be installed or depend on native backends.
# Mock them before any test file imports src modules.
MOCK_MODULES = [
    "google",
    "google.genai",
    "google.genai.errors",
    "google.generativeai",
    "google.generativeai.types",
    "google.generativeai.protos",
    "google.oauth2",
    "google.oauth2.service_account",
    "googleapiclient",
    "googleapiclient.discovery",
    "googleapiclient.errors",
    "httpx",
    "yfinance",
    "numpy",
    "dotenv",
]

for mod in MOCK_MODULES:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()
