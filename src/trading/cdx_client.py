"""
crypto.com Exchange v1 API client.
Handles authentication, signing, and all REST calls the bot needs.
Never import API keys directly — always read from environment variables.
"""

import hashlib
import hmac
import json
import os
import time
from typing import Any

import httpx

CDX_BASE = "https://api.crypto.com/exchange/v1"
CDX_TIMEOUT = 10  # seconds


class CDXError(Exception):
    """Raised when the crypto.com API returns a non-zero code."""
    pass


class CDXClient:
    def __init__(self):
        self.api_key = os.environ["CDX_API_KEY"]
        self.secret  = os.environ["CDX_API_SECRET"]
        self.client  = httpx.Client(timeout=CDX_TIMEOUT)

    # ------------------------------------------------------------------ #
    #  Auth helpers                                                         #
    # ------------------------------------------------------------------ #

    def _sign(self, method: str, req_id: int, params: dict, nonce: str) -> str:
        """
        crypto.com v1 signature:
          sig_payload = method + id + api_key + params_string + nonce
          sig         = HMAC-SHA256(sig_payload, secret).hexdigest()

        params_string: sort params by key, concatenate key+value recursively.
        Nested dicts are flattened key-by-key; lists are joined as-is.
        """
        def flatten(d: dict) -> str:
            out = ""
            for k in sorted(d.keys()):
                v = d[k]
                if isinstance(v, dict):
                    out += k + flatten(v)
                elif isinstance(v, list):
                    out += k + "".join(str(i) for i in v)
                else:
                    out += k + str(v)
            return out

        param_str   = flatten(params) if params else ""
        sig_payload = f"{method}{req_id}{self.api_key}{param_str}{nonce}"
        return hmac.new(
            self.secret.encode("utf-8"),
            sig_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _post(self, method: str, params: dict | None = None) -> dict:
        """Sign and POST a private request. Returns the result dict."""
        params  = params or {}
        nonce   = str(int(time.time() * 1000))
        req_id  = int(nonce)
        payload = {
            "id":      req_id,
            "method":  method,
            "api_key": self.api_key,
            "params":  params,
            "nonce":   nonce,
            "sig":     self._sign(method, req_id, params, nonce),
        }
        resp = self.client.post(f"{CDX_BASE}/{method}", json=payload)
        resp.raise_for_status()
        body = resp.json()
        if body.get("code", 0) != 0:
            raise CDXError(f"API error {body.get('code')}: {body.get('message', body)}")
        return body.get("result", {})

    def _get_public(self, endpoint: str, params: dict | None = None) -> dict:
        """GET a public endpoint (no auth required)."""
        resp = self.client.get(f"{CDX_BASE}/{endpoint}", params=params or {})
        resp.raise_for_status()
        body = resp.json()
        if body.get("code", 0) != 0:
            raise CDXError(f"API error {body.get('code')}: {body.get('message', body)}")
        return body.get("result", {})

    # ------------------------------------------------------------------ #
    #  Account                                                              #
    # ------------------------------------------------------------------ #

    def get_balance(self, currency: str = "USDT") -> float:
        """Return the available balance for a given currency."""
        result = self._post("private/get-account-summary", {"currency": currency})
        for account in result.get("accounts", []):
            if account.get("currency") == currency:
                return float(account.get("available", 0))
        return 0.0

    # ------------------------------------------------------------------ #
    #  Market data                                                          #
    # ------------------------------------------------------------------ #

    def get_ticker(self, instrument: str = "BTC_USDT") -> dict:
        """Return the latest ticker for an instrument."""
        result = self._get_public("public/get-ticker", {"instrument_name": instrument})
        data   = result.get("data", {})
        return {
            "price":    float(data.get("a", 0)),   # last traded price
            "bid":      float(data.get("b", 0)),
            "ask":      float(data.get("k", 0)),
            "high_24h": float(data.get("h", 0)),
            "low_24h":  float(data.get("l", 0)),
            "volume":   float(data.get("v", 0)),
        }

    def get_candlesticks(
        self,
        instrument: str = "BTC_USDT",
        timeframe: str  = "1D",
        count:     int  = 30,
    ) -> list[dict]:
        """
        Return OHLCV candles for regime classification.
        timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1D
        """
        result  = self._get_public(
            "public/get-candlestick",
            {"instrument_name": instrument, "timeframe": timeframe, "count": count},
        )
        candles = result.get("data", [])
        return [
            {
                "ts":     c["t"],
                "open":   float(c["o"]),
                "high":   float(c["h"]),
                "low":    float(c["l"]),
                "close":  float(c["c"]),
                "volume": float(c["v"]),
            }
            for c in candles
        ]

    # ------------------------------------------------------------------ #
    #  Orders                                                               #
    # ------------------------------------------------------------------ #

    def place_limit_order(
        self,
        instrument: str,
        side:       str,    # BUY or SELL
        price:      float,
        quantity:   float,
    ) -> str:
        """Place a GTC limit order. Returns the order ID."""
        result = self._post(
            "private/create-order",
            {
                "instrument_name": instrument,
                "side":            side.upper(),
                "type":            "LIMIT",
                "price":           f"{price:.2f}",
                "quantity":        f"{quantity:.6f}",
                "time_in_force":   "GOOD_TILL_CANCEL",
                "exec_inst":       ["POST_ONLY"],   # maker only — saves fees
            },
        )
        return result.get("order_id", "")

    def cancel_order(self, instrument: str, order_id: str) -> None:
        """Cancel a single order by ID."""
        self._post(
            "private/cancel-order",
            {"instrument_name": instrument, "order_id": order_id},
        )

    def cancel_all_orders(self, instrument: str) -> None:
        """Cancel every open order for an instrument (kill switch use)."""
        self._post(
            "private/cancel-all-orders",
            {"instrument_name": instrument},
        )

    def get_open_orders(self, instrument: str) -> list[dict]:
        """Return all currently open orders for an instrument."""
        result = self._post(
            "private/get-open-orders",
            {"instrument_name": instrument},
        )
        return result.get("order_list", [])

    def get_order_history(self, instrument: str, limit: int = 20) -> list[dict]:
        """Return recent order history (filled and cancelled)."""
        result = self._post(
            "private/get-order-history",
            {"instrument_name": instrument, "page_size": limit},
        )
        return result.get("order_list", [])
