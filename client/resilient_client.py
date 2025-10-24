
import os
import time
import json
import random
import requests
from typing import Any, Dict, Optional, Tuple
from requests.exceptions import Timeout, ConnectionError as ReqConnectionError, HTTPError

DEFAULT_TIMEOUT = float(os.getenv("CLIENT_DEFAULT_TIMEOUT", "10"))
MAX_RETRIES = int(os.getenv("CLIENT_MAX_RETRIES", "3"))
BACKOFF_BASE = float(os.getenv("CLIENT_BACKOFF_BASE", "0.25"))
BACKOFF_JITTER = float(os.getenv("CLIENT_BACKOFF_JITTER", "0.25"))
LONG_TAIL_GUARD_MS: int = int(os.getenv("CLIENT_LONG_TAIL_GUARD_MS", "15000"))

ERROR_BUCKET_UNKNOWN = "unknown"


class ResilientRAGClient:
    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        resolved = base_url or os.getenv("EVAL_BACKEND_URL", "http://localhost:8001")
        self.base_url = resolved.rstrip("/")
        self.timeout = float(timeout) if timeout is not None else DEFAULT_TIMEOUT
        self.answer_url = f"{self.base_url}/api/answer"
        self.health_url = f"{self.base_url}/health"

    def health(self) -> Dict[str, Any]:
        r = requests.get(self.health_url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def classify_error(self, exc: Exception, response: Optional[requests.Response]) -> str:
        """Map exception or HTTP response to a canonical error bucket."""
        if isinstance(exc, Timeout):
            return "timeout"
        if isinstance(exc, ReqConnectionError):
            return "networkerror"

        if isinstance(exc, HTTPError):
            resp = getattr(exc, "response", None) or response
            if resp is not None:
                sc = resp.status_code
                if sc == 429:
                    return "ratelimited"
                if 500 <= sc < 600:
                    return "servererror"
                if 400 <= sc < 500:
                    return "clienterror"
            return "servererror"

        if response is not None:
            sc = response.status_code
            if sc == 429:
                return "ratelimited"
            if 500 <= sc < 600:
                return "servererror"
            if 400 <= sc < 500:
                return "clienterror"

        if isinstance(exc, json.JSONDecodeError):
            return "malformed"

        return ERROR_BUCKET_UNKNOWN

    def _sleep_backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter. attempt starts at 1."""
        base = BACKOFF_BASE * (2 ** (attempt - 1))
        jitter = random.uniform(0.0, BACKOFF_JITTER)
        time.sleep(base + jitter)

    def get_answer(self, query: str, k: int = 5) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Returns (response_json_or_none, net_diag)
        net_diag = {latency_ms, retries_used, error_class}
        """
        payload = {"query": query, "k": int(k)}
        start = time.perf_counter()
        retries_used = 0
        last_error_class: Optional[str] = None

        for attempt_idx in range(MAX_RETRIES + 1):
            resp: Optional[requests.Response] = None
            try:
                resp = requests.post(self.answer_url, json=payload, timeout=self.timeout)

                if not resp.ok:
                    last_error_class = self.classify_error(HTTPError(), resp)
                    raise HTTPError(f"HTTP {resp.status_code}", response=resp)

                ctype = (resp.headers.get("content-type") or "").lower()
                if not ctype.startswith("application/json"):
                    try:
                        data = resp.json()
                    except Exception as je:
                        last_error_class = self.classify_error(je, resp)
                        raise je
                else:
                    data = resp.json()

                latency_ms = int((time.perf_counter() - start) * 1000.0)
                latency_ms = min(latency_ms, LONG_TAIL_GUARD_MS)
                return data, {
                    "latency_ms": latency_ms,
                    "retries_used": retries_used,
                    "error_class": "",
                }

            except KeyboardInterrupt:
                raise  
            except Exception as exc:
                last_error_class = self.classify_error(exc, resp)
                if attempt_idx >= MAX_RETRIES:
                    break
                retries_used += 1
                self._sleep_backoff(retries_used)
                continue

        latency_ms = int((time.perf_counter() - start) * 1000.0)
        latency_ms = min(latency_ms, LONG_TAIL_GUARD_MS)
        return None, {
            "latency_ms": latency_ms,
            "retries_used": retries_used,
            "error_class": last_error_class or ERROR_BUCKET_UNKNOWN,
        }
