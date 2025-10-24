
import os
import requests
from typing import Any, Dict, Optional

DEFAULT_TIMEOUT = float(os.getenv("CLIENT_DEFAULT_TIMEOUT", "10"))

class MockRAGClient:
    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url = base_url or os.getenv("EVAL_BACKEND_URL", "http://localhost:8001")
        self.timeout = timeout or DEFAULT_TIMEOUT
        self._answer_url = f"{self.base_url.rstrip('/')}/api/answer"
        self._health_url = f"{self.base_url.rstrip('/')}/health"

    def health(self) -> Dict[str, Any]:
        r = requests.get(self._health_url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_answer(self, query: str, k: int = 5) -> Dict[str, Any]:
        payload = {"query": query, "k": k}
        r = requests.post(self._answer_url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
