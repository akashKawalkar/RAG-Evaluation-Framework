from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import httpx

REAL_BACKEND_URL = os.getenv("REAL_BACKEND_URL")

app = FastAPI(title="RAG Backend Proxy", version="1.0.0")

class AnswerRequest(BaseModel):
    query: str
    k: Optional[int] = 5

class TopChunk(BaseModel):
    id: str
    chunk: str
    score: float
    source: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    subqueries: List[str]
    top_chunks: List[TopChunk]


@app.get("/health")
async def health() -> Dict[str, Any]:
    if not REAL_BACKEND_URL:
        return {"status": "ok", "service": "proxy", "upstream": "not-configured"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{REAL_BACKEND_URL.rstrip('/')}/health")
            if r.status_code == 200:
                return {"status": "ok", "service": "proxy", "upstream": "ok"}
            return {"status": "ok", "service": "proxy", "upstream": f"bad-{r.status_code}"}
    except Exception:
        return {"status": "ok", "service": "proxy", "upstream": "error"}


@app.post("/api/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest):
    if not REAL_BACKEND_URL:
        raise HTTPException(status_code=500, detail="REAL_BACKEND_URL not set")

    upstream_url = f"{REAL_BACKEND_URL.rstrip('/')}/api/answer"

    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(upstream_url, json=req.dict())
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Upstream timeout")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e!s}")

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()

    def g(obj, *keys, default=None):
        for k in keys:
            if isinstance(obj, dict) and k in obj and obj[k] is not None:
                return obj[k]
        return default

    raw_chunks = data.get("top_chunks") or data.get("chunks") or data.get("retrieved") or []
    norm_chunks: List[Dict[str, Any]] = []
    for i, tc in enumerate(raw_chunks):
        if isinstance(tc, str):
            norm_chunks.append({
                "id": f"doc-{i+1}",
                "chunk": tc,
                "score": 0.0,
                "source": None,
            })
            continue

        norm_chunks.append({
            "id": str(g(tc, "id", "doc_id", "uuid", default=f"doc-{i+1}")),
            "chunk": g(tc, "chunk", "text", "content", default=""),
            "score": float(g(tc, "score", "similarity", "relevance", default=0.0)),
            "source": g(tc, "source", "collection", "path", "file", default=None),
        })

    payload = {
        "answer": g(data, "answer", "output", "result", default=""),
        "subqueries": data.get("subqueries") or [req.query],
        "top_chunks": norm_chunks,
    }

    return payload
