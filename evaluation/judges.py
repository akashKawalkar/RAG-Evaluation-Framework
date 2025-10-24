from __future__ import annotations

import json
import re
from typing import Dict, Any, Protocol, List, Optional, Tuple


class JudgeOutput(Dict[str, Any]):
    pass


class Judge(Protocol):
    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_context: List[Dict[str, Any]],
        gold_passages: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JudgeOutput: ...


def _norm01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return max(0.0, min(1.0, v))


def _safe_len(s: Optional[str]) -> int:
    return len(s) if isinstance(s, str) else 0


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _token_overlap(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / float(max(1, len(ta)))


def _coverage_hits_answer_vs_chunks(answer: str, chunks: List[str]) -> int:
    ah = set(_tokenize(answer))
    hits = 0
    for ch in chunks:
        if ah and (ah & set(_tokenize(ch))):
            hits += 1
    return hits


class HeuristicJudge:
    def __init__(self, name: str = "heuristic-judge", version: str = "v1"):
        self.name = name
        self.version = version

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_context: List[Dict[str, Any]],
        gold_passages: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JudgeOutput:
        answer = answer or ""
        chunks = [c.get("chunk", "") for c in (retrieved_context or []) if isinstance(c, dict)]
        scores = [c.get("score") for c in (retrieved_context or []) if isinstance(c, dict) and isinstance(c.get("score"), (int, float))]
        top1 = float(scores[0]) if scores else 0.0

        a_len = _safe_len(answer.strip())
        q_overlap = _token_overlap(query or "", answer)
        helpfulness = 0.6 * _norm01(a_len, 20, 400) + 0.4 * q_overlap

        if gold_passages:
            best = 0.0
            for g in gold_passages:
                best = max(best, _token_overlap(g or "", answer))
            correctness = best
        else:
            cov_hits = _coverage_hits_answer_vs_chunks(answer, chunks)
            correctness = _norm01(cov_hits, 0, max(1, min(5, len(chunks))))

        ans_tokens = _tokenize(answer)
        ctx_vocab = set()
        for ch in chunks:
            ctx_vocab.update(_tokenize(ch))
        grounded_tokens = sum(1 for t in ans_tokens if t in ctx_vocab)
        grounding_ratio = (grounded_tokens / float(max(1, len(ans_tokens)))) if ans_tokens else 0.0
        grounding = 0.85 * grounding_ratio + 0.15 * _norm01(top1, 0.1, 1.0)

        return JudgeOutput(
            helpfulness=round(float(helpfulness), 3),
            correctness=round(float(correctness), 3),
            grounding=round(float(grounding), 3),
            notes=f"len={a_len}, q_overlap={q_overlap:.2f}, top1={top1:.3f}",
            judge_name=self.name,
            judge_version=self.version,
        )


class LLMJudge:
    """
    Adapter that calls an external/local LLM with a fixed prompt and temperature=0.
    Supply a callable 'inference_fn(prompt: str) -> str' that returns JSON text.
    Deterministic by default (temperature=0, seed control).
    """

    def __init__(
        self,
        inference_fn,
        name: str = "llm-judge",
        version: str = "v1",
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: Optional[int] = 42,
    ):
        self.inference_fn = inference_fn
        self.name = name
        self.version = version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    @staticmethod
    def _clamp01(x: float) -> float:
        try:
            xf = float(x)
        except Exception:
            return 0.0
        if xf < 0.0:
            return 0.0
        if xf > 1.0:
            return 1.0
        return xf

    @staticmethod
    def _safe_call_infer(fn, prompt: str, seed: Optional[int]) -> str:
        try:
            return fn(prompt, seed) 
        except TypeError:
            return fn(prompt) 
    @staticmethod
    def build_prompt(
        query: str,
        answer: str,
        retrieved_context: List[Dict[str, Any]],
        gold_passages: Optional[List[str]],
    ) -> str:
        def fmt_chunk(i: int, c: Dict[str, Any]) -> str:
            cid = c.get("id", f"chunk-{i+1}")
            score = c.get("score", "")
            text = c.get("chunk", "")
            return f"[{i+1}] id={cid} score={score}\n{text}"

        ctx = "\n\n".join([fmt_chunk(i, c) for i, c in enumerate(retrieved_context or [])])
        gold = "\n".join(gold_passages or [])

        return f"""
You are a strict evaluator of Retrieval-Augmented Generation (RAG) answers.

Task:
- Judge the response on three criteria in [0,1]: helpfulness, correctness, grounding.
- Be conservative and deterministic (assume temperature≈0).
- Output ONLY JSON with keys: helpfulness, correctness, grounding, notes.

Grounding requirement:
- The answer should be supported by the retrieved context.
- Reward explicit agreement with provided context and penalize unsupported claims.
- If gold passages exist, prioritize correctness against gold.

Inputs
Query:
{query}

Answer:
{answer}

Retrieved Context:
{ctx if ctx else "[none]"}

Gold Passages (may be empty):
{gold if gold else "[none]"}

Scoring rubric:
- helpfulness: clarity, completeness, and direct relevance to the query.
- correctness: fidelity to gold facts when available; otherwise to retrieved context.
- grounding: evidence present in retrieved context; penalize hallucinations.

Return JSON now with this schema:
{{
  "helpfulness": 0.0,
  "correctness": 0.0,
  "grounding": 0.0,
  "notes": "short rationale"
}}
""".strip()

    def _parse_json_scores(self, raw: str) -> Tuple[float, float, float, str]:
        """
        JSON parsing:
        - Try strict JSON first.
        - On failure, lenient extraction of 0..1 floats in order (h, c, g).
        - Clamp values to [0,1]; notes carry a brief rationale or a fallback marker.
        """
        try:
            d = json.loads(raw)
            h = self._clamp01(d.get("helpfulness", 0.0))
            c = self._clamp01(d.get("correctness", 0.0))
            g = self._clamp01(d.get("grounding", 0.0))
            notes = str(d.get("notes", "") or "")
            return float(h), float(c), float(g), notes[:300]
        except Exception:
            nums = re.findall(r"(?<!\\d)(?:0(?:\\.\\d+)?|1(?:\\.0+)?)", raw)
            try:
                h = self._clamp01(float(nums[0])) if len(nums) > 0 else 0.0
                c = self._clamp01(float(nums[1])) if len(nums) > 1 else 0.0
                g = self._clamp01(float(nums[2])) if len(nums) > 2 else 0.0
            except Exception:
                h = c = g = 0.0
            m = re.search(r'notes["\']?\s*[:=]\s*["\']([^"\']{0,200})', raw, re.IGNORECASE)
            notes = m.group(1) if m else "parse_fallback"
            return float(h), float(c), float(g), notes

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_context: List[Dict[str, Any]],
        gold_passages: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JudgeOutput:
        prompt = self.build_prompt(query, answer, retrieved_context, gold_passages)
        raw = self._safe_call_infer(self.inference_fn, prompt, self.seed)
        h, c, g, notes = self._parse_json_scores(raw)
        return JudgeOutput(
            helpfulness=round(float(h), 3),
            correctness=round(float(c), 3),
            grounding=round(float(g), 3),
            notes=notes,
            judge_name=self.name,
            judge_version=self.version,
        )
