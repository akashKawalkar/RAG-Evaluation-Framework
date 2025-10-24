from evaluation.judges import HeuristicJudge, LLMJudge
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import json
import os
import time
import math
import random
import collections
from datetime import datetime
from pathlib import Path
import csv
from client.resilient_client import ResilientRAGClient
from evaluation.judges import HeuristicJudge, LLMJudge

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    simple_mrr,
    hitatk,
    mean_reciprocal_rank,
    dcgatk,
    ndcgatk
)

JsonObj = Dict[str, Any]


def load_jsonl(path: Union[str, Path]):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def append_jsonl(path: Union[str, Path], record: JsonObj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = min(len(arr) - 1, max(0, int(math.ceil(p * len(arr)) - 1)))
    return float(arr[idx])

def _extract_retrieved_ids(resp: JsonObj) -> List[str]:
    items = (resp or {}).get("top_chunks", []) or []
    ids: List[str] = []
    for it in items:
        if isinstance(it, dict):
            cid = it.get("id")
        else:
            cid = getattr(it, "id", None)
        if isinstance(cid, str):
            ids.append(cid)
    return ids

def _extract_scores(resp: JsonObj) -> List[float]:
    items = (resp or {}).get("top_chunks", []) or []
    scores: List[float] = []
    for it in items:
        if isinstance(it, dict):
            s = it.get("score")
        else:
            s = getattr(it, "score", None)
        if isinstance(s, (int, float)):
            scores.append(float(s))
    return scores

def _compute_diagnostics(resp: JsonObj, net_diag: Optional[JsonObj], relevant_ids: Set[str], k: int) -> JsonObj:
    ids = _extract_retrieved_ids(resp or {})
    scores = _extract_scores(resp or {})
    top1 = scores[0] if scores else None
    d: JsonObj = {
        "retrieved_count": len(ids),
        "scores_mean": (sum(scores) / len(scores)) if scores else None,
        "scores_std": None, 
        "top1_score": top1,
        "coverage_hits": len(set(ids) & relevant_ids) if relevant_ids else 0,
        "coverage_at_k": len(set(ids[:k]) & relevant_ids) if relevant_ids else 0,
    }
    if isinstance(net_diag, dict):
        d["latency_ms"] = net_diag.get("latency_ms")
        d["retries_used"] = net_diag.get("retries_used") or net_diag.get("retries") or 0
        d["error_class"] = net_diag.get("error_class")
    else:
        d["latency_ms"] = None
        d["retries_used"] = 0
        d["error_class"] = None
    return d


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = max(0, min(len(arr) - 1, int(math.ceil(p * (len(arr) - 1)))))
    return float(arr[idx])

def evaluate_dataset(
    dataset_path: str,
    k: int = 3,
    outdir: str = "runs",
    backend_url: Optional[str] = None,
    seed: Optional[int] = None,
    judgetype: str = "heuristic",
    llm_inference_fn=None,
    abort_failure_rate: Optional[float] = None,
) -> Dict[str, str]:
    if abort_failure_rate is None:
        try:
            env_thresh = os.getenv("EVAL_ABORT_FAILURE_RATE")
            abort_failure_rate = float(env_thresh) if env_thresh is not None else None
        except Exception:
            abort_failure_rate = None
    if abort_failure_rate is not None:
        if not (0.0 < float(abort_failure_rate) <= 1.0):
            raise ValueError("abort_failure_rate must be in (0,1] or None")

    try:
        from client.resilient_client import ResilientRAGClient
    except Exception:
        from client.resilient_client import ResilientRAGClient

    from evaluation.judges import HeuristicJudge, LLMJudge
    try:
        from typing import TypeAlias
        JSONObj: TypeAlias = Dict[str, Any]
    except Exception:
        JSONObj = Dict[str, Any]

    class _CallableProvider:
        def __init__(self, fn): self._fn = fn
        def infer(self, prompt: str, seed: Optional[int] = None) -> str:
            try: return self._fn(prompt, seed)
            except TypeError: return self._fn(prompt)

    provider = None
    if llm_inference_fn is not None:
        if callable(llm_inference_fn):
            provider = _CallableProvider(llm_inference_fn)
        elif hasattr(llm_inference_fn, "infer") and callable(getattr(llm_inference_fn, "infer")):
            provider = llm_inference_fn

    jt = (judgetype or "heuristic").lower()
    if jt == "llm":
        judge = LLMJudge(inference_fn=(provider.infer if provider else (lambda prompt, seed=None: '{"helpfulness":0,"correctness":0,"grounding":0,"notes":"no_provider"}')), name="llm-judge", version="v1")
    else:
        judge = HeuristicJudge(name="heuristic-judge", version="v1")

    JsonObj = Dict[str, Any]
    def load_jsonl(path: Union[str, Path]):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    def append_jsonl(path: Union[str, Path], record: JSONObj):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    def pct(values: List[float], p: float) -> float:
        if not values: return 0.0
        arr = sorted(values)
        idx = max(0, min(len(arr) - 1, int(math.ceil(p * (len(arr) - 1)))))
        return float(arr[idx])
    def _extract_retrieved_ids(resp: JSONObj) -> List[str]:
        items = (resp or {}).get("top_chunks") or []
        ids: List[str] = []
        for it in items:
            cid = it.get("id") if isinstance(it, dict) else getattr(it, "id", None)
            if isinstance(cid, str): ids.append(cid)
        return ids
    def _extract_scores(resp: JSONObj) -> List[float]:
        items = (resp or {}).get("top_chunks") or []
        scores: List[float] = []
        for it in items:
            s = it.get("score") if isinstance(it, dict) else getattr(it, "score", None)
            if isinstance(s, (int, float)): scores.append(float(s))
        return scores
    def _compute_diagnostics(resp: Optional[JSONObj], net_diag: Optional[JSONObj], relevant_ids: Set[str], k_val: int) -> JSONObj:
        ids = _extract_retrieved_ids(resp or {}) if resp else []
        scores = _extract_scores(resp or {}) if resp else []
        top1 = scores[0] if scores else None
        d: JSONObj = {
            "retrieved_count": len(ids),
            "scores_mean": (sum(scores) / len(scores)) if scores else None,
            "scores_std": None,
            "top1_score": top1,
            "coverage_hits": len(set(ids[:k_val]) & relevant_ids) if relevant_ids else 0,
            "coverage_at_k": len(set(ids[:k_val]) & relevant_ids) if relevant_ids else 0,
        }
        if isinstance(net_diag, dict):
            d["latency_ms"] = net_diag.get("latency_ms")
            d["retries_used"] = net_diag.get("retries_used") or net_diag.get("retries") or 0
            d["error_class"] = net_diag.get("error_class")
        else:
            d["latency_ms"] = None
            d["retries_used"] = 0
            d["error_class"] = None
        return d

    baseurl = backend_url or os.getenv("EVAL_BACKEND_URL", "http://localhost:8001")
    if seed is not None:
        random.seed(seed)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rundir = Path(outdir) / f"run_{run_id}"
    rundir.mkdir(parents=True, exist_ok=True)

    per_query_path = rundir / "queries.jsonl"
    summary_path = rundir / "summary.csv"
    summary_slices_path = rundir / "summary_slices.csv"
    diag_summary_path = rundir / "summary_diagnostics.csv"
    config_path = rundir / "config.json"
    judge_summary_path = rundir / "summary_judge.csv"
    judge_slices_path = rundir / "summary_slices_judge.csv"

    client = ResilientRAGClient(base_url=baseurl)

    n = 0
    attempted = 0
    failures = 0
    sump = sumr = summrr = 0.0
    sumhitk = sumndcgk = 0.0
    latencies: List[float] = []
    scores_all: List[float] = []
    retrieved_counts: List[int] = []
    coverage_hits_list: List[int] = []
    top1_scores: List[float] = []
    slice_stats = collections.defaultdict(lambda: {"n": 0, "sump": 0.0, "sumr": 0.0, "summrr": 0.0, "sumhitk": 0.0, "sumndcgk": 0.0})
    error_buckets = collections.Counter()
    retries_used_all: List[int] = []
    jsum = {"n": 0, "helpfulness": 0.0, "correctness": 0.0, "grounding": 0.0}
    jslice = collections.defaultdict(lambda: {"n": 0, "helpfulness": 0.0, "correctness": 0.0, "grounding": 0.0})

    for rec in load_jsonl(dataset_path):
        attempted += 1
        qid = rec.get("id") or f"row-{attempted}"
        query = rec.get("query", "")
        relevant_ids_list = rec.get("relevant_chunks") or rec.get("relevantchunks") or []
        relevant_ids: Set[str] = {x for x in relevant_ids_list if isinstance(x, str)}
        sl = rec.get("slice")
        slices: List[str] = [sl] if isinstance(sl, str) else ([s for s in sl if isinstance(s, str)] if isinstance(sl, list) else [])

        t0 = time.perf_counter()
        resp_json, net_diag = client.get_answer(query=query, k=k)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        error_class = None
        retries_used = 0
        if isinstance(net_diag, dict):
            error_class = net_diag.get("error_class")
            retries_used = net_diag.get("retries_used") or net_diag.get("retries") or 0
        retries_used_all.append(int(retries_used))

        if resp_json is None:
            failures += 1
            error_buckets[error_class or "unknown"] += 1
            per_query_diag = {
                "retrieved_count": 0,
                "scores_mean": None,
                "scores_std": None,
                "top1_score": None,
                "coverage_hits": 0,
                "coverage_at_k": 0,
                "latency_ms": round(dt_ms, 2),
                "retries_used": retries_used,
                "error_class": error_class,
            }
            jout = {
                "helpfulness": 0.0,
                "correctness": 0.0,
                "grounding": 0.0,
                "notes": "no_response",
                "judge_name": getattr(judge, "name", "unknown"),
                "judge_version": getattr(judge, "version", "v1"),
            }
            append_jsonl(per_query_path, {
                "id": qid,
                "query": query,
                "relevant_chunks": list(relevant_ids),
                "response": None,
                "metrics": {
                    f"precision@{k}": None,
                    f"recall@{k}": None,
                    f"hit@{k}": None,
                    f"ndcg@{k}": None,
                    "mrr": None,
                },
                "diagnostics": per_query_diag,
                "judge": jout,
                "backend_url": baseurl,
                "slices": slices,
                "network": net_diag,
                "timestamp": datetime.utcnow().isoformat(),
            })
            if abort_failure_rate is not None and attempted >= 10:
                observed = failures / float(attempted)
                if observed > abort_failure_rate:
                    break
            continue

        retrieved_ids = _extract_retrieved_ids(resp_json)
        scores = _extract_scores(resp_json)
        retrieved_counts.append(len(retrieved_ids))
        if scores:
            scores_all.extend(scores)
            top1_scores.append(scores[0])

        p = precision_at_k(retrieved_ids, relevant_ids, k) if relevant_ids else 0.0
        r = recall_at_k(retrieved_ids, relevant_ids, k) if relevant_ids else 0.0
        mrr = simple_mrr(retrieved_ids, relevant_ids) if relevant_ids else 0.0
        h = hitatk(retrieved_ids, relevant_ids, k) if relevant_ids else 0.0
        nd = ndcgatk(retrieved_ids, relevant_ids, k) if relevant_ids else 0.0

        sump += p; sumr += r; summrr += mrr; sumhitk += h; sumndcgk += nd
        for sl_key in slices:
            ss = slice_stats[sl_key]
            ss["n"] += 1
            ss["sump"] += p
            ss["sumr"] += r
            ss["summrr"] += mrr
            ss["sumhitk"] += h
            ss["sumndcgk"] += nd

        per_query_diag = _compute_diagnostics(resp_json, net_diag, relevant_ids, k)
        if isinstance(per_query_diag.get("coverage_hits"), int):
            coverage_hits_list.append(per_query_diag["coverage_hits"])

        answer_text = resp_json.get("answer", "") if isinstance(resp_json, dict) else ""
        items = (resp_json or {}).get("top_chunks") or []
        retrieved_context = [it for it in items if isinstance(it, dict)]

        gold_passages = None
        gold = rec.get("gold_answer")
        if isinstance(gold, str) and gold.strip():
            gold_passages = [gold]

        try:
            jout = judge.evaluate(
                query=query,
                answer=answer_text,
                retrieved_context=retrieved_context,
                gold_passages=gold_passages,
                metadata={"id": qid, "slices": slices},
            ) or {"helpfulness": 0.0, "correctness": 0.0, "grounding": 0.0}
        except Exception:
            jout = {"helpfulness": 0.0, "correctness": 0.0, "grounding": 0.0}
        jsum["n"] += 1
        jsum["helpfulness"] += float(jout.get("helpfulness", 0.0))
        jsum["correctness"] += float(jout.get("correctness", 0.0))
        jsum["grounding"] += float(jout.get("grounding", 0.0))
        for sl_key in slices:
            acc = jslice[sl_key]
            acc["n"] += 1
            acc["helpfulness"] += float(jout.get("helpfulness", 0.0))
            acc["correctness"] += float(jout.get("correctness", 0.0))
            acc["grounding"] += float(jout.get("grounding", 0.0))

        append_jsonl(per_query_path, {
            "id": qid,
            "query": query,
            "relevant_chunks": list(relevant_ids),
            "response": resp_json,
            "metrics": {
                f"precision@{k}": p,
                f"recall@{k}": r,
                f"hit@{k}": h,
                f"ndcg@{k}": nd,
                "mrr": mrr,
            },
            "diagnostics": per_query_diag,
            "judge": jout,
            "backend_url": baseurl,
            "slices": slices,
            "network": net_diag,
            "timestamp": datetime.utcnow().isoformat(),
        })
        n += 1

        if abort_failure_rate is not None and attempted >= 10:
            observed = failures / float(attempted)
            if observed > abort_failure_rate:
                break

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        denom = max(1, n)
        w.writerow([f"mean_precision@{k}", round(sump / denom, 4)])
        w.writerow([f"mean_recall@{k}", round(sumr / denom, 4)])
        w.writerow([f"mean_hit@{k}", round(sumhitk / denom, 4)])
        w.writerow([f"mean_ndcg@{k}", round(sumndcgk / denom, 4)])
        w.writerow(["mean_mrr", round(summrr / denom, 4)])

    if slice_stats:
        with open(summary_slices_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["slice", "n", f"mean_precision@{k}", f"mean_recall@{k}", f"mean_hit@{k}", f"mean_ndcg@{k}", "mean_mrr"])
            for sl, ss in sorted(slice_stats.items()):
                dn = max(1, int(ss["n"]))
                w.writerow([sl, int(ss["n"]), round(ss["sump"] / dn, 4), round(ss["sumr"] / dn, 4), round(ss["sumhitk"] / dn, 4), round(ss["sumndcgk"] / dn, 4), round(ss["summrr"] / dn, 4)])

    # Diagnostics and failures
    with open(diag_summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        if attempted > 0:
            w.writerow(["latency_p50_ms", round(pct(latencies, 0.50), 2)])
            w.writerow(["latency_p95_ms", round(pct(latencies, 0.95), 2)])
            w.writerow(["latency_p99_ms", round(pct(latencies, 0.99), 2)])
        total_fail = sum(error_buckets.values())
        observed_failure_rate = (total_fail / float(attempted)) if attempted else 0.0
        w.writerow(["error_rate", round(observed_failure_rate, 6)])
        for cls, cnt in sorted(error_buckets.items()):
            w.writerow([f"errors::{cls}", cnt])
        if retries_used_all:
            w.writerow(["retries_mean", round(sum(retries_used_all) / len(retries_used_all), 3)])
            w.writerow(["retries_p95", round(pct(retries_used_all, 0.95), 3)])
        if retrieved_counts:
            w.writerow(["retrieved_count_mean", round(sum(retrieved_counts) / len(retrieved_counts), 3)])
            w.writerow(["retrieved_count_min", min(retrieved_counts)])
            w.writerow(["retrieved_count_max", max(retrieved_counts)])
        if scores_all:
            w.writerow(["scores_mean_overall", round(sum(scores_all) / len(scores_all), 4)])
            w.writerow(["scores_min", round(min(scores_all), 4)])
            w.writerow(["scores_max", round(max(scores_all), 4)])
        if top1_scores:
            w.writerow(["top1_scores_mean", round(sum(top1_scores) / len(top1_scores), 4)])
        if coverage_hits_list:
            w.writerow(["coverage_hits_mean", round(sum(coverage_hits_list) / len(coverage_hits_list), 3)])

    # Judge summaries
    with open(judge_summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "judge", "helpfulness_mean", "correctness_mean", "grounding_mean"])
        if jsum["n"] > 0:
            dn = float(jsum["n"])
            w.writerow([int(jsum["n"]), getattr(judge, "name", "unknown"),
                        round(jsum["helpfulness"] / dn, 4),
                        round(jsum["correctness"] / dn, 4),
                        round(jsum["grounding"] / dn, 4)])
        else:
            w.writerow([0, getattr(judge, "name", "unknown"), 0, 0, 0])

    with open(judge_slices_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slice", "n", "helpfulness_mean", "correctness_mean", "grounding_mean"])
        for sl, acc in sorted(jslice.items()):
            dn = max(1, int(acc["n"]))
            w.writerow([sl, int(acc["n"]),
                        round(acc["helpfulness"] / dn, 4),
                        round(acc["correctness"] / dn, 4),
                        round(acc["grounding"] / dn, 4)])

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_path": dataset_path,
                "k": k,
                "backend_url": baseurl,
                "judge": judgetype,
                "random_seed": seed,
                "code_version": os.getenv("EVAL_CODE_SHA", "PLACEHOLDER_VERSION"),
                "created_utc": datetime.utcnow().isoformat(),
                "abort_failure_rate": abort_failure_rate,
                "attempted": attempted,
                "failures": failures,
                "observed_failure_rate": round((failures / float(attempted)) if attempted else 0.0, 6),
                "aborted_early": (abort_failure_rate is not None and attempted > 0 and (failures / float(attempted)) > abort_failure_rate),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "summary_judge_csv": str(judge_summary_path),
        "summary_slices_judge_csv": str(judge_slices_path),
        "rundir": str(rundir),
        "queries_log": str(per_query_path),
        "summary_csv": str(summary_path),
        "summary_diagnostics_csv": str(diag_summary_path),
        "summary_slices_csv": str(summary_slices_path),
    }




if __name__ == "__main__":
    default_dataset = os.getenv("EVAL_DATASET_PATH", "rag_eval/data/sample_smoke.jsonl")
    evaluate_dataset(dataset_path=default_dataset, k=3, outdir="runs/")
