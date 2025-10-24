
import os
import sys
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from rag_eval.evaluation.batch_eval import evaluate_dataset
except ModuleNotFoundError:
    try:
        from evaluation.batch_eval import evaluate_dataset 
    except ModuleNotFoundError:
        import sys
        from pathlib import Path as _Path
        _PROJECT_ROOT = _Path(__file__).resolve().parents[2]
        if str(_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(_PROJECT_ROOT))
        from rag_eval.evaluation.batch_eval import evaluate_dataset


st.set_page_config(page_title="Batch Evaluation", layout="wide")
st.title("Batch Evaluation")


def read_csv(path: Path) -> List[List[str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.reader(f))


def read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
            if limit and i >= limit:
                break
    return out


def metrics_map(rows: List[List[str]]) -> Dict[str, Any]:
    if not rows:
        return {}
    if len(rows) >= 1 and rows[0] and rows[0][0].lower() == "metric":
        rows = rows[1:]
    out: Dict[str, Any] = {}
    for r in rows:
        if len(r) < 2:
            continue
        out[str(r[0])] = r[1]
    return out


def hist_values_from_queries(queries: List[Dict[str, Any]]) -> Tuple[List[float], List[int], List[float], List[int]]:
    latencies: List[float] = []
    retrieved_counts: List[int] = []
    top1_scores: List[float] = []
    coverage_hits: List[int] = []

    for rec in queries:
        diag = rec.get("diagnostics") or {}
        lat = diag.get("latency_ms")
        if isinstance(lat, (int, float)):
            latencies.append(float(lat))
        rc = diag.get("retrieved_count")
        if isinstance(rc, int):
            retrieved_counts.append(rc)
        t1 = diag.get("top1_score")
        if isinstance(t1, (int, float)):
            top1_scores.append(float(t1))
        cov = diag.get("coverage_hits")
        if isinstance(cov, int):
            coverage_hits.append(cov)
    return latencies, retrieved_counts, top1_scores, coverage_hits


def worst5_by_precision(queries: List[Dict[str, Any]], k: int) -> List[Tuple[float, float, Dict[str, Any]]]:
    rows: List[Tuple[float, float, Dict[str, Any]]] = []
    km = f"precision@{k}"
    for rec in queries:
        m = rec.get("metrics") or {}
        p = m.get(km)
        diag = rec.get("diagnostics") or {}
        lat = diag.get("latency_ms")
        if isinstance(p, (int, float)):
            rows.append((float(p), float(lat) if isinstance(lat, (int, float)) else 0.0, rec))
    rows.sort(key=lambda x: (x[0], -x[1])) 
    return rows[:5]


def extract_judge_score(rec: Dict[str, Any], key: str = "correctness") -> Optional[float]:
    """
    Returns judge score in [0,1] if available: rec['judge'][key] or rec['judge_outputs'][key].
    Falls back to None if missing or not a number.
    """
    judge = rec.get("judge") or rec.get("judge_outputs") or {}
    val = judge.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    out = judge.get("output")
    if isinstance(out, dict):
        v = out.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    if isinstance(out, str):
        try:
            parsed = json.loads(out)
            v2 = parsed.get(key)
            if isinstance(v2, (int, float)):
                return float(v2)
        except Exception:
            pass
    return None


def worst5_by_judge_correctness(queries: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
    rows: List[Tuple[float, Dict[str, Any]]] = []
    for rec in queries:
        score = extract_judge_score(rec, key="correctness")
        if isinstance(score, float):
            rows.append((score, rec))
    rows.sort(key=lambda x: x[0])
    return rows[:5]


def worst5_by_judge_correctness_proxy(queries: List[Dict[str, Any]], k: int) -> List[Tuple[float, Dict[str, Any]]]:
    rows: List[Tuple[float, Dict[str, Any]]] = []
    km = f"precision@{k}"
    for rec in queries:
        m = rec.get("metrics") or {}
        p = m.get(km)
        if isinstance(p, (int, float)):
            rows.append((float(p), rec))
    rows.sort(key=lambda x: x[0])
    return rows[:5]


with st.sidebar:
    st.header("Run configuration")
    default_data_dir = Path("data")
    options = [str(p) for p in sorted(default_data_dir.glob("*.jsonl"))] if default_data_dir.exists() else []
    dataset_path = st.selectbox("Dataset (.jsonl)", options=["--"] + options, index=1 if options else 0,
                                help="Select a dataset file from rag_eval/data, or type a path below.")
    dataset_path_text = st.text_input("Or type dataset path",
                                      value=dataset_path if dataset_path and dataset_path != "--" else os.getenv("EVAL_DATASET_PATH", ""),
                                      help="Absolute or relative path to a JSONL file with id, query, relevant_chunks, slice?, metadata?.")
    dataset_path = dataset_path_text or (dataset_path if dataset_path != "--" else "")

    k = st.number_input("k for P@k/R@k", min_value=1, max_value=20, value=3, step=1)
    judge_type = st.selectbox("Judge", options=["heuristic", "llm"], index=0)

    backend_url = st.text_input(
        "Backend URL",
        value=os.getenv("EVAL_BACKEND_URL", "http://localhost:8001"),
        help="Evaluations use the resilient client; flip this to switch backends."
    )

    outdir = st.text_input("Runs output dir", value="runs")

    runbutton = st.button("Run evaluation")


if runbutton:
    if not dataset_path:
        st.error("Please select or enter a dataset path.")
        st.stop()
    with st.spinner("Evaluating dataset..."):
        res = evaluate_dataset(
            dataset_path=dataset_path,
            k=int(k),
            outdir=outdir,
            backend_url=backend_url,
            judgetype=judge_type,
            llm_inference_fn=None,
            seed=None,
        )
    st.success("Run complete.")
    st.session_state["last_run"] = res


st.divider()
st.header("Run results")

default_run = None
if "last_run" in st.session_state:
    default_run = st.session_state["last_run"].get("run_dir")

rundir_text = st.text_input("Run directory", value=default_run or "",
                            help="Path to a run/YYYYMMDD_HHMMSS directory.")
if not rundir_text:
    st.info("Start a run or paste a run directory path to visualize results.")
    st.stop()

rundir = Path(rundir_text)
if not rundir.exists():
    st.error("Run directory not found.")
    st.stop()

summary_rows = read_csv(rundir / "summary.csv")
diag_rows = read_csv(rundir / "summary_diagnostics.csv")
slice_rows = read_csv(rundir / "summary_slices.csv")
queries = read_jsonl(rundir / "queries.jsonl", limit=None)
cfg_path = rundir / "config.json"

judge_summary_rows = read_csv(rundir / "summary_judge.csv")
judge_slices_rows = read_csv(rundir / "summary_slices_judge.csv")

st.subheader("Config")
if cfg_path.exists():
    st.code((rundir / "config.json").read_text(encoding="utf-8"), language="json")
else:
    st.write("config.json not found in this run.")

st.subheader("Summary metrics")
if summary_rows:
    st.table(summary_rows)
else:
    st.write("summary.csv not found or empty.")

st.subheader("Diagnostics")
if diag_rows:
    st.table(diag_rows)
else:
    st.write("summary_diagnostics.csv not found or empty.")
diagmap = metrics_map(diag_rows)
cols = st.columns(4)
cols[0].metric("p95 latency (ms)", diagmap.get("latency_p95_ms", "-"))
cols[1].metric("p99 latency (ms)", diagmap.get("latency_p99_ms", "-"))
cols[2].metric("error rate", diagmap.get("error_rate", "-"))
cols[3].metric("retries mean", diagmap.get("retries_mean", "-"))

fail_rows = [(k.replace("errors::", ""), v) for k, v in diagmap.items() if str(k).startswith("errors::")]
if fail_rows:
    st.subheader("Failures")
    st.table([("bucket", "count")] + fail_rows)
retries_p95 = diagmap.get("retries_p95", None)
if retries_p95 is not None:
    cols_r = st.columns(2)
    cols_r[0].metric("retries mean", diagmap.get("retries_mean", "-"))
    cols_r[1].metric("retries p95", retries_p95)

st.subheader("Failure containment")
observed_rate = diagmap.get("error_rate", None)
abort_info = {}
try:
    if cfg_path.exists():
        abort_info = json.loads(cfg_path.read_text(encoding="utf-8"))
except Exception:
    abort_info = {}

cols2 = st.columns(3)
cols2[0].metric("observed error_rate", observed_rate if observed_rate is not None else "-")
cols2[1].metric("abort threshold", str(abort_info.get("abort_failure_rate", "None")))
cols2[2].metric("aborted early", str(abort_info.get("aborted_early", False)))

try:
    p95_budget = float(os.getenv("EVAL_LATENCY_P95_BUDGET_MS", "0") or "0")
except Exception:
    p95_budget = 0.0
try:
    p99_budget = float(os.getenv("EVAL_LATENCY_P99_BUDGET_MS", "0") or "0")
except Exception:
    p99_budget = 0.0

lat_p95 = float(diagmap.get("latency_p95_ms", "0") or "0")
lat_p99 = float(diagmap.get("latency_p99_ms", "0") or "0")

if p95_budget > 0 and lat_p95 > p95_budget:
    st.warning(f"Latency p95 ({lat_p95:.1f} ms) exceeds budget ({p95_budget:.1f} ms).")
if p99_budget > 0 and lat_p99 > p99_budget:
    st.error(f"Latency p99 ({lat_p99:.1f} ms) exceeds budget ({p99_budget:.1f} ms).")   
try:
    budget_p99 = float(os.getenv("EVAL_LATENCY_P99_BUDGET_MS", "0") or "0")
except Exception:
    budget_p99 = 0.0
p99 = float(diagmap.get("latency_p99_ms", "0") or "0")
if budget_p99 > 0 and p99 > budget_p99:
    st.warning(f"Latency p99 ({p99:.1f} ms) exceeds budget ({budget_p99:.1f} ms). Consider tuning retries/timeouts or backend.")

st.subheader("Per-slice summary")
if slice_rows:
    st.table(slice_rows)
else:
    st.write("summary_slices.csv not found or empty.")

st.subheader("Worst 5 by precision")
if queries:
    worst_prec = worst5_by_precision(queries, int(k))
    for p, lat, rec in worst_prec:
        st.markdown(f"- Query: {rec.get('query','')}")
        st.markdown(f"- precision@{int(k)}: {p:.3f}")
        st.markdown(f"- latency_ms: {lat:.2f}")
        resp = rec.get("response") or {}
        tops = resp.get("top_chunks") or []
        if tops:
            st.markdown("- top_chunks:")
            for i, t in enumerate(tops[:3], 1):
                chunk = t.get("chunk", "")
                show = chunk[:140] + " ..." if len(chunk) > 140 else chunk
                st.markdown(f"  {i}. {show}")

st.divider()
st.header("Model-graded judge")

colA, colB = st.columns(2)
with colA:
    st.subheader("Judge summary")
    if judge_summary_rows:
        st.table(judge_summary_rows)
    else:
        st.write("summary_judge.csv not found. Run with a judge to populate.")

with colB:
    st.subheader("Judge per-slice summary")
    if judge_slices_rows:
        st.table(judge_slices_rows)
    else:
        st.write("summary_slices_judge.csv not found.")

st.subheader("Worst 5 by judge correctness")
have_real_judge = False
for rec in queries:
    if extract_judge_score(rec, key="correctness") is not None:
        have_real_judge = True
        break

if queries:
    if have_real_judge:
        worst_judge = worst5_by_judge_correctness(queries)
        for score, rec in worst_judge:
            st.markdown(f"- Query: {rec.get('query','')}")
            st.markdown(f"- judge_correctness: {score:.3f}")
            resp = rec.get("response") or {}
            tops = resp.get("top_chunks") or []
            if tops:
                st.markdown("- top_chunks:")
                for i, t in enumerate(tops[:2], 1):
                    chunk = t.get("chunk", "")
                    show = chunk[:140] + " ..." if len(chunk) > 140 else chunk
                    st.markdown(f"  {i}. {show}")
    else:
        st.caption("Judge outputs not found in per-query logs; falling back to precision@k proxy.")
        worst_proxy = worst5_by_judge_correctness_proxy(queries, int(k))
        for score, rec in worst_proxy:
            st.markdown(f"- Query: {rec.get('query','')}")
            st.markdown(f"- proxy_correctness (precision@{int(k)}): {score:.3f}")
            resp = rec.get("response") or {}
            tops = resp.get("top_chunks") or []
            if tops:
                st.markdown("- top_chunks:")
                for i, t in enumerate(tops[:2], 1):
                    chunk = t.get("chunk", "")
                    show = chunk[:140] + " ..." if len(chunk) > 140 else chunk
                    st.markdown(f"  {i}. {show}")

st.subheader("Latency histogram")
latencies, retrieved_counts, top1_scores, coverage_hits = hist_values_from_queries(queries)
if latencies:
    st.bar_chart(latencies)
else:
    st.write("No latency data in queries.jsonl.")

st.subheader("Retrieved count histogram")
if retrieved_counts:
    st.bar_chart(retrieved_counts)
else:
    st.write("No retrieved_count data in queries.jsonl.")

st.subheader("Top-1 score histogram")
if top1_scores:
    st.bar_chart(top1_scores)
else:
    st.write("No top1_score data in queries.jsonl.")

st.subheader("Coverage hits histogram")
if coverage_hits:
    st.bar_chart(coverage_hits)
else:
    st.write("No coverage_hits data in queries.jsonl.")
