import streamlit as st
import csv, json
from pathlib import Path
import os, sys
from typing import List, Dict, Any, Optional
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Runs", layout="wide")
st.title("Runs")

runs_root = st.text_input("Runs directory", value="runs")
root = Path(runs_root)
if not root.exists():
    st.warning("Runs directory not found")
    st.stop()

runs = sorted([p for p in root.glob("run_*") if p.is_dir()], reverse=True)
labels = [p.name for p in runs]
sel = st.selectbox("Select a run", options=["--"] + labels)

def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.reader(f))

def read_jsonl(path: Path, limit: int = None):
    items = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if limit and i >= limit:
                break
    return items

if sel != "--":
    run_dir = root / sel
    st.subheader("Config")
    cfg = run_dir / "config.json"
    if cfg.exists():
        st.code(cfg.read_text(encoding="utf-8"))

    st.subheader("Summary")
    st.table(read_csv(run_dir / "summary.csv"))

    st.subheader("Diagnostics")
    st.table(read_csv(run_dir / "summary_diagnostics.csv"))

    st.subheader("Per-slice summary")
    st.table(read_csv(run_dir / "summary_slices.csv"))

    st.subheader("Sample per-query records")
    qs = read_jsonl(run_dir / "queries.jsonl", limit=20)
    for rec in qs:
        st.json(rec)

st.divider()
st.subheader("Compare two runs (quick view)")


def table_to_dict(table):
    if not table or len(table) < 2:
        return {}
    header = table
    out = {}
    for row in table[1:]:
        if len(row) != len(header):
            continue
        key = header
        if key == "n":
            for i, h in enumerate(header):
                out[h] = row[i]
            break
        else:
            out[row] = row[1:]
    return out


def list_runs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates
def summary_to_map(table: List[List[str]]) -> Dict[str, str]:
    if not table:
        return {}
    header = table[0]
    if header and header[0].lower() == "metric":
        pairs = table[1:]
        out: Dict[str, str] = {}
        for row in pairs:
            if len(row) >= 2:
                out[str(row[0])] = row[1]
        return out
    out: Dict[str, str] = {}
    for row in table:
        if len(row) >= 2:
            out[str(row[0])] = row[1]
    return out
def slices_csv_to_map(table: List[List[str]]) -> Dict[str, Dict[str, str]]:
    """
    Turns summary_slices.csv into {slice_name: {col: val, ...}} preserving strings for display and diffing.
    Expects header like: ["slice", "n", f"mean_precision@{k}", ...]
    """
    out: Dict[str, Dict[str, str]] = {}
    if not table or len(table) < 2:
        return out
    header = table[0]
    for row in table[1:]:
        if not row or len(row) != len(header):
            continue
        row_map = {header[i]: row[i] for i in range(len(header))}
        sname = row_map.get("slice") or row[0]
        if sname is None:
            continue
        out[str(sname)] = row_map
    return out
def judge_summary_to_map(table: List[List[str]]) -> Dict[str, str]:
    if not table or len(table) < 2:
        return {}
    header = table[0]
    rows = table[1:]
    out: Dict[str, str] = {}
    if len(rows) >= 1:
        last = rows[-1]
        for i, h in enumerate(header):
            if i < len(last):
                out[str(h)] = last[i]
    return out
def judge_slices_to_map(table: List[List[str]]) -> Dict[str, Dict[str, str]]:
    if not table or len(table) < 2:
        return {}
    header = table[0]
    out: Dict[str, Dict[str, str]] = {}
    for row in table[1:]:
        if len(row) != len(header):
            continue
        row_map = {header[i]: row[i] for i in range(len(header))}
        sname = row_map.get("slice") or row[0]
        out[str(sname)] = row_map
    return out
with st.sidebar:
    st.header("Runs root")
    runs_root = st.text_input("Directory", value=os.getenv("EVAL_RUNS_DIR", "runs"))
    st.caption("Change the directory if runs are stored elsewhere.")


root = Path(runs_root)
if not root.exists():
    st.warning("Runs directory not found.")
    st.stop()

runs = list_runs(root)
if not runs:
    st.info("No runs found under the selected directory.")
    st.stop()

labels = [p.name for p in runs]
sel = st.selectbox("Select a run", options=["--"] + labels, index=1 if labels else 0)

if sel == "--":
    st.info("Pick a run to see details.")
else:
    rundir = root / sel

    st.subheader("Config")
    cfg = rundir / "config.json"
    if cfg.exists():
        st.code(cfg.read_text(encoding="utf-8"), language="json")
    else:
        st.write("config.json not found.")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Summary")
        summary_rows = read_csv(rundir / "summary.csv")
        st.table(summary_rows if summary_rows else [["-", "-"]])
        if (rundir / "summary.csv").exists():
            st.download_button("Download summary.csv", data=(rundir / "summary.csv").read_bytes(), file_name="summary.csv", mime="text/csv")

        st.subheader("Judge summary")
        judge_rows = read_csv(rundir / "summary_judge.csv")
        st.table(judge_rows if judge_rows else [["-", "-"]])
        if (rundir / "summary_judge.csv").exists():
            st.download_button("Download summary_judge.csv", data=(rundir / "summary_judge.csv").read_bytes(), file_name="summary_judge.csv", mime="text/csv")

    with colB:
        st.subheader("Diagnostics")
        diag_rows = read_csv(rundir / "summary_diagnostics.csv")
        st.table(diag_rows if diag_rows else [["-", "-"]])
        if (rundir / "summary_diagnostics.csv").exists():
            st.download_button("Download summary_diagnostics.csv", data=(rundir / "summary_diagnostics.csv").read_bytes(), file_name="summary_diagnostics.csv", mime="text/csv")

        st.subheader("Per-slice summary")
        slice_rows = read_csv(rundir / "summary_slices.csv")
        st.table(slice_rows if slice_rows else [["-", "-"]])
        if (rundir / "summary_slices.csv").exists():
            st.download_button("Download summary_slices.csv", data=(rundir / "summary_slices.csv").read_bytes(), file_name="summary_slices.csv", mime="text/csv")

        st.subheader("Judge per-slice")
        judge_slice_rows = read_csv(rundir / "summary_slices_judge.csv")
        st.table(judge_slice_rows if judge_slice_rows else [["-", "-"]])
        if (rundir / "summary_slices_judge.csv").exists():
            st.download_button("Download summary_slices_judge.csv", data=(rundir / "summary_slices_judge.csv").read_bytes(), file_name="summary_slices_judge.csv", mime="text/csv")

    st.subheader("Sample per-query records")
    sample_n = st.slider("Sample size", min_value=5, max_value=100, value=20, step=5)
    sample = read_jsonl(rundir / "queries.jsonl", limit=sample_n)
    if sample:
        for rec in sample:
            with st.expander(f"{rec.get('id', '')}: {rec.get('query', '')[:80]}"):
                st.json(rec)
        if (rundir / "queries.jsonl").exists():
            st.download_button("Download queries.jsonl", data=(rundir / "queries.jsonl").read_bytes(), file_name="queries.jsonl", mime="application/json")
    else:
        st.write("queries.jsonl not found or empty.")

st.divider()
st.subheader("Compare two runs")

col1, col2 = st.columns(2)
with col1:
    selA = st.selectbox("Run A", options=["--"] + labels, key="runA")
with col2:
    selB = st.selectbox("Run B", options=["--"] + labels, key="runB")

def make_diff_table(a_map: Dict[str, str], b_map: Dict[str, str], wanted: Optional[List[str]] = None) -> List[List[str]]:
    keys = sorted((set(a_map.keys()) & set(b_map.keys())))
    if wanted:
        keys = [k for k in keys if any(w in k for w in wanted)]
    rows = [["metric", "A", "B", "delta"]]
    for k in keys:
        av, bv = a_map.get(k, ""), b_map.get(k, "")
        try:
            fa = float(av)
            fb = float(bv)
            dv = fb - fa
            rows.append([k, f"{fa:.6f}", f"{fb:.6f}", f"{dv:+.6f}"])
        except Exception:
            rows.append([k, av, bv, "n/a"])
    return rows

def make_slice_diff_table(a_slices: Dict[str, Dict[str, str]], b_slices: Dict[str, Dict[str, str]],
                          metric_cols: List[str]) -> List[List[str]]:
    slices = sorted(set(a_slices.keys()) | set(b_slices.keys()))
    header = ["slice"]
    for m in metric_cols:
        header += [f"A:{m}", f"B:{m}", f"Δ:{m}"]
    rows: List[List[str]] = [header]
    for s in slices:
        arow = a_slices.get(s, {})
        brow = b_slices.get(s, {})
        row = [s]
        for m in metric_cols:
            av = arow.get(m, "")
            bv = brow.get(m, "")
            try:
                fa = float(av)
                fb = float(bv)
                dv = fb - fa
                row += [f"{fa:.6f}", f"{fb:.6f}", f"{dv:+.6f}"]
            except Exception:
                row += [av, bv, "n/a"]
        rows.append(row)
    return rows

if selA != "--" and selB != "--" and selA != selB:
    adir = root / selA
    bdir = root / selB

    # Overall summary
    asum = read_csv(adir / "summary.csv")
    bsum = read_csv(bdir / "summary.csv")
    a_map = summary_to_map(asum)
    b_map = summary_to_map(bsum)

    st.markdown("### Overall metrics")
    wanted_overall = ["precision@", "recall@", "mean_mrr", "hit@", "ndcg@"]
    diff_rows = make_diff_table(a_map, b_map, wanted=wanted_overall)
    st.table(diff_rows)
    diff_csv = "\n".join([",".join(r) for r in diff_rows]).encode("utf-8")
    st.download_button("Download overall diff CSV", data=diff_csv, file_name="overall_diff.csv", mime="text/csv")

    aj = read_csv(adir / "summary_judge.csv")
    bj = read_csv(bdir / "summary_judge.csv")
    aj_map = summary_to_map(aj)
    bj_map = summary_to_map(bj)
    st.markdown("### Judge metrics (overall)")
    judge_wanted = ["judge_correctness", "judge_grounding", "judge_helpfulness"]
    judge_diff_rows = make_diff_table(aj_map, bj_map, wanted=judge_wanted)
    st.table(judge_diff_rows)
    judge_diff_csv = "\n".join([",".join(r) for r in judge_diff_rows]).encode("utf-8")
    st.download_button("Download judge overall diff CSV", data=judge_diff_csv, file_name="judge_overall_diff.csv", mime="text/csv")

    aslices = read_csv(adir / "summary_slices.csv")
    bslices = read_csv(bdir / "summary_slices.csv")
    a_slices_map = slices_csv_to_map(aslices)
    b_slices_map = slices_csv_to_map(bslices)

    metric_cols = []
    if aslices and bslices and len(aslices[0]) == len(bslices[0]):
        a_cols = aslices[0][:]
        b_cols = bslices[0][:]
        try:
            a_cols.remove("slice")
        except Exception:
            pass
        try:
            b_cols.remove("slice")
        except Exception:
            pass
        metric_cols = [c for c in a_cols if c in b_cols]
    else:
        metric_cols = ["n", "mean_precision@3", "mean_recall@3", "mean_hit@3", "mean_ndcg@3", "mean_mrr"]

    st.markdown("### Per-slice metrics (retrieval)")
    slice_diff_rows = make_slice_diff_table(a_slices_map, b_slices_map, metric_cols=metric_cols)
    st.table(slice_diff_rows)
    slice_diff_csv = "\n".join([",".join(r) for r in slice_diff_rows]).encode("utf-8")
    st.download_button("Download per-slice diff CSV", data=slice_diff_csv, file_name="per_slice_diff.csv", mime="text/csv")

    aj_slices = read_csv(adir / "summary_slices_judge.csv")
    bj_slices = read_csv(bdir / "summary_slices_judge.csv")
    aj_map_s = judge_slices_to_map(aj_slices)
    bj_map_s = judge_slices_to_map(bj_slices)
    judge_slice_cols = []
    if aj_slices and bj_slices and len(aj_slices[0]) == len(bj_slices[0]):
        jcols = aj_slices[0][:]
        try:
            jcols.remove("slice")
        except Exception:
            pass
        judge_slice_cols = [c for c in jcols if c in bj_slices[0]]
    else:
        judge_slice_cols = ["n", "judge_helpfulness", "judge_correctness", "judge_grounding"]

    norm_map = {
        "helpfulness_mean": "judge_helpfulness",
        "correctness_mean": "judge_correctness",
        "grounding_mean": "judge_grounding",
    }
    def normalize_cols(cols: List[str]) -> List[str]:
        out = []
        for c in cols:
            out.append(norm_map.get(c, c))
        return out

    judge_slice_cols_norm = normalize_cols(judge_slice_cols)

    def normalize_map_cols(m: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        for s, row in m.items():
            newrow = dict(row)
            for src, dst in norm_map.items():
                if src in newrow and dst not in newrow:
                    newrow[dst] = newrow[src]
            out[s] = newrow
        return out

    aj_map_s_norm = normalize_map_cols(aj_map_s)
    bj_map_s_norm = normalize_map_cols(bj_map_s)

    st.markdown("### Per-slice metrics (judge)")
    judge_slice_diff_rows = make_slice_diff_table(aj_map_s_norm, bj_map_s_norm, metric_cols=judge_slice_cols_norm)
    st.table(judge_slice_diff_rows)
    judge_slice_diff_csv = "\n".join([",".join(r) for r in judge_slice_diff_rows]).encode("utf-8")
    st.download_button("Download per-slice judge diff CSV", data=judge_slice_diff_csv, file_name="per_slice_judge_diff.csv", mime="text/csv")
else:
    st.caption("Pick two different runs to compare.")