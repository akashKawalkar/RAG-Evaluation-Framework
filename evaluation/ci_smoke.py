import os, sys, json, csv
from pathlib import Path
from evaluation.batch_eval import evaluate_dataset

def load_summary(path: Path):
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    mp = {}
    if rows and rows[0] and rows[0][0].lower() == "metric":
        for r in rows[1:]:
            if len(r) >= 2:
                mp[r[0]] = r[1]
    return mp

def as_float(mp, key, default=0.0):
    try:
        return float(mp.get(key, default))
    except Exception:
        return float(default)

def main():
    dataset = os.environ.get("CI_SMOKE_DATASET", "rag_eval/data/smoke_ci.jsonl")
    outdir = os.environ.get("EVAL_RUNS_DIR", "runs")
    backend = os.environ.get("EVAL_BACKEND_URL", "http://localhost:8001")
    k = int(os.environ.get("CI_SMOKE_K", "3"))

    max_error_rate = float(os.environ.get("CI_MAX_ERROR_RATE", "0.20"))
    min_precision = float(os.environ.get("CI_MIN_PRECISION", "0.10"))
    min_recall = float(os.environ.get("CI_MIN_RECALL", "0.10"))

    res = evaluate_dataset(
        dataset_path=dataset,
        k=k,
        outdir=outdir,
        backend_url=backend,
        judgetype="heuristic",
        llm_inference_fn=None,
    )
    rundir = Path(res["rundir"]) if "rundir" in res else Path(res.get("run_dir", ""))

    # Load summaries
    diag = load_summary(rundir / "summary_diagnostics.csv")
    summ = load_summary(rundir / "summary.csv")

    error_rate = as_float(diag, "error_rate", 0.0)

    precision_key = f"mean_precision@{k}"
    recall_key = f"mean_recall@{k}"

    precision = as_float(summ, precision_key, 0.0)
    recall = as_float(summ, recall_key, 0.0)

    report = {
        "rundir": str(rundir),
        "error_rate": error_rate,
        "precision_key": precision_key,
        "precision": precision,
        "recall_key": recall_key,
        "recall": recall,
        "gates": {
            "max_error_rate": max_error_rate,
            "min_precision": min_precision,
            "min_recall": min_recall,
        },
    }
    print(json.dumps(report, indent=2))

    failures = []
    if error_rate > max_error_rate:
        failures.append(f"error_rate {error_rate:.4f} > {max_error_rate:.4f}")
    if precision < min_precision:
        failures.append(f"{precision_key} {precision:.4f} < {min_precision:.4f}")
    if recall < min_recall:
        failures.append(f"{recall_key} {recall:.4f} < {min_recall:.4f}")

    if failures:
        print("CI SMOKE FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)

    print("CI SMOKE PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()
