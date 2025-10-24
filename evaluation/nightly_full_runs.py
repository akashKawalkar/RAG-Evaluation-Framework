import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from evaluation.batch_eval import evaluate_dataset
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rag_eval.evaluation.batch_eval import evaluate_dataset 


def list_datasets(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        pth = Path(p)
        if pth.is_file() and pth.suffix.lower() == ".jsonl":
            out.append(str(pth))
        elif pth.is_dir():
            out.extend([str(x) for x in sorted(pth.glob("*.jsonl"))])
    return out


def dump_manifest(run_records: List[Dict[str, Any]], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": run_records,
    }
    path = outdir / "nightly_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def optional_sync_to_object_storage(local_runs: List[Path], dest_dir: Optional[str]) -> Optional[List[str]]:
    """
    Very simple file copy to a local or mounted path.
    Replace this with an S3/GCS client if desired (e.g., boto3, gsutil).
    """
    if not dest_dir:
        return None
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for r in local_runs:
        target = dest / r.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(r, target)
        copied.append(str(target))
    return copied


def main():

    datasets_arg = os.getenv("NIGHTLY_DATASETS", "rag_eval/data")
    datasets_paths = [s.strip() for s in datasets_arg.split(",") if s.strip()]
    datasets = list_datasets(datasets_paths)
    if not datasets:
        print("No datasets found for nightly run. Set NIGHTLY_DATASETS to files/dirs with .jsonl datasets.", file=sys.stderr)
        sys.exit(1)

    outdir = os.getenv("EVAL_RUNS_DIR", "runs")
    backend = os.getenv("EVAL_BACKEND_URL", "http://localhost:8001")
    k = int(os.getenv("NIGHTLY_K", "3"))
    seed = None  

    judge_modes = [s.strip().lower() for s in os.getenv("NIGHTLY_JUDGES", "heuristic").split(",") if s.strip()]
    judge_modes = [jm for jm in judge_modes if jm in ("heuristic", "llm")]

    object_store_dir = os.getenv("NIGHTLY_OBJECT_STORE_DIR", "")

    print(f"Nightly datasets: {datasets}")
    print(f"Judges: {judge_modes}")
    print(f"Backend: {backend}")
    print(f"Output dir: {outdir}")
    print(f"Object store dir: {object_store_dir or '(disabled)'}")

    all_run_records: List[Dict[str, Any]] = []
    created_run_dirs: List[Path] = []

    for ds in datasets:
        for jm in judge_modes:
            print(f"Running dataset={ds} judge={jm} k={k}")
            res = evaluate_dataset(
                dataset_path=ds,
                k=k,
                outdir=outdir,
                backend_url=backend,
                judgetype=jm,
                llm_inference_fn=None, 
            )
            record = {
                "dataset_path": ds,
                "judge": jm,
                "k": k,
                "backend_url": backend,
                "run_dir": res.get("run_dir") or res.get("rundir") or "",
                "summary_csv": res.get("summary_csv", ""),
                "summary_diagnostics_csv": res.get("summary_diagnostics_csv", ""),
                "summary_slices_csv": res.get("summary_slices_csv", ""),
                "summary_judge_csv": res.get("summary_judge_csv", ""),
                "summary_slices_judge_csv": res.get("summary_slices_judge_csv", ""),
                "queries_log": res.get("queries_log", ""),
                "config_json": res.get("config_json", ""),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            all_run_records.append(record)
            rdir = record["run_dir"]
            if rdir:
                rp = Path(rdir)
                if rp.exists():
                    created_run_dirs.append(rp)

    manifest_path = dump_manifest(all_run_records, Path(outdir))

    copied = optional_sync_to_object_storage(created_run_dirs, object_store_dir)
    if copied is not None:
        print("Copied run artifacts to:")
        for p in copied:
            print(" -", p)

    print("Nightly runs completed.")
    print("Manifest:", str(manifest_path))


if __name__ == "__main__":
    main()
