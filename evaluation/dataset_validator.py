import json
import csv
import sys, os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List

REQUIRED_FIELDS = ["query"]
OPTIONAL_FIELDS = ["relevant_chunks", "slice", "metadata", "gold_answer"]

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln, json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {ln}: {e}")

def normalize_slices(rec):
    sl = rec.get("slice")
    if sl is None:
        return []
    if isinstance(sl, str):
        return [sl]
    if isinstance(sl, list):
        return [s for s in sl if isinstance(s, str)]
    return []

def validate_dataset(dataset_path: str, out_dir: str = "runs/") -> dict:
    path = Path(dataset_path)
    assert path.exists(), f"Dataset not found: {dataset_path}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = out_dir / "dataset_stats.csv"

    n = 0
    missing_required = defaultdict(int)
    type_errors = []
    slice_counts = defaultdict(int)
    with_relevant = 0

    for ln, rec in load_jsonl(dataset_path):
        n += 1
        for f in REQUIRED_FIELDS:
            if f not in rec:
                missing_required[f] += 1
        rc = rec.get("relevant_chunks")
        if rc is not None and not isinstance(rc, list):
            type_errors.append((ln, "relevant_chunks must be a list"))
        if isinstance(rc, list) and rc:
            with_relevant += 1
        for s in normalize_slices(rec):
            slice_counts[s] += 1

    with open(stats_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["total_queries", n])
        w.writerow(["with_relevant_chunks", with_relevant])
        w.writerow(["unique_slices", len(slice_counts)])
        for s, c in sorted(slice_counts.items()):
            w.writerow([f"slice_count::{s}", c])
        for f, c in sorted(missing_required.items()):
            w.writerow([f"missing_required::{f}", c])
        for ln, msg in type_errors:
            w.writerow([f"type_error::line_{ln}", msg])

    warnings = []
    for s, c in slice_counts.items():
        if c < 15:
            warnings.append(f"Slice '{s}' has low coverage (n={c}); consider adding queries to stabilize metrics.")
    return {
        "dataset_path": dataset_path,
        "total": n,
        "with_relevant_chunks": with_relevant,
        "slices": dict(slice_counts),
        "missing_required": dict(missing_required),
        "type_errors": type_errors,
        "stats_csv": str(stats_csv),
        "warnings": warnings,
    }

def validate_record(rec: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    if "id" not in rec or not isinstance(rec["id"], str) or not rec["id"].strip():
        errs.append("id must be a non-empty string")
    if "query" not in rec or not isinstance(rec["query"], str) or not rec["query"].strip():
        errs.append("query must be a non-empty string")
    if "relevant_chunks" not in rec or not isinstance(rec["relevant_chunks"], list) or any(not isinstance(x, str) or not x.strip() for x in rec["relevant_chunks"]):
        errs.append("relevant_chunks must be a list of non-empty string ids")

    if "gold_answer" in rec and rec["gold_answer"] is not None and not isinstance(rec["gold_answer"], str):
        errs.append("gold_answer must be a string if present")
    if "slice" in rec and rec["slice"] is not None and not (isinstance(rec["slice"], str) or (isinstance(rec["slice"], list) and all(isinstance(s, str) for s in rec["slice"]))):
        errs.append("slice must be a string or list of strings if present")
    if "metadata" in rec and rec["metadata"] is not None and not isinstance(rec["metadata"], dict):
        errs.append("metadata must be an object if present")

    return errs


def summarize_slices(recs: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in recs:
        sl = rec.get("slice")
        if isinstance(sl, str):
            counts[sl] = counts.get(sl, 0) + 1
        elif isinstance(sl, list):
            for s in sl:
                if isinstance(s, str):
                    counts[s] = counts.get(s, 0) + 1
    return counts

def write_dataset_stats(out_csv: str, total: int, counts_by_slice: Dict[str, int]) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["metric", "value"])
        w.writerow(["total_records", total])
        for s, c in sorted(counts_by_slice.items()):
            w.writerow([f"slice::{s}", c])

def main():
    ap = argparse.ArgumentParser(description="Validate RAG dataset JSONL and write dataset stats CSV.")
    ap.add_argument("dataset", help="Path to dataset .jsonl")
    ap.add_argument("--out", default="runs/dataset_stats.csv", help="Output CSV path for stats (default: runs/dataset_stats.csv)")
    ap.add_argument("--schema", default=None, help="Optional JSON Schema path for strict validation")
    ap.add_argument("--strict", action="store_true", help="Fail on errors; by default only prints warnings")
    args = ap.parse_args()

    pairs = list(load_jsonl(args.dataset))
    recs_only: List[Dict[str, Any]] = [rec for (_ln, rec) in pairs]

    errors: List[str] = []
    for idx, (_ln, rec) in enumerate(pairs, start=1):
        errs = validate_record(rec)
        if errs:
            errors.append(f"line {idx}: " + "; ".join(errs))

    if args.schema:
        try:
            import jsonschema
            with open(args.schema, "r", encoding="utf-8") as sf:
                schema = json.load(sf)
            for idx, (_ln, rec) in enumerate(pairs, start=1):
                try:
                    jsonschema.validate(instance=rec, schema=schema)
                except Exception as e:
                    errors.append(f"schema line {idx}: {e}")
        except ImportError:
            print("jsonschema not installed; skipping --schema validation", file=sys.stderr)

    counts_by_slice = summarize_slices(recs_only)
    write_dataset_stats(args.out, total=len(recs_only), counts_by_slice=counts_by_slice)

    if errors:
        print("WARNINGS/ERRORS:")
        for e in errors:
            print(" -", e)
        if args.strict:
            print("Strict mode: failing due to errors.")
            sys.exit(1)
    else:
        print("Dataset validated with no errors.")
if __name__ == "__main__":
    main()