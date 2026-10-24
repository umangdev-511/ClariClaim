"""Evaluate the claims pipeline against dataset/sample_claims.csv.

Run:  python3 -m evaluation.run_eval [--no-cache]
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.perception import extract_claim, read_image
from src.decide import decide
from src.schema import OUTPUT_COLUMNS, validate_row

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"
SAMPLE_CSV = DATASET_ROOT / "sample_claims.csv"
HISTORY_CSV = DATASET_ROOT / "user_history.csv"
CACHE_DIR = REPO_ROOT / ".cache"
CACHE_FILE = CACHE_DIR / "perception.json"
ERRORS_CSV = REPO_ROOT / "evaluation" / "errors.csv"

COMPARED_FIELDS = [
    "claim_object",
    "evidence_standard_met",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_STATUSES = ["supported", "contradicted", "not_enough_information"]


def _cache_key(claim_text: str, image_path: str) -> str:
    raw = f"{claim_text}||{image_path}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_cache(path: Path) -> dict:
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def _load_history() -> dict[str, dict]:
    if not HISTORY_CSV.is_file():
        return {}
    with open(HISTORY_CSV, newline="", encoding="utf-8") as f:
        return {r["user_id"]: r for r in csv.DictReader(f)}


def _normalize_for_compare(value: str) -> str:
    parts = sorted(s.strip().lower() for s in value.split(";") if s.strip())
    return ";".join(parts)


def run_pipeline_row(row: dict, cache: dict, use_cache: bool, history: dict):
    claim_text = row["user_claim"]
    claim_object = row["claim_object"]

    claim_key = _cache_key(claim_text, "__claim__")
    if use_cache and claim_key in cache:
        claim_extract = cache[claim_key]
    else:
        claim_extract = extract_claim(claim_text, claim_object)
        cache[claim_key] = claim_extract

    image_paths = [p.strip() for p in row["image_paths"].split(";") if p.strip()]
    image_observations = []
    for img_path in image_paths:
        img_key = _cache_key(claim_text, img_path)
        if use_cache and img_key in cache:
            obs = cache[img_key]
        else:
            obs = read_image(img_path, claim_object)
            cache[img_key] = obs
        image_observations.append(obs)

    claim_row = {
        "user_id": row["user_id"],
        "image_paths": row["image_paths"],
        "user_claim": claim_text,
        "claim_object": claim_object,
    }
    history_row = history.get(row["user_id"])
    predicted = decide(claim_row, claim_extract, image_observations, history_row)
    return predicted, claim_extract, image_observations


def compare_field(predicted: str, expected: str, field: str) -> bool:
    if field in ("risk_flags", "supporting_image_ids"):
        return _normalize_for_compare(predicted) == _normalize_for_compare(expected)
    return predicted.strip().lower() == expected.strip().lower()


def print_confusion_matrix(matrix: dict[str, dict[str, int]]):
    col_w = 24
    label_w = 28
    print("\n=== Confusion Matrix: claim_status ===")
    print(f"{'':>{label_w}s}  {'PREDICTED':^{col_w * len(CLAIM_STATUSES)}s}")
    header = "".join(f"{s:>{col_w}s}" for s in CLAIM_STATUSES)
    print(f"{'':>{label_w}s}{header}")
    print(f"{'':>{label_w}s}{'-' * (col_w * len(CLAIM_STATUSES))}")
    for actual in CLAIM_STATUSES:
        counts = "".join(f"{matrix[actual].get(p, 0):>{col_w}d}" for p in CLAIM_STATUSES)
        label = f"ACTUAL {actual}"
        print(f"{label:>{label_w}s}{counts}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate claims pipeline")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached perception results")
    args = parser.parse_args()

    use_cache = not args.no_cache

    with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
        sample_rows = list(csv.DictReader(f))

    history = _load_history()
    cache = _load_cache(CACHE_FILE) if use_cache else {}
    total = len(sample_rows)

    print(f"Evaluating {total} rows from sample_claims.csv …\n")

    field_correct = Counter()
    field_total = Counter()
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    schema_violations = 0
    error_rows = []

    for i, row in enumerate(sample_rows):
        uid = row["user_id"]
        print(f"  [{i + 1:>2}/{total}] {uid} …", end=" ", flush=True)

        predicted, claim_ext, img_obs = run_pipeline_row(row, cache, use_cache, history)

        errors = validate_row(predicted)
        if errors:
            schema_violations += len(errors)
            print(f"SCHEMA ERROR: {errors}")
        else:
            print("ok")

        for field in COMPARED_FIELDS:
            pred_val = predicted.get(field, "")
            exp_val = row.get(field, "")
            field_total[field] += 1
            if compare_field(pred_val, exp_val, field):
                field_correct[field] += 1

        pred_status = predicted.get("claim_status", "").strip().lower()
        exp_status = row.get("claim_status", "").strip().lower()
        confusion[exp_status][pred_status] += 1

        if pred_status != exp_status:
            err = {"user_id": uid, "user_claim": row["user_claim"]}
            for field in COMPARED_FIELDS:
                err[f"predicted_{field}"] = predicted.get(field, "")
                err[f"expected_{field}"] = row.get(field, "")
            err["perception_json"] = json.dumps(
                {"claim_extract": claim_ext, "image_observations": img_obs}
            )
            error_rows.append(err)

    _save_cache(cache, CACHE_FILE)

    print("\n=== Per-Field Exact-Match Accuracy ===")
    print(f"{'Field':<30s} {'Correct':>8s} {'Total':>8s} {'Accuracy':>10s}")
    print("-" * 60)
    for field in COMPARED_FIELDS:
        correct = field_correct[field]
        t = field_total[field]
        acc = correct / t if t else 0
        print(f"{field:<30s} {correct:>8d} {t:>8d} {acc:>10.1%}")

    overall = sum(field_correct[f] for f in COMPARED_FIELDS)
    overall_total = sum(field_total[f] for f in COMPARED_FIELDS)
    print("-" * 60)
    print(f"{'OVERALL':<30s} {overall:>8d} {overall_total:>8d} {overall / overall_total:>10.1%}")

    print_confusion_matrix(confusion)

    print(f"\nSchema violations: {schema_violations}")

    if error_rows:
        err_cols = ["user_id", "user_claim"]
        for field in COMPARED_FIELDS:
            err_cols.extend([f"predicted_{field}", f"expected_{field}"])
        err_cols.append("perception_json")

        with open(ERRORS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=err_cols, quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(error_rows)
        print(f"Wrote {len(error_rows)} mismatched rows to {ERRORS_CSV}")
    else:
        print("No claim_status errors — errors.csv not written.")

    print()


if __name__ == "__main__":
    main()
