"""Run the ClariClaim pipeline on dataset/claims.csv and write output.csv.

Usage:  python3 -m src.main
"""

import csv
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.perception import extract_claim, read_image
from src.decide import decide
from src.schema import write_output

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"
CLAIMS_CSV = DATASET_ROOT / "claims.csv"
HISTORY_CSV = DATASET_ROOT / "user_history.csv"
OUTPUT_CSV = REPO_ROOT / "output.csv"
CACHE_DIR = REPO_ROOT / ".cache"
CACHE_FILE = CACHE_DIR / "perception.json"


def _cache_key(claim_text: str, image_path: str) -> str:
    return hashlib.sha256(f"{claim_text}||{image_path}".encode()).hexdigest()


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


def main():
    with open(CLAIMS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    print(f"Processing {total} rows from claims.csv\n")

    history = _load_history()
    cache = _load_cache(CACHE_FILE)
    results = []

    for i, row in enumerate(rows, 1):
        uid = row["user_id"]
        print(f"  row {i}/{total}  {uid}", flush=True)

        claim_text = row["user_claim"]
        claim_object = row["claim_object"]

        claim_key = _cache_key(claim_text, "__claim__")
        if claim_key in cache:
            claim_extract = cache[claim_key]
        else:
            claim_extract = extract_claim(claim_text, claim_object)
            cache[claim_key] = claim_extract

        image_paths = [p.strip() for p in row["image_paths"].split(";") if p.strip()]
        image_observations = []
        for img_path in image_paths:
            img_key = _cache_key(claim_text, img_path)
            if img_key in cache:
                obs = cache[img_key]
            else:
                obs = read_image(img_path, claim_object)
                cache[img_key] = obs
            image_observations.append(obs)

        claim_row = {
            "user_id": uid,
            "image_paths": row["image_paths"],
            "user_claim": claim_text,
            "claim_object": claim_object,
        }
        predicted = decide(claim_row, claim_extract, image_observations,
                           history.get(uid))
        results.append(predicted)

    _save_cache(cache, CACHE_FILE)

    write_output(results, str(OUTPUT_CSV))
    print(f"\nDone. Wrote {len(results)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
