from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set


SEMANTIC_SET = {
    "person",
    "people",
    "man",
    "woman",
    "dog",
    "cat",
    "car",
    "bus",
    "food",
    "table",
    "street",
    "water",
    "pool",
    "horse",
    "child",
}

NON_SEMANTIC_SET = {
    "red",
    "blue",
    "green",
    "black",
    "white",
    "close",
    "blur",
    "striped",
    "sign",
    "metal",
    "texture",
}

GENERIC_SET = {
    "man",
    "woman",
    "person",
    "people",
    "boy",
    "girl",
    "standing",
    "sitting",
    "wearing",
    "looking",
    "holding",
    "walking",
    "group",
    "outside",
    "inside",
}


@dataclass(frozen=True)
class LatentRow:
    latent_id: int
    interpretation: str
    keywords: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select latents for report categories.")
    parser.add_argument(
        "--csv_path", default="artifacts/interpret/latents_0_300_blip.csv"
    )
    parser.add_argument(
        "--out_path", default="artifacts/interpret/selected_latents.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_keywords(raw: str) -> List[str]:
    if not raw:
        return []
    parts = [p.strip().lower() for p in raw.split(",")]
    return [p for p in parts if p]


def load_rows(csv_path: Path) -> List[LatentRow]:
    rows: List[LatentRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            latent_id = int(row["latent_id"])
            interpretation = row.get("interpretation", "").strip()
            keywords = normalize_keywords(row.get("keywords", ""))
            rows.append(LatentRow(latent_id, interpretation, keywords))
    return rows


def pick_n(
    rng: random.Random, candidates: Iterable[LatentRow], n: int, used: Set[int]
) -> List[int]:
    pool = [row for row in candidates if row.latent_id not in used]
    rng.shuffle(pool)
    picked = [row.latent_id for row in pool[:n]]
    used.update(picked)
    return picked


def keyword_hits(keywords: List[str], vocab: Set[str]) -> bool:
    return any(k in vocab for k in keywords)


def is_no_clear(keywords: List[str]) -> bool:
    if not keywords:
        return True
    hits = sum(1 for k in keywords[:5] if k in GENERIC_SET)
    return hits >= 4


def clean_score(keywords: List[str]) -> int:
    return sum(1 for k in keywords if k not in NON_SEMANTIC_SET)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    rows = load_rows(Path(args.csv_path))
    used: Set[int] = set()

    no_clear_candidates = [row for row in rows if is_no_clear(row.keywords)]
    semantic_candidates = [
        row for row in rows if keyword_hits(row.keywords, SEMANTIC_SET)
    ]
    too_specific_candidates = [
        row
        for row in rows
        if keyword_hits(row.keywords, NON_SEMANTIC_SET)
        and not keyword_hits(row.keywords, SEMANTIC_SET)
    ]

    no_clear = pick_n(rng, no_clear_candidates, 3, used)
    semantic = pick_n(rng, semantic_candidates, 3, used)
    too_specific = pick_n(rng, too_specific_candidates, 3, used)

    remaining_semantic = [row for row in semantic_candidates if row.latent_id not in used]
    rng.shuffle(remaining_semantic)
    remaining_semantic.sort(key=lambda row: clean_score(row.keywords), reverse=True)
    favorites = [row.latent_id for row in remaining_semantic[:3]]
    used.update(favorites)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "no_clear": no_clear,
        "too_specific": too_specific,
        "semantic": semantic,
        "favorites": favorites,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
