from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.interpret.caption import caption_pil_images, load_blip
from src.interpret.topk import parse_latent_range

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-interpret SAE latents using BLIP captions."
    )
    parser.add_argument(
        "--topk_path",
        default="artifacts/interpret/topk_vitb32_dict4096_l3e2/topk.jsonl",
    )
    parser.add_argument(
        "--images_dir",
        default="artifacts/datasets/flickr30k/flickr30k_images",
    )
    parser.add_argument(
        "--out_csv", default="artifacts/interpret/latents_0_300_blip.csv"
    )
    parser.add_argument("--latents", default="0:300")
    parser.add_argument("--max_images_per_latent", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_image_path(path_str: str, images_dir: Path) -> Optional[Path]:
    p = Path(path_str)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    candidates.append(images_dir / p)
    candidates.append(images_dir / p.name)
    parts = list(p.parts)
    if images_dir.name in parts:
        idx = parts.index(images_dir.name)
        rel = Path(*parts[idx + 1 :])
        candidates.append(images_dir / rel)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def normalize_tokens(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if len(t) >= 3 and t not in STOPWORDS]
    return tokens


def extract_keywords(captions: Iterable[str]) -> List[str]:
    counter: Counter[str] = Counter()
    for caption in captions:
        counter.update(normalize_tokens(caption))
    return [token for token, _ in counter.most_common(5)]


def load_topk_map(topk_path: Path) -> Dict[int, List[str]]:
    topk_map: Dict[int, List[str]] = {}
    with topk_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            latent_id = int(record["latent_id"])
            best_by_name: Dict[str, Tuple[float, str]] = {}
            for entry in record.get("top", []):
                path = entry.get("path", "")
                activation = float(entry.get("activation", 0.0))
                name = Path(path).name
                prev = best_by_name.get(name)
                if prev is None or activation > prev[0]:
                    best_by_name[name] = (activation, path)
            dedup_sorted = sorted(
                best_by_name.values(), key=lambda x: x[0], reverse=True
            )
            topk_map[latent_id] = [path for _, path in dedup_sorted]
    return topk_map


def pick_images(
    paths: List[str], images_dir: Path, max_images: int
) -> Tuple[List[Image.Image], List[str]]:
    pil_images: List[Image.Image] = []
    sample_paths: List[str] = []
    seen_paths: set[str] = set()
    seen_names: set[str] = set()
    seen_files: set[Path] = set()
    for path_str in paths:
        if len(pil_images) >= max_images:
            break
        name = Path(path_str).name
        if path_str in seen_paths or name in seen_names:
            continue
        seen_paths.add(path_str)
        seen_names.add(name)
        resolved = resolve_image_path(path_str, images_dir)
        if resolved is None or resolved in seen_files:
            continue
        try:
            img = Image.open(resolved).convert("RGB")
        except OSError:
            continue
        seen_files.add(resolved)
        pil_images.append(img)
        sample_paths.append(resolved.name)
    return pil_images, sample_paths


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    topk_path = Path(args.topk_path)
    images_dir = Path(args.images_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    latents = parse_latent_range(args.latents)
    topk_map = load_topk_map(topk_path)
    processor, model = load_blip(device)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["latent_id", "interpretation", "keywords", "sample_paths"],
        )
        writer.writeheader()

        for latent_id in latents:
            paths = topk_map.get(latent_id, [])
            pil_images, sample_paths = pick_images(
                paths, images_dir, args.max_images_per_latent
            )
            captions = caption_pil_images(pil_images, processor, model, device)
            keywords = extract_keywords(captions) if captions else []
            if keywords:
                interpretation = (
                    "This feature activates on images containing: "
                    + ", ".join(keywords)
                    + "."
                )
            else:
                interpretation = "This feature has no clear activating pattern."

            writer.writerow(
                {
                    "latent_id": latent_id,
                    "interpretation": interpretation,
                    "keywords": ", ".join(keywords),
                    "sample_paths": ";".join(sample_paths),
                }
            )


if __name__ == "__main__":
    main()
