import json
from pathlib import Path
from typing import Dict, List

from PIL import Image


def load_topk(topk_path: str) -> Dict[int, List[str]]:
    results: Dict[int, List[str]] = {}
    with Path(topk_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            latent_id = int(record["latent_id"])
            top = record.get("top", [])
            paths = [item["path"] for item in top]
            results[latent_id] = paths
    return results


def make_collage(
    image_paths: List[str],
    images_dir: Path,
    tile_size: int,
    grid_size: int = 4,
) -> Image.Image:
    canvas_size = tile_size * grid_size
    collage = Image.new("RGB", (canvas_size, canvas_size), color=(0, 0, 0))

    for idx in range(grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        x0 = col * tile_size
        y0 = row * tile_size

        if idx >= len(image_paths):
            continue

        img_path = images_dir / image_paths[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize((tile_size, tile_size))
                collage.paste(img, (x0, y0))
        except Exception:
            continue

    return collage
