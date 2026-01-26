import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.interpret.collage import load_topk, make_collage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create collage images for SAE latents.")
    parser.add_argument(
        "--topk_path",
        default="artifacts/interpret/topk_vitb32_dict4096_l3e2/topk.jsonl",
        help="Path to topk.jsonl.",
    )
    parser.add_argument(
        "--images_dir",
        default="artifacts/datasets/flickr30k/flickr30k_images",
        help="Directory containing original images.",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/interpret/collages_vitb32_dict4096_l3e2",
        help="Output directory for collage PNGs.",
    )
    parser.add_argument(
        "--latent_ids",
        default="0,1,2,3,4,5,6,7,8",
        help="Comma-separated list of latent ids.",
    )
    parser.add_argument("--tile_size", type=int, default=224)
    return parser.parse_args()


def parse_latent_ids(spec: str) -> list[int]:
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def main() -> None:
    args = parse_args()
    topk = load_topk(args.topk_path)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_ids = parse_latent_ids(args.latent_ids)
    for latent_id in latent_ids:
        image_paths = topk.get(latent_id, [])
        collage = make_collage(image_paths, images_dir, args.tile_size, grid_size=4)
        out_path = out_dir / f"latent_{latent_id}.png"
        collage.save(out_path)


if __name__ == "__main__":
    main()
