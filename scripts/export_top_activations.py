import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import torch

from src.interpret.topk import (
    export_topk_jsonl,
    load_sae,
    parse_latent_range,
    to_relative_path,
    update_topk_heaps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export top-activating images per SAE latent.")
    parser.add_argument(
        "--emb_dir",
        default="artifacts/embeddings/flickr30k_openclip_vitb32",
        help="Directory containing meta.json and chunked embeddings.",
    )
    parser.add_argument(
        "--images_dir",
        default="artifacts/datasets/flickr30k/flickr30k_images",
        help="Directory containing original images.",
    )
    parser.add_argument(
        "--sae_ckpt",
        default="artifacts/sae/sae_vitb32_dict4096_l3e2/ckpt_epoch_5.pt",
        help="Path to SAE checkpoint.",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/interpret/topk_vitb32_dict4096_l3e2",
        help="Output directory for top-k JSONL.",
    )
    parser.add_argument("--latents", default="0:300")
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    emb_dir = Path(args.emb_dir)
    images_dir = Path(args.images_dir)
    meta_path = emb_dir / "meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    embedding_dim = int(meta["embedding_dim"])
    device = torch.device(args.device)
    sae = load_sae(args.sae_ckpt, input_dim=embedding_dim, dict_size=None, device=device)

    latent_ids = parse_latent_range(args.latents)
    heaps = {latent_id: [] for latent_id in latent_ids}

    chunk_files = meta.get("chunks")
    if not chunk_files:
        chunk_files = sorted(p.name for p in emb_dir.glob("chunk_*.pt"))

    for chunk_file in chunk_files:
        chunk_path = emb_dir / chunk_file
        payload = torch.load(chunk_path, map_location="cpu")
        embeddings = payload["embeddings"].float()
        paths = payload["paths"]

        num = embeddings.shape[0]
        for start in range(0, num, args.batch_size):
            batch = embeddings[start : start + args.batch_size].to(device)
            batch_paths = paths[start : start + args.batch_size]

            with torch.no_grad():
                h = sae.act(sae.encoder(batch))

            rel_paths = [to_relative_path(p, images_dir) for p in batch_paths]

            update_topk_heaps(heaps, latent_ids, h, rel_paths, args.top_k)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "topk.jsonl"
    export_topk_jsonl(str(out_path), heaps)


if __name__ == "__main__":
    main()
