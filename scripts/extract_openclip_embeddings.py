import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.clip_utils import load_openclip, encode_pil_batch
from src.data import ImagePathsDataset
from src.io_utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract OpenCLIP image embeddings.")
    parser.add_argument("--data_dir", required=True, help="Root directory of images.")
    parser.add_argument("--out_dir", required=True, help="Output directory for embeddings.")
    parser.add_argument("--model_name", default="ViT-B-32", help="OpenCLIP model name.")
    parser.add_argument(
        "--pretrained",
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained tag.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def collate_paths(batch):
    return batch


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    dataset = ImagePathsDataset(str(data_dir))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_paths,
    )

    model, preprocess = load_openclip(args.model_name, args.pretrained, args.device)
    embedding_dim = getattr(model.visual, "output_dim", None)
    last_embed_dim = None

    chunk_paths = []
    chunk_embeddings = []
    chunk_files = []
    total_images = 0
    chunk_idx = 0

    for batch_paths in tqdm(loader, desc="Encoding"):
        pil_images = []
        rel_paths = []
        for p in batch_paths:
            with Image.open(p) as img:
                pil_images.append(img.convert("RGB"))
            rel_paths.append(str(Path(p).relative_to(data_dir)))

        embeddings = encode_pil_batch(model, preprocess, pil_images, args.device)
        embeddings_cpu = embeddings.cpu()
        last_embed_dim = embeddings_cpu.shape[-1]

        chunk_embeddings.append(embeddings_cpu)
        chunk_paths.extend(rel_paths)
        total_images += len(rel_paths)

        if len(chunk_paths) >= args.chunk_size:
            chunk_file = f"chunk_{chunk_idx:03d}.pt"
            chunk_path = out_dir / chunk_file
            embeddings_tensor = torch.cat(chunk_embeddings, dim=0).cpu()
            torch.save({"embeddings": embeddings_tensor, "paths": chunk_paths}, chunk_path)
            chunk_files.append(chunk_file)
            chunk_paths = []
            chunk_embeddings = []
            chunk_idx += 1

    if chunk_paths:
        chunk_file = f"chunk_{chunk_idx:03d}.pt"
        chunk_path = out_dir / chunk_file
        embeddings_tensor = torch.cat(chunk_embeddings, dim=0).cpu()
        torch.save({"embeddings": embeddings_tensor, "paths": chunk_paths}, chunk_path)
        chunk_files.append(chunk_file)

    if embedding_dim is None:
        embedding_dim = last_embed_dim

    meta = {
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "embedding_dim": embedding_dim,
        "total_images": total_images,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "chunks": chunk_files,
    }
    save_json(str(out_dir / "meta.json"), meta)


if __name__ == "__main__":
    main()
