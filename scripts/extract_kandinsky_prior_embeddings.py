from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import torch
from diffusers import DiffusionPipeline
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths: List[Path]) -> None:
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return img, path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Kandinsky 2.2 prior image embeddings for a folder."
    )
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--model_id", default="kandinsky-community/kandinsky-2-2-prior"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    return torch.device(device_str)


def get_image_processor(prior) -> object:
    if hasattr(prior, "image_processor"):
        return prior.image_processor
    if hasattr(prior, "feature_extractor"):
        return prior.feature_extractor
    raise RuntimeError(
        "Prior pipeline missing image processor/feature_extractor. "
        f"Available attributes: {dir(prior)}"
    )


def get_image_encoder(prior) -> object:
    if hasattr(prior, "image_encoder"):
        return prior.image_encoder
    raise RuntimeError(
        "Prior pipeline missing image_encoder. " f"Available attributes: {dir(prior)}"
    )


def encode_images(prior, images: List[Image.Image], device: torch.device) -> torch.Tensor:
    processor = get_image_processor(prior)
    encoder = get_image_encoder(prior)
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = encoder(**inputs)
    if hasattr(outputs, "image_embeds"):
        return outputs.image_embeds
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output
    raise RuntimeError(
        "Image encoder output missing image_embeds/pooler_output. "
        f"Available attributes: {dir(outputs)}"
    )


def save_chunk(
    out_dir: Path, index: int, embeddings: torch.Tensor, filenames: List[str]
) -> str:
    out_path = out_dir / f"chunk_{index:03d}.pt"
    torch.save(
        {"embeddings": embeddings, "filenames": filenames},
        out_path,
    )
    return out_path.name


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = resolve_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    image_paths = sorted(
        [
            p
            for p in images_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    dataset = ImageFolderDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=lambda batch: list(zip(*batch)),
    )

    prior = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(
        device
    )
    if hasattr(prior, "enable_attention_slicing"):
        prior.enable_attention_slicing()

    chunk_embeddings: List[torch.Tensor] = []
    chunk_filenames: List[str] = []
    chunk_names: List[str] = []
    chunk_index = 0
    processed = 0
    total = len(image_paths)
    start_time = time.time()
    progress_path = out_dir / "progress.json"
    meta_path = out_dir / "meta.json"

    def current_chunk_size() -> int:
        return sum(e.shape[0] for e in chunk_embeddings)

    def write_meta() -> None:
        meta = {
            "embedding_dim": 1280,
            "total_images": total,
            "chunks": chunk_names,
            "model_id": args.model_id,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def flush_full_chunks() -> None:
        nonlocal chunk_embeddings, chunk_filenames, chunk_index
        if not chunk_embeddings:
            return
        all_embeds = torch.cat(chunk_embeddings, dim=0)
        all_names = list(chunk_filenames)
        while all_embeds.shape[0] >= args.chunk_size:
            chunk_embeds = all_embeds[: args.chunk_size]
            chunk_names_list = all_names[: args.chunk_size]
            chunk_name = save_chunk(out_dir, chunk_index, chunk_embeds, chunk_names_list)
            chunk_names.append(chunk_name)
            chunk_index += 1
            write_meta()
            all_embeds = all_embeds[args.chunk_size :]
            all_names = all_names[args.chunk_size :]
        chunk_embeddings = [all_embeds] if all_embeds.numel() > 0 else []
        chunk_filenames = all_names

    with torch.no_grad():
        for batch_idx, (images, names) in enumerate(loader, start=1):
            embeds = encode_images(prior, list(images), device)
            embeds = embeds.to(dtype).cpu()
            chunk_embeddings.append(embeds)
            chunk_filenames.extend(list(names))
            processed += len(names)
            flush_full_chunks()

            if batch_idx % 20 == 0 or processed == total:
                elapsed = max(time.time() - start_time, 1e-6)
                images_per_sec = processed / elapsed
                print(
                    f"{processed}/{total} | {images_per_sec:.2f} img/s | "
                    f"chunk_in_mem={current_chunk_size()}"
                )

            if batch_idx % 50 == 0 or processed == total:
                last_name = names[-1] if names else ""
                progress = {
                    "processed": processed,
                    "total": total,
                    "last_filename": last_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                progress_path.write_text(
                    json.dumps(progress, indent=2), encoding="utf-8"
                )

    if chunk_filenames:
        all_embeds = (
            torch.cat(chunk_embeddings, dim=0)
            if chunk_embeddings
            else torch.empty((0, 1280), dtype=dtype)
        )
        chunk_name = save_chunk(out_dir, chunk_index, all_embeds, chunk_filenames)
        chunk_names.append(chunk_name)
        write_meta()


if __name__ == "__main__":
    main()
