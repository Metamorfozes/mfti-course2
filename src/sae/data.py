import json
from pathlib import Path
from typing import Iterator

import torch


def iter_embedding_batches(
    meta_path: str, batch_size: int, shuffle: bool, seed: int
) -> Iterator[torch.Tensor]:
    meta_file = Path(meta_path)
    with meta_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    emb_dir = meta_file.parent
    chunk_files = meta.get("chunks")
    if not chunk_files:
        chunk_files = sorted(p.name for p in emb_dir.glob("chunk_*.pt"))

    rng = torch.Generator()
    rng.manual_seed(seed)

    for chunk_file in chunk_files:
        chunk_path = emb_dir / chunk_file
        payload = torch.load(chunk_path, map_location="cpu")
        embeddings = payload["embeddings"].float()
        num = embeddings.shape[0]

        if shuffle:
            indices = torch.randperm(num, generator=rng)
            embeddings = embeddings[indices]

        for start in range(0, num, batch_size):
            batch = embeddings[start : start + batch_size]
            if batch.numel() == 0:
                continue
            yield batch
