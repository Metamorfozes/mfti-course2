import json
import heapq
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from src.sae.model import SparseAutoencoder


def parse_latent_range(spec: str) -> List[int]:
    if ":" not in spec:
        return [int(spec)]
    start_str, end_str = spec.split(":", 1)
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else start
    return list(range(start, end))


def load_sae(
    ckpt_path: str, input_dim: int, dict_size: int | None, device: torch.device
) -> SparseAutoencoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    if dict_size is None:
        dict_size = int(ckpt["model_state"]["encoder.weight"].shape[0])
    sae = SparseAutoencoder(input_dim=input_dim, dict_size=dict_size).to(device)
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    return sae


def to_relative_path(path: str, images_dir: Path) -> str:
    p = Path(path)
    return p.name


def update_topk_heaps(
    heaps: Dict[int, List[Tuple[float, str]]],
    latent_ids: List[int],
    activations: torch.Tensor,
    rel_paths: List[str],
    top_k: int,
) -> None:
    acts = activations.cpu()
    for i, path in enumerate(rel_paths):
        for latent_id in latent_ids:
            val = float(acts[i, latent_id].item())
            heap = heaps[latent_id]
            if len(heap) < top_k:
                heapq.heappush(heap, (val, path))
            else:
                if val > heap[0][0]:
                    heapq.heapreplace(heap, (val, path))


def export_topk_jsonl(
    out_path: str, heaps: Dict[int, List[Tuple[float, str]]]
) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for latent_id in sorted(heaps.keys()):
            best_by_path: Dict[str, float] = {}
            for val, path in heaps[latent_id]:
                prev = best_by_path.get(path)
                if prev is None or val > prev:
                    best_by_path[path] = val
            top = sorted(best_by_path.items(), key=lambda x: x[1], reverse=True)
            record = {
                "latent_id": latent_id,
                "top": [{"path": path, "activation": val} for path, val in top],
            }
            f.write(json.dumps(record) + "\n")
