from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


README_TEXT = """---
license: mit
tags:
- mfti
- clip
- openclip
- sparse-autoencoder
- interpretability
library_name: pytorch
---

# SAE for OpenCLIP ViT-B/32

This repository contains a Sparse Autoencoder (SAE) checkpoint trained on OpenCLIP ViT-B/32 image embeddings.

- Dictionary size: 4096
- L1 coefficient: 3e-2
- Checkpoint: ckpt_epoch_5.pt
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload SAE checkpoint to Hugging Face Hub.")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument(
        "--ckpt_path",
        default="artifacts/sae/sae_vitb32_dict4096_l3e2/ckpt_epoch_5.pt",
    )
    parser.add_argument(
        "--meta_path",
        default="artifacts/embeddings/flickr30k_openclip_vitb32/meta.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is missing. Export HF_TOKEN before running.")

    api = HfApi(token=token)
    whoami = api.whoami()
    role = str(whoami.get("role", "")).lower()
    if role not in {"write", "admin"}:
        print(
            "HF_TOKEN has read-only permissions. Create a new token with Write access and retry."
        )
        raise SystemExit(1)
    api.create_repo(repo_id=args.repo_id, private=False, exist_ok=True)

    ckpt_path = Path(args.ckpt_path)
    api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo="ckpt_epoch_5.pt",
        repo_id=args.repo_id,
    )

    api.upload_file(
        path_or_fileobj=README_TEXT.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
    )

    meta_path = Path(args.meta_path)
    if meta_path.exists():
        api.upload_file(
            path_or_fileobj=str(meta_path),
            path_in_repo="meta.json",
            repo_id=args.repo_id,
        )


if __name__ == "__main__":
    main()
