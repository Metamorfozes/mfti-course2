import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import torch

from src.sae.data import iter_embedding_batches
from src.sae.metrics import l0_metric, l1_loss, mse_loss, r2_score
from src.sae.model import SparseAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder on OpenCLIP embeddings.")
    parser.add_argument(
        "--emb_dir",
        default="artifacts/embeddings/flickr30k_openclip_vitb32",
        help="Directory containing meta.json and chunked embeddings.",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/sae/sae_vitb32_dict4096_l1",
        help="Output directory for checkpoints.",
    )
    parser.add_argument(
        "--logs_dir",
        default="logs/sae_vitb32_dict4096_l1",
        help="Directory for training logs.",
    )
    parser.add_argument("--dict_size", type=int, default=4096)
    parser.add_argument("--l1_coef", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    emb_dir = Path(args.emb_dir)
    meta_path = emb_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found at {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    embedding_dim = int(meta["embedding_dim"])
    out_dir = Path(args.out_dir)
    logs_dir = Path(args.logs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = SparseAutoencoder(input_dim=embedding_dim, dict_size=args.dict_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_path = logs_dir / "train.jsonl"
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in iter_embedding_batches(
            str(meta_path), args.batch_size, shuffle=True, seed=args.seed + epoch
        ):
            batch = batch.to(device)
            x_hat, h = model(batch)
            mse = mse_loss(x_hat, batch)
            l1 = l1_loss(h)
            loss = mse + args.l1_coef * l1

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            with torch.no_grad():
                l0 = l0_metric(h).item()
                r2 = r2_score(x_hat, batch).item()
                mse_val = mse.item()
                l1_val = l1.item()

            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "mse": mse_val,
                    "l1": l1_val,
                    "l0": l0,
                    "r2": r2,
                    "dict_size": args.dict_size,
                }
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                print(
                    f"epoch={epoch} step={step} mse={mse_val:.6f} "
                    f"l1={l1_val:.6f} l0={l0:.2f} r2={r2:.4f}"
                )

            step += 1

        ckpt = {
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "config": vars(args),
            "last_metrics": {
                "step": step - 1,
                "epoch": epoch,
                "mse": mse_val,
                "l1": l1_val,
                "l0": l0,
                "r2": r2,
                "dict_size": args.dict_size,
            },
        }
        ckpt_path = out_dir / f"ckpt_epoch_{epoch}.pt"
        torch.save(ckpt, ckpt_path)


if __name__ == "__main__":
    main()
