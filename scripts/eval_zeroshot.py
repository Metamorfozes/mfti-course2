import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import torch
import open_clip

from src.eval.zeroshot import run_zeroshot_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenCLIP zero-shot accuracy with and without SAE.")
    parser.add_argument(
        "--sae_ckpt",
        default="artifacts/sae/sae_vitb32_dict4096_l3e2/ckpt_epoch_5.pt",
        help="Path to SAE checkpoint.",
    )
    parser.add_argument("--dict_size", type=int, default=4096)
    parser.add_argument("--model_name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--logs_dir", default="logs/zeroshot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device)
    model.eval()

    results = run_zeroshot_eval(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        sae_ckpt=args.sae_ckpt,
        dict_size=args.dict_size,
        batch_size=args.batch_size,
        device=device,
    )

    out = {
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "sae_ckpt": args.sae_ckpt,
        "dict_size": args.dict_size,
        **results,
    }

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "zeroshot_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("| Dataset | Baseline Acc | SAE Acc |")
    print("|---|---|---|")
    print(f"| CIFAR10 | {out['cifar10_baseline_acc']:.4f} | {out['cifar10_sae_acc']:.4f} |")
    print(f"| CIFAR100 | {out['cifar100_baseline_acc']:.4f} | {out['cifar100_sae_acc']:.4f} |")


if __name__ == "__main__":
    main()
