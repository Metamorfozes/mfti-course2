import sys
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import re

import torch
import open_clip

from src.eval.zeroshot import run_zeroshot_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OpenCLIP zero-shot accuracy for multiple SAE checkpoints."
    )
    parser.add_argument(
        "--sae_ckpts",
        nargs="+",
        default=[
            "artifacts/sae/sae_vitb32_dict4096_l1/ckpt_epoch_5.pt",
            "artifacts/sae/sae_vitb32_dict4096_l1e2/ckpt_epoch_5.pt",
            "artifacts/sae/sae_vitb32_dict4096_l3e2/ckpt_epoch_5.pt",
        ],
        help="List of SAE checkpoint paths.",
    )
    parser.add_argument("--dict_size", type=int, default=4096)
    parser.add_argument("--model_name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out_json", default="logs/zeroshot/zeroshot_results_all.json"
    )
    return parser.parse_args()


def parse_lambda_from_path(path_str: str) -> str:
    name = Path(path_str).parent.name
    if "_l3e2" in name:
        return "3e-2"
    if "_l1e2" in name:
        return "1e-2"
    if re.search(r"_l1(_|$)", name):
        return "1e-3"
    match = re.search(r"_l(\d+e\d+)", name)
    if match:
        coef = match.group(1)
        return coef.replace("e", "e-")
    return "unknown"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device)
    model.eval()

    results: List[Dict[str, float]] = []
    for ckpt in args.sae_ckpts:
        metrics = run_zeroshot_eval(
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            sae_ckpt=ckpt,
            dict_size=args.dict_size,
            batch_size=args.batch_size,
            device=device,
        )
        results.append(
            {
                "lambda": parse_lambda_from_path(ckpt),
                "sae_ckpt": ckpt,
                "dict_size": args.dict_size,
                **metrics,
            }
        )

    out = {
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "results": results,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("| Lambda | Dataset | Baseline Acc | SAE Acc |")
    print("|---|---|---|---|")
    for row in results:
        print(
            f"| {row['lambda']} | CIFAR10 | {row['cifar10_baseline_acc']:.4f} | {row['cifar10_sae_acc']:.4f} |"
        )
        print(
            f"| {row['lambda']} | CIFAR100 | {row['cifar100_baseline_acc']:.4f} | {row['cifar100_sae_acc']:.4f} |"
        )


if __name__ == "__main__":
    main()
