import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.sae.model import SparseAutoencoder


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def build_text_embeddings(
    model,
    tokenizer,
    class_names: List[str],
    device: torch.device,
) -> torch.Tensor:
    prompts = [f"a photo of a {name}" for name in class_names]
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_embeds = model.encode_text(tokens)
    return _l2_normalize(text_embeds)


def load_sae(ckpt_path: str, input_dim: int, dict_size: int, device: torch.device) -> SparseAutoencoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    sae = SparseAutoencoder(input_dim=input_dim, dict_size=dict_size).to(device)
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    return sae


def evaluate_dataset(
    model,
    preprocess,
    dataset,
    text_embeds: torch.Tensor,
    batch_size: int,
    device: torch.device,
    sae: SparseAutoencoder | None = None,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = 0
    total = 0

    for images, labels in loader:
        if torch.is_tensor(images):
            images = images.to(device)
        else:
            images = torch.stack([preprocess(img) for img in images]).to(device)
        labels = labels.to(device)
        with torch.no_grad():
            img_embeds = model.encode_image(images)
            if sae is not None:
                img_embeds, _ = sae(img_embeds)
            img_embeds = _l2_normalize(img_embeds)
            logits = img_embeds @ text_embeds.T
            preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    return correct / max(total, 1)


def run_zeroshot_eval(
    model,
    preprocess,
    tokenizer,
    sae_ckpt: str,
    dict_size: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    input_dim = getattr(model.visual, "output_dim", 512)
    sae = load_sae(sae_ckpt, input_dim=input_dim, dict_size=dict_size, device=device)

    cifar10 = datasets.CIFAR10(
        root="artifacts/datasets", train=False, download=True, transform=preprocess
    )
    cifar100 = datasets.CIFAR100(
        root="artifacts/datasets", train=False, download=True, transform=preprocess
    )

    cifar10_text = build_text_embeddings(model, tokenizer, cifar10.classes, device)
    cifar100_text = build_text_embeddings(model, tokenizer, cifar100.classes, device)

    cifar10_baseline = evaluate_dataset(
        model, preprocess, cifar10, cifar10_text, batch_size, device, sae=None
    )
    cifar10_sae = evaluate_dataset(
        model, preprocess, cifar10, cifar10_text, batch_size, device, sae=sae
    )
    cifar100_baseline = evaluate_dataset(
        model, preprocess, cifar100, cifar100_text, batch_size, device, sae=None
    )
    cifar100_sae = evaluate_dataset(
        model, preprocess, cifar100, cifar100_text, batch_size, device, sae=sae
    )

    return {
        "cifar10_baseline_acc": cifar10_baseline,
        "cifar10_sae_acc": cifar10_sae,
        "cifar100_baseline_acc": cifar100_baseline,
        "cifar100_sae_acc": cifar100_sae,
    }
