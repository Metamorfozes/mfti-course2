from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Steer Kandinsky2.2 with SAE feature directions."
    )
    parser.add_argument(
        "--sae_ckpt",
        default="artifacts/sae/sae_vitb32_dict4096_l3e2/ckpt_epoch_5.pt",
    )
    parser.add_argument("--latent_id", type=int, required=True)
    parser.add_argument("--strength", type=float, default=3.0)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="artifacts/steer")
    return parser.parse_args()


def load_w_dec(ckpt_path: str) -> torch.Tensor:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    if "W_dec" in state:
        w_dec = state["W_dec"]
    elif "decoder.weight" in state:
        w_dec = state["decoder.weight"]
    else:
        raise KeyError("W_dec or decoder.weight not found in checkpoint model_state.")
    return w_dec.float()


def get_direction(
    w_dec: torch.Tensor, latent_id: int, embed_dim: int
) -> torch.Tensor:
    if w_dec.ndim != 2:
        raise ValueError(f"W_dec must be 2D, got shape {tuple(w_dec.shape)}")

    if w_dec.shape[1] == embed_dim:
        w_dec_aligned = w_dec
    elif w_dec.shape[0] == embed_dim:
        w_dec_aligned = w_dec.T
    else:
        raise ValueError(
            f"W_dec shape {tuple(w_dec.shape)} does not match embed_dim={embed_dim}"
        )

    if latent_id < 0 or latent_id >= w_dec_aligned.shape[0]:
        raise ValueError(
            f"latent_id {latent_id} out of range for dict_size={w_dec_aligned.shape[0]}"
        )

    direction = w_dec_aligned[latent_id]
    direction = direction / (direction.norm(p=2) + 1e-8)
    return direction


def maybe_enable_offload(pipe) -> None:
    if hasattr(pipe, "enable_model_cpu_offload"):
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass


def sanitize_prompt(prompt: str, max_len: int = 80) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in prompt.strip().lower())
    safe = "_".join(filter(None, safe.split("_")))
    if not safe:
        safe = "prompt"
    return safe[:max_len]


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    dtype = torch.float32

    w_dec = load_w_dec(args.sae_ckpt)

    prior = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior",
        torch_dtype=dtype,
    ).to(device)
    decoder = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder",
        torch_dtype=dtype,
    ).to(device)

    with torch.no_grad():
        prior_out = prior(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            generator=make_generator(device, args.seed),
        )
        if not hasattr(prior_out, "image_embeds") or not hasattr(
            prior_out, "negative_image_embeds"
        ):
            raise RuntimeError(
                "Prior output missing image_embeds/negative_image_embeds. "
                f"Available attributes: {dir(prior_out)}"
            )
        image_embeds = prior_out.image_embeds.to(dtype)
        negative_image_embeds = prior_out.negative_image_embeds.to(dtype)

        direction = get_direction(
            w_dec, args.latent_id, embed_dim=image_embeds.shape[-1]
        ).to(device=device, dtype=dtype)

        orig = image_embeds
        plus = image_embeds + args.strength * direction
        minus = image_embeds - args.strength * direction

        results = {
            "orig": orig,
            "plus": plus,
            "minus": minus,
        }

        safe_prompt = sanitize_prompt(args.prompt)
        saved_paths: list[Path] = []
        for key, embeds in results.items():
            dec_out = decoder(
                image_embeds=embeds,
                negative_image_embeds=negative_image_embeds,
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                generator=make_generator(device, args.seed),
            )
            if not hasattr(dec_out, "images"):
                raise RuntimeError(
                    "Decoder output missing images. Available attributes: "
                    f"{dir(dec_out)}"
                )
            img0 = dec_out.images[0]
            arr = np.asarray(img0).astype(np.float32)
            print(
                "DEBUG image stats:",
                "min",
                float(arr.min()),
                "max",
                float(arr.max()),
                "mean",
                float(arr.mean()),
                "has_nan",
                bool(np.isnan(arr).any()),
                "has_inf",
                bool(np.isinf(arr).any()),
            )
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            out_path = (
                out_dir / f"{safe_prompt}_latent{args.latent_id}_{key}.png"
            )
            img.save(out_path)
            saved_paths.append(out_path)

        for path in saved_paths:
            print(str(path))


if __name__ == "__main__":
    main()
