from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


def load_blip(device: torch.device) -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)
    model.eval()
    return processor, model


def caption_pil_images(
    pil_images: List,  # PIL.Image.Image
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: torch.device,
) -> List[str]:
    if not pil_images:
        return []
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=30)
    captions = processor.batch_decode(output_ids, skip_special_tokens=True)
    return [c.strip() for c in captions]
