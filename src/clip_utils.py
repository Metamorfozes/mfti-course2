import torch
import open_clip


def load_openclip(model_name: str, pretrained: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
    )
    model = model.to(device)
    model.eval()
    return model, preprocess


def encode_pil_batch(model, preprocess, pil_images, device):
    images = torch.stack([preprocess(img) for img in pil_images]).to(device)
    with torch.no_grad():
        embeddings = model.encode_image(images)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings
