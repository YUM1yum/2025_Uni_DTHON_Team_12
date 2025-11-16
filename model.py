# model.py
import torch
import open_clip

def load_clip_model(model_name="ViT-B-32", pretrained="openai", device=None):
    # 항상 torch.device 객체로 통일
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)

    return model, preprocess, tokenizer, device   # ← 이제 torch.device 반환


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_finetuned_model(model, ckpt_path, device=None):
    # 여기서도 torch.device 객체로 통일
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    return model