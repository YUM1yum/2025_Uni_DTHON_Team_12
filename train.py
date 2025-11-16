# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import load_clip_model, save_model
from preprocess import VisualQueryDataset

def collate_fn(batch, tokenizer):
    images = torch.stack([b["image"] for b in batch])
    texts = [b["text"] for b in batch]
    text_tokens = tokenizer(texts)
    return images, text_tokens

def train_clip(samples, batch_size=32, lr=1e-5, epochs=5, save_path="clip_finetuned.pt"):
    model, preprocess, tokenizer, device = load_clip_model()
    dataset = VisualQueryDataset(samples, transform=preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                        collate_fn=lambda b: collate_fn(b, tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, text_tokens in tqdm(loader):
            images, text_tokens = images.to(device), text_tokens.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                img_feat = model.encode_image(images)
                txt_feat = model.encode_text(text_tokens)
                img_feat = F.normalize(img_feat, dim=-1)
                txt_feat = F.normalize(txt_feat, dim=-1)
                logit_scale = model.logit_scale.exp()
                logits_text = logit_scale * txt_feat @ img_feat.t()
                logits_image = logits_text.t()
                targets = torch.arange(images.size(0), device=device)
                loss = (F.cross_entropy(logits_text, targets) + F.cross_entropy(logits_image, targets)) / 2
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        print(f"Epoch {epoch+1} avg loss: {total_loss/len(dataset):.4f}")
    save_model(model, save_path)
    print("Saved:", save_path)

if __name__ == "__main__":
    from preprocess import collect_samples
    from pathlib import Path

    print("⚙️ CLIP 학습을 위한 sample 수집 중...")

    press_samples = collect_samples(
        Path("./sample/press_json"),
        Path("./sample/press_jpg"),
        "press"
    )

    report_samples = collect_samples(
        Path("./sample/report_json"),
        Path("./sample/report_jpg"),
        "report"
    )

    samples = press_samples + report_samples
    print(f"len(samples) : {len(samples)}")

    print("CLIP fine-tuning start")

    train_clip(
        samples,
        batch_size=32,
        lr=1e-5,
        epochs=5,
        save_path="clip_finetuned.pt",
    )

    print("train complete! save_model: clip_finetuned.pt")

    