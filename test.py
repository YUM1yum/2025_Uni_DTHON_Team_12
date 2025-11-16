# test.py
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from ultralytics import YOLO
from model import load_clip_model, load_finetuned_model


def get_yolo_candidates(image_path, yolo, conf=0.25, iou=0.5):
    results = yolo(image_path, conf=conf, iou=iou, verbose=False)
    boxes = results[0].boxes
    if boxes is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    candidates = [[float(x1), float(y1), float(x2), float(y2)]
                  for (x1,y1,x2,y2), c in zip(xyxy, cls) if int(c)==0]
    return candidates


@torch.no_grad()
def score_candidates_with_clip(model, preprocess, tokenizer,
                                image_path, query_text, cand_bboxes, device):
    if len(cand_bboxes) == 0:
        return [], torch.empty(0)

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    crops = []

    for (x1,y1,x2,y2) in cand_bboxes:
        cx1,cy1,cx2,cy2 = max(0,x1), max(0,y1), min(W,x2), min(H,y2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        crops.append(preprocess(img.crop((cx1,cy1,cx2,cy2))))

    if not crops:
        return [], torch.empty(0)

    images = torch.stack(crops).to(device)
    text_tokens = tokenizer([query_text]).to(device)

    try:
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            img_feat = model.encode_image(images)
            txt_feat = model.encode_text(text_tokens)
    except:
        return [], torch.empty(0)

    img_feat = F.normalize(img_feat, dim=-1)
    txt_feat = F.normalize(txt_feat, dim=-1)

    logit_scale = model.logit_scale.exp()
    logits = logit_scale * (txt_feat @ img_feat.t())

    return cand_bboxes, logits.squeeze(0).cpu()


def xyxy_to_xywh(box):
    x1,y1,x2,y2 = box
    return [x1, y1, x2-x1, y2-y1]


def run_inference(test_dir="./open/test",
                  ckpt_path="clip_finetuned.pt",
                  yolo_weight="./runs/detect/ui_yolo8n3/weights/best.pt",
                  save_path="submission.csv"):

    # ----------------------------------------
    # 모델 준비
    # ----------------------------------------
    print(f"DEBUG: 1. 모델 로드")
    model, preprocess, tokenizer, device_str = load_clip_model()
    device = torch.device(device_str)
    model = load_finetuned_model(model, ckpt_path, device)
    print(f"DEBUG: CLIP 모델 로드 완료 (Device={device})")

    print(f"DEBUG: 2. YOLO 로드: {yolo_weight}")
    yolo = YOLO(str(yolo_weight))

    # ----------------------------------------
    # 경로 준비
    # ----------------------------------------
    test_dir = Path(test_dir)
    query_dir = test_dir/"query"
    images_dir = test_dir/"images"

    query_files = sorted(query_dir.glob("*.json"))
    if not query_files:
        print(f"ERROR: query JSON 없음: {query_dir}")
        return

    print(f"DEBUG: 총 {len(query_files)}개의 쿼리 JSON 로드됨")

    # ----------------------------------------
    # insert code 방식 CSV row 저장 배열
    # ----------------------------------------
    rows = []

    # ----------------------------------------
    # JSON loop
    # ----------------------------------------
    for qpath in query_files:
        try:
            data = json.load(open(qpath, "r", encoding="utf-8"))
        except Exception as e:
            print(f"ERROR: JSON 로드 실패 {qpath}: {e}")
            continue

        src = data.get("source_data_info", {})
        anns = data.get("learning_data_info", {}).get("annotation", [])

        if not anns:
            continue

        image_path = images_dir / src.get("source_data_name_jpg", "")
        if not image_path.exists():
            print(f"WARN: 이미지 없음 → skip: {image_path}")
            continue

        for ann in anns:
            qid = ann.get("instance_id")
            qtext = ann.get("visual_instruction")

            if not qid or not qtext:
                continue

            # -----------------------------
            # YOLO 후보
            # -----------------------------
            cand = get_yolo_candidates(str(image_path), yolo)

            if len(cand) == 0:
                # fallback (0,0,1,1)
                rows.append({
                    "query_id": qid,
                    "query_text": qtext,
                    "pred_x": 0,
                    "pred_y": 0,
                    "pred_w": 1,
                    "pred_h": 1
                })
                continue

            # -----------------------------
            # CLIP scoring
            # -----------------------------
            cand, scores = score_candidates_with_clip(
                model, preprocess, tokenizer,
                str(image_path), qtext, cand, device
            )

            if scores.numel() == 0:
                rows.append({
                    "query_id": qid,
                    "query_text": qtext,
                    "pred_x": 0,
                    "pred_y": 0,
                    "pred_w": 1,
                    "pred_h": 1
                })
                continue

            # best bbox
            best = cand[scores.argmax().item()]
            x,y,w,h = xyxy_to_xywh(best)

            rows.append({
                "query_id": qid,
                "query_text": qtext,
                "pred_x": x,
                "pred_y": y,
                "pred_w": w,
                "pred_h": h
            })

    print(f"DEBUG: 총 {len(rows)}개의 예측 row 생성 완료")

    # ----------------------------------------
    # insert code 스타일 CSV 저장
    # ----------------------------------------
    df = pd.DataFrame(rows,
                      columns=["query_id", "query_text",
                               "pred_x", "pred_y", "pred_w", "pred_h"])

    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    run_inference()