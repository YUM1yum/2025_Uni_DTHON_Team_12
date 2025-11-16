#############################################
# 통합 파이프라인 파일
# (샘플 추출 → ZIP 생성 → YOLO 변환 → dataset 구성 → train/val/test split)
# + preprocess 기능(collect_samples, VisualQueryDataset)
#############################################

import os
import random
import shutil
import zipfile
import json
from pathlib import Path
import yaml
from typing import List, Dict
from PIL import Image
from torch.utils.data import Dataset

#############################################
# 0. 기본 설정
#############################################
ROOT_DIR = "./"
TRAIN_DIR = os.path.join(ROOT_DIR, "train_valid/train")
SAMPLE_DIR = os.path.join(ROOT_DIR, "sample")
YOLO_OUTPUT_BASE = "yolo_labels"
DATASET_DIR = Path("dataset")
N_SAMPLE = 1000
CATEGORIES = ["press", "report"]

CLASS_MAP = {
    "V": 0,       # table
    "others": 1   # all others
}


#############################################
# 1. prefix 정리 함수
#############################################
def remove_prefix(filename: str):
    return "_".join(filename.split("_")[1:])


#############################################
# 2. 샘플 데이터 추출
#############################################
def make_sample_dataset():
    print("### STEP 1: Sampling dataset ###")
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    for cat in CATEGORIES:
        jpg_dir = os.path.join(TRAIN_DIR, f"{cat}_jpg")
        json_dir = os.path.join(TRAIN_DIR, f"{cat}_json")

        out_jpg = os.path.join(SAMPLE_DIR, f"{cat}_jpg")
        out_json = os.path.join(SAMPLE_DIR, f"{cat}_json")
        os.makedirs(out_jpg, exist_ok=True)
        os.makedirs(out_json, exist_ok=True)

        jpg_files = sorted([f for f in os.listdir(jpg_dir) if f.endswith(".jpg")])
        json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

        jpg_key_map = {remove_prefix(f.replace(".jpg","")): f for f in jpg_files}
        json_key_map = {remove_prefix(f.replace(".json","")): f for f in json_files}

        matched_keys = list(set(jpg_key_map.keys()) & set(json_key_map.keys()))
        print(f"[{cat}] matched:", len(matched_keys))

        sample_keys = random.sample(matched_keys, min(N_SAMPLE, len(matched_keys)))

        for key in sample_keys:
            shutil.copy2(os.path.join(jpg_dir, jpg_key_map[key]), os.path.join(out_jpg, jpg_key_map[key]))
            shutil.copy2(os.path.join(json_dir, json_key_map[key]), os.path.join(out_json, json_key_map[key]))

    print("Sampling complete.")


#############################################
# 3. 샘플 zip 생성
#############################################
def zip_sample():
    zip_path = os.path.join(ROOT_DIR, "sample.zip")

    def zipdir(path, ziph):
        for root, _, files in os.walk(path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, path)
                ziph.write(full_path, arcname)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipdir(SAMPLE_DIR, zipf)

    print("Zip complete:", zip_path)


#############################################
# 4. JSON → YOLO 변환
#############################################
def convert_to_yolo():
    print("### STEP 2: Converting JSON to YOLO format ###")

    json_dirs = ["press_json", "report_json"]

    for jd in json_dirs:
        os.makedirs(os.path.join(YOLO_OUTPUT_BASE, jd.replace("_json", "")), exist_ok=True)

    for json_dir in json_dirs:
        input_dir = os.path.join(SAMPLE_DIR, json_dir)
        output_dir = os.path.join(YOLO_OUTPUT_BASE, json_dir.replace("_json", ""))

        for file in os.listdir(input_dir):
            if not file.endswith(".json"):
                continue

            input_path = os.path.join(input_dir, file)

            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                res = data.get("source_data_info", {}).get("document_resolution", None)
                if not res:
                    continue

                img_w, img_h = res
                annotations = data.get("learning_data_info", {}).get("annotation", [])

                lines = []
                for ann in annotations:
                    cid = ann.get("class_id", "")
                    bbox = ann.get("bounding_box", [])
                    if len(bbox) != 4:
                        continue

                    x, y, w, h = bbox
                    cx = (x + w / 2) / img_w
                    cy = (y + h / 2) / img_h
                    norm_w = w / img_w
                    norm_h = h / img_h

                    label = "V" if cid.startswith("V") else "others"
                    class_id = CLASS_MAP[label]

                    lines.append(f"{class_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")

                output_name = file.replace("MI3_", "MI2_").replace(".json", ".txt")
                output_path = os.path.join(output_dir, output_name)

                with open(output_path, "w") as f:
                    f.write("\n".join(lines))

            except Exception as e:
                print(f"Error in {file}: {e}")


#############################################
# 5. YOLO 학습용 images/labels 구성
#############################################
def build_dataset_dirs():
    print("### STEP 3: Building dataset directory ###")

    IMG_DIRS = [
        f"{SAMPLE_DIR}/press_jpg",
        f"{SAMPLE_DIR}/report_jpg"
    ]
    LBL_DIRS = [
        "yolo_labels/press",
        "yolo_labels/report"
    ]

    target_imgs = DATASET_DIR / "images"
    target_lbls = DATASET_DIR / "labels"

    target_imgs.mkdir(parents=True, exist_ok=True)
    target_lbls.mkdir(parents=True, exist_ok=True)

    # 이미지 복사
    for src in IMG_DIRS:
        for fname in os.listdir(src):
            if fname.endswith(".jpg"):
                shutil.copy(os.path.join(src, fname), os.path.join(target_imgs, fname))

    # 라벨 복사
    for src in LBL_DIRS:
        for fname in os.listdir(src):
            if fname.endswith(".txt"):
                shutil.copy(os.path.join(src, fname), os.path.join(target_lbls, fname))


#############################################
# 6. train/val/test split
#############################################
def split_dataset():
    print("### STEP 4: Splitting dataset ###")

    img_dir = DATASET_DIR / "images"
    lbl_dir = DATASET_DIR / "labels"

    train_ratio = 0.7
    val_ratio = 0.2

    img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    random.shuffle(img_files)

    total = len(img_files)
    train_cut = int(total * train_ratio)
    val_cut = int(total * (train_ratio + val_ratio))

    splits = {
        "train": img_files[:train_cut],
        "val": img_files[train_cut:val_cut],
        "test": img_files[val_cut:]
    }

    for split, files in splits.items():
        (img_dir / split).mkdir(exist_ok=True)
        (lbl_dir / split).mkdir(exist_ok=True)

        for fname in files:
            stem = fname.replace(".jpg", "")
            shutil.move(str(img_dir / fname), str(img_dir / split / fname))

            if (lbl_dir / f"{stem}.txt").exists():
                shutil.move(str(lbl_dir / f"{stem}.txt"), str(lbl_dir / split / f"{stem}.txt"))

    # YAML 생성
    yaml_path = DATASET_DIR / "data.yaml"
    data_yaml = {
        "train": str((img_dir / "train").resolve()),
        "val": str((img_dir / "val").resolve()),
        "test": str((img_dir / "test").resolve()),
        "nc": 2,
        "names": ["table", "others"]
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    print("data.yaml generated:", yaml_path)


#############################################
# 🔵 추가: preprocess 기능
#############################################
def collect_samples(json_dir: Path, img_dir: Path, source_name: str) -> List[Dict]:
    all_samples = []
    json_files = sorted(json_dir.glob("*.json"))

    for jpath in json_files:
        with open(jpath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                print(f"[WARN] JSON parse error → skip: {jpath}")
                continue

        sdi = data.get("source_data_info", {})
        jpg_name = sdi.get("source_data_name_jpg")

        if not jpg_name:
            stem = jpath.stem
            if stem.startswith("MI3_"):
                jpg_name = "MI2_" + stem[4:] + ".jpg"
            else:
                jpg_name = stem + ".jpg"

        img_path = img_dir / jpg_name
        page_id = img_path.stem
        ldi = data.get("learning_data_info", {})
        ann_list = ldi.get("annotation", [])

        for ann in ann_list:
            class_id = ann.get("class_id", "")
            if not class_id.startswith("V"):
                continue

            bbox = ann.get("bounding_box")
            if not bbox or len(bbox) != 4:
                continue

            x, y, w, h = bbox
            query_text = ann.get("visual_instruction", "")
            if not query_text:
                continue

            answer_text = ann.get("visual_answer", "")
            sample = {
                "source": source_name,
                "page_id": page_id,
                "image_path": str(img_path),
                "instance_id": ann.get("instance_id"),
                "class_id": class_id,
                "bbox_xywh": [float(x), float(y), float(w), float(h)],
                "query_text": query_text.strip(),
                "answer_text": str(answer_text).strip(),
            }
            all_samples.append(sample)

    return all_samples


class VisualQueryDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = s["image_path"]
        x, y, w, h = s["bbox_xywh"]
        query_text = s["query_text"]

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x+w), min(H, y+h)
        crop = img.crop((x0, y0, x1, y1))

        if self.transform:
            crop = self.transform(crop)

        return {"image": crop, "text": query_text}


#############################################
# MAIN PIPELINE
#############################################
def main():
    make_sample_dataset()
    zip_sample()
    convert_to_yolo()
    build_dataset_dirs()
    split_dataset()
    print("### ALL DONE ###")


if __name__ == "__main__":
    main()