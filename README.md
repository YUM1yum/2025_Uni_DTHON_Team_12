
# 📘 UI Visual Element Detection + CLIP Matching Pipeline

이 프로젝트는 다음 3가지를 수행하는 통합 파이프라인입니다.

1. **데이터 전처리(preprocess.py)**  
   - sample 추출  
   - JSON → YOLO txt 변환  
   - dataset/ 구조 자동 생성  
   - train/val/test split  
   - data.yaml 생성  

2. **YOLOv8 탐지 모델 학습**

3. **CLIP Fine-tuning(train.py) + 최종 Inference(test.py)**  
   - YOLO 후보 박스 중 CLIP이 가장 적합하다고 판단한 박스를 선택  
   - 최종 submission.csv 생성


---

# 🔧 1. Installation (Anaconda)

### Create Conda Environment
```bash
conda create -n ui_detect python=3.10 -y
conda activate ui_detect
````

### Install Dependencies

```bash
pip install -r requirements.txt
```

> Torch CUDA 12.1 버전이 포함되어 있으므로 NVIDIA GPU 환경을 권장합니다.

---

# 🛠 2. Preprocessing

아래 명령어 하나로 다음이 모두 자동 수행됩니다.

* train_valid/train 에서 샘플 추출
* sample 디렉토리 생성
* sample.zip 생성
* JSON → YOLO 변환
* dataset/images, dataset/labels 구성
* train/val/test split
* data.yaml 자동 생성

### Run

```bash
python preprocess.py
```

### Generated Structure

```
dataset/
│── images/
│    ├── train/
│    ├── val/
│    └── test/
│
└── labels/
     ├── train/
     ├── val/
     └── test/

dataset/data.yaml
```

---

# 🚀 3. YOLOv8 Training

전처리 후 `dataset/data.yaml` 이 생성되면 아래 명령어로 YOLO 학습을 시작합니다.

### Train Command

```bash
yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=16 name=ui_yolo8n patience=10
```

### Output

```
runs/detect/ui_yolo8n/weights/best.pt
```

---

# 🧠 4. CLIP Fine-tuning

preprocess 단계에서 생성한 sample 데이터를 이용해 CLIP을 학습합니다.

### Run

```bash
python train.py
```

### Output

```
clip_finetuned.pt
```

---

# 🔍 5. Inference (test.py)

YOLO 후보 bbox + CLIP scoring 조합으로 최종 bbox를 선택합니다.

### Run

```bash
python test.py
```

### Output

```
submission.csv
```

### CSV Columns

* `query_id`
* `query_text`
* `pred_x`
* `pred_y`
* `pred_w`
* `pred_h`

---

# 📂 Full Pipeline Summary

```bash
# 1) Create environment
conda create -n ui_detect python=3.10 -y
conda activate ui_detect
pip install -r requirements.txt

# 2) Preprocessing
python preprocess.py

# 3) YOLO Training
yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=16 name=ui_yolo8n patience=10

# 4) CLIP Fine-tuning
python train.py

# 5) Final Inference
python test.py
```

