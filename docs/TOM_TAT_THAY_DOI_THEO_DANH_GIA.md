# TÃ³m táº¯t thay Ä‘á»•i theo báº£n nháº­n xÃ©t vÃ  Ä‘Ã¡nh giÃ¡ repo

## 1. Pháº¡m vi

CÃ¡c thay Ä‘á»•i nÃ y xá»­ lÃ½ nhá»¯ng Ä‘iá»ƒm cÃ³ thá»ƒ sá»­a báº±ng code ngay, Ä‘á»“ng thá»i khÃ´ng giáº£ láº­p káº¿t quáº£ ML khi chÆ°a cÃ³ nhÃ£n tháº­t. Dá»¯ liá»‡u áº£nh/video cá»§a ngÆ°á»i dÃ¹ng Ä‘Æ°á»£c giá»¯ nguyÃªn; khÃ´ng tá»± Ã½ xÃ³a, khÃ´i phá»¥c hoáº·c chia láº¡i.

## 2. CÃ¡c lá»—i Æ°u tiÃªn Ä‘Ã£ sá»­a

### 2.1. BiLSTM aggregation

TrÆ°á»›c Ä‘Ã¢y classifier dÃ¹ng:

```python
sequence[:, -1, :]
```

Vá»›i BiLSTM, backward component táº¡i vá»‹ trÃ­ cuá»‘i khÃ´ng Ä‘áº¡i diá»‡n Ä‘áº§y Ä‘á»§ cho toÃ n chuá»—i. Model má»›i láº¥y final hidden state cá»§a forward vÃ  backward á»Ÿ layer cuá»‘i rá»“i ghÃ©p láº¡i:

```python
forward = hidden[-2]
backward = hidden[-1]
context = torch.cat((forward, backward), dim=1)
```

Implementation má»›i: `src/assembly/models/action_net.py`.

### 2.2. KhÃ´ng lá»c wrong action trÆ°á»›c FSM

`ComponentDwellGate` cÅ© chá»‰ giá»¯ action thuá»™c `expected_actions`. VÃ¬ váº­y khi FSM chá» `insert_refill` nhÆ°ng camera tháº¥y `cap`, evidence `screw_cap` bá»‹ bá» vÃ  FSM khÃ´ng thá»ƒ bÃ¡o lá»—i.

Gate má»›i:

- theo dÃµi má»i linh kiá»‡n há»£p lá»‡ Ä‘i vÃ o WORK ZONE;
- Ä‘Ã¡nh dáº¥u suggestion lÃ  expected hoáº·c unexpected nhÆ°ng khÃ´ng loáº¡i bá»;
- chá»‘ng phÃ¡t láº¡i linh kiá»‡n Ä‘ang giá»¯ nguyÃªn trong vÃ¹ng;
- cho phÃ©p linh kiá»‡n má»›i phÃ¡t event dÃ¹ linh kiá»‡n cÅ© váº«n cÃ²n tháº¥y;
- Ä‘á»ƒ FSM lÃ  nÆ¡i duy nháº¥t quyáº¿t Ä‘á»‹nh PASS/VIOLATION.

### 2.3. Bá» hard-code embedding dimension khá»i model

ÄÃ£ thÃªm `configs/action_model_config.json`, tÃ¡ch riÃªng:

- backbone vÃ  `embedding_dim`;
- FPS láº¥y máº«u;
- sequence length vÃ  stride;
- hidden size, layer, bidirectional, dropout;
- batch size, epoch, learning rate vÃ  seed;
- danh sÃ¡ch sÃ¡u action.

Khi Ä‘á»•i tá»« ViT-Base 768 chiá»u sang backbone khÃ¡c, sá»­a config vÃ  cache láº¡i feature thay vÃ¬ sá»­a code LSTM.

## 3. Pipeline ViT + LSTM Ä‘Ã£ bá»• sung

### 3.1. Quay video temporal

ThÃªm `scripts/record_assembly_videos.py`:

- quay má»™t chu trÃ¬nh/clip báº±ng Space;
- báº¯t buá»™c khai bÃ¡o person, session vÃ  scenario;
- tá»± Ä‘áº·t tÃªn video duy nháº¥t;
- tá»± ghi `recording_log.csv`.

### 3.2. Spatial encoder tháº­t

ThÃªm `src/assembly/models/spatial_encoder.py`:

- táº£i Hugging Face ViT theo config;
- preprocess áº£nh;
- láº¥y CLS embedding tá»«ng frame;
- freeze backbone cho giai Ä‘oáº¡n cache feature.

### 3.3. Feature caching

ThÃªm `scripts/extract_spatial_features.py`:

- Ä‘á»c video Ä‘á»‡ quy theo person/session;
- láº¥y máº«u theo FPS trong config;
- trÃ­ch embedding má»™t láº§n;
- lÆ°u `<video_id>.npy` vÃ  metadata `<video_id>.json`.

### 3.4. Temporal window dataset

ThÃªm `src/assembly/action_dataset.py`:

- Ä‘á»c annotation theo thá»i gian giÃ¢y;
- chá»‰ láº¥y Ä‘Ãºng split train/val/test;
- táº¡o sliding window 16 frame;
- padding Ä‘oáº¡n ngáº¯n;
- kiá»ƒm tra embedding dimension;
- cache tá»‘i Ä‘a bá»‘n video trong RAM vÃ  khÃ´ng giá»¯ mmap khÃ³a file trÃªn Windows.

### 3.5. Train vÃ  evaluate

ThÃªm:

- `scripts/train_action_model.py` â€” AdamW, CrossEntropy, CosineAnnealing, lÆ°u checkpoint tá»‘t nháº¥t theo validation Macro-F1.
- `scripts/evaluate_action_model.py` â€” Macro-F1, confusion matrix vÃ  classification report trÃªn val/test.
- `data/pen_actions/annotations_template.csv` â€” template annotation thá»‘ng nháº¥t.
- `src/assembly/models/vit_lstm_recognizer.py` â€” adapter tráº£ `Prediction(action, confidence)` cho runtime sau khi cÃ³ checkpoint.

Dataset temporal tá»± tá»« chá»‘i náº¿u cÃ¹ng má»™t `video_id` hoáº·c cÃ¹ng cáº·p `person/session` xuáº¥t hiá»‡n á»Ÿ nhiá»u split. ÄÃ¢y lÃ  hÃ ng rÃ o chá»‘ng data leakage, khÃ´ng chá»‰ lÃ  lá»i nháº¯c trong tÃ i liá»‡u.

## 4. Báº£o vá»‡ pipeline detector

Kiá»ƒm tra repo ngÃ y 03/09/2026 cho tháº¥y:

```text
61 images
61 label files
0 non-empty label files
0 bounding boxes
```

ÄÃ£ thÃªm:

- `src/assembly/detection_dataset.py` Ä‘á»ƒ kiá»ƒm tra áº£nh, label, tá»a Ä‘á»™ vÃ  sá»‘ box theo lá»›p.
- `scripts/validate_detection_dataset.py` Ä‘á»ƒ ngÆ°á»i dÃ¹ng cháº¡y trÆ°á»›c khi train.
- `scripts/train_detector.py` tá»± gá»i validator vÃ  dá»«ng náº¿u dataset khÃ´ng train Ä‘Æ°á»£c.

VÃ¬ váº­y lá»‡nh train hiá»‡n dá»«ng cÃ³ chá»§ Ä‘Ã­ch cho tá»›i khi cÃ³ bounding box tháº­t; há»‡ thá»‘ng khÃ´ng cÃ²n Ã¢m tháº§m train 61 áº£nh nhÆ° background.

## 5. Tá»• chá»©c package vÃ  storage

- Neural model tháº­t Ä‘Æ°á»£c chuyá»ƒn vÃ o `src/assembly/models/`.
- `models/pen_action_net.py` Ä‘Æ°á»£c giá»¯ lÃ m compatibility import Ä‘á»ƒ khÃ´ng phÃ¡ code cÅ©.
- `.gitignore` bá» qua raw video, feature cache, áº£nh dataset lá»›n, cache Ultralytics vÃ  model `.h5`.
- Annotation/config/code váº«n cÃ³ thá»ƒ lÆ°u báº±ng Git.
- `scripts/split_dataset.py` khÃ´ng cÃ²n máº·c Ä‘á»‹nh táº¡o label rá»—ng; tÃ¹y chá»n nÃ y chá»‰ báº­t rÃµ rÃ ng báº±ng `--create-empty-labels` cho áº£nh background tháº­t. Script cÅ©ng cáº£nh bÃ¡o ráº±ng random split chá»‰ dÃ¹ng smoke test.

KhÃ´ng Ä‘á»•i tÃªn thÆ° má»¥c repo trong láº§n sá»­a nÃ y vÃ¬ thao tÃ¡c Ä‘Ã³ cÃ³ thá»ƒ phÃ¡ remote, Ä‘Æ°á»ng dáº«n tÃ i liá»‡u vÃ  mÃ´i trÆ°á»ng Ä‘ang dÃ¹ng. Viá»‡c Ä‘á»•i tÃªn repo nÃªn thá»±c hiá»‡n riÃªng khi ngÆ°á»i dÃ¹ng xÃ¡c nháº­n.

## 6. Model ImageNet-1K hiá»‡n cÃ³

`models/tf_model.h5` Ä‘Ã£ Ä‘Æ°á»£c kiá»ƒm tra á»Ÿ cháº¿ Ä‘á»™ read-only:

- TensorFlow/Keras weights;
- ViT-Base, patch 16Ã—16;
- 12 encoder layer;
- embedding dimension 768;
- classifier 1.000 lá»›p ImageNet.

Model nÃ y cÃ³ Ã­ch lÃ m pretrained spatial backbone nhÆ°ng khÃ´ng pháº£i object detector vÃ  khÃ´ng tráº£ bounding box. Pipeline PyTorch hiá»‡n dÃ¹ng model name trong `action_model_config.json`. Viá»‡c chuyá»ƒn/tÃ­ch há»£p trá»±c tiáº¿p `.h5` cáº§n Ä‘Ãºng Hugging Face ViT config vÃ  image processor, nÃªn chÆ°a Ã©p file nÃ y vÃ o YOLO hoáº·c LSTM sai má»¥c Ä‘Ã­ch.

## 7. Nhá»¯ng viá»‡c cá»‘ Ã½ chÆ°a lÃ m

- KhÃ´ng train YOLO vÃ¬ táº¥t cáº£ label hiá»‡n rá»—ng.
- KhÃ´ng train LSTM vÃ¬ chÆ°a cÃ³ temporal videos vÃ  annotations.
- KhÃ´ng cÃ´ng bá»‘ accuracy/F1 giáº£.
- KhÃ´ng xÃ³a hoáº·c chia láº¡i 61 áº£nh cá»§a ngÆ°á»i dÃ¹ng.
- KhÃ´ng báº­t tá»± Ä‘á»™ng action recognition khi chÆ°a cÃ³ checkpoint Ä‘Æ°á»£c Ä‘Ã¡nh giÃ¡.
- KhÃ´ng Ä‘á»•i tÃªn repo hoáº·c Git remote.

## 8. CÃ¡ch kiá»ƒm thá»­

CÃ¡c nhÃ³m test bao gá»“m:

- FSM vÃ  violation.
- Temporal debouncer.
- Camera config/switching.
- Component dwell vÃ  wrong-action passthrough.
- BiLSTM forward/backward hidden aggregation.
- Action config vÃ  tensor shape.
- Temporal window padding.
- Detection dataset validation.

Káº¿t quáº£ kiá»ƒm thá»­ cáº­p nháº­t ngÃ y 07/09/2026:

```text
Ran 34 tests
OK
```

NgoÃ i ra, `python -m compileall -q src scripts tests models` hoÃ n táº¥t vá»›i exit code `0`. Validator detector dá»«ng Ä‘Ãºng chá»§ Ä‘Ã­ch vÃ¬ 61 label hiá»‡n cÃ³ chÆ°a chá»©a bounding box; cÃ¡c lá»‡nh train/evaluate action cÅ©ng dá»«ng báº±ng thÃ´ng bÃ¡o hÆ°á»›ng dáº«n khi chÆ°a cÃ³ annotation, feature hoáº·c checkpoint.

## 9. Äiá»ƒm báº¯t Ä‘áº§u tiáº¿p theo

NgÆ°á»i dÃ¹ng thá»±c hiá»‡n theo `docs/VIEC_BAN_CAN_LAM.md`. Theo láº§n kiá»ƒm tra ngÃ y 07/09/2026, repo cÃ³ 317 áº£nh raw trong hai session cá»§a `person01`, nhÆ°ng cÃ¡c split train/val/test Ä‘ang trá»‘ng vÃ  chÆ°a cÃ³ bounding box. BÆ°á»›c gáº§n nháº¥t lÃ  chá»n 20â€“30 áº£nh sáº¡ch tá»« `session02`, gÃ¡n nhÃ£n thá»­, sau Ä‘Ã³ thu thÃªm session Ä‘á»™c láº­p trÆ°á»›c khi train.

## 10. Cáº­p nháº­t cÃ´ng cá»¥ capture detection

`scripts/capture_detection_images.py` Ä‘Ã£ Ä‘Æ°á»£c chá»‰nh Ä‘á»ƒ áº£nh JPG lÆ°u ra khÃ´ng chá»©a dÃ²ng chá»¯ Ä‘iá»u khiá»ƒn cá»§a cá»­a sá»• preview. Script há»— trá»£ thÃªm `--width`, `--height`, `--no-mirror`, hiá»ƒn thá»‹ Ä‘á»™ phÃ¢n giáº£i camera thá»±c táº¿ vÃ  dÃ¹ng timestamp microsecond Ä‘á»ƒ trÃ¡nh ghi Ä‘Ã¨ áº£nh khi cháº¡y nhiá»u láº§n. HÆ°á»›ng dáº«n chá»¥p láº¡i theo tá»«ng person/session náº±m táº¡i má»¥c 4 cá»§a `docs/VIEC_BAN_CAN_LAM.md`.

## 11. Kiá»ƒm tra trÆ°á»›c khi Ä‘Æ°a lÃªn GitHub

NgÃ y 07/09/2026:

- 34/34 unit test Ä‘áº¡t vÃ  toÃ n bá»™ `src`, `scripts`, `tests`, `models` compile thÃ nh cÃ´ng;
- khÃ´ng phÃ¡t hiá»‡n file khÃ´ng bá»‹ ignore nÃ o lá»›n hÆ¡n 5 MB;
- raw images, model weights, artifacts, feature cache vÃ  61 label rá»—ng Ä‘á»u khÃ´ng Ä‘Æ°á»£c stage;
- `docs/plan.md` chá»‰ Ä‘Æ°á»£c chuyá»ƒn tá»« UTF-16 sang UTF-8, ná»™i dung khÃ´ng thay Ä‘á»•i;
- remote hiá»‡n váº«n lÃ  `23120340/Region_Based_Convolutional_Neural_Network_Demo`.

Checkpoint thá»­ `pen_parts_detector-3` cÃ³ precision, recall vÃ  mAP báº±ng 0 nÃªn khÃ´ng Ä‘Æ°á»£c phÃ¡t hÃ nh nhÆ° model há»£p lá»‡.

## 12. Tá»± Ä‘á»™ng hÃ³a pháº§n viá»‡c trong checklist dá»¯ liá»‡u

Cáº­p nháº­t ngÃ y 07/09/2026 theo yÃªu cáº§u rÃ  soÃ¡t `docs/VIEC_BAN_CAN_LAM.md`:

- ThÃªm `scripts/prepare_detection_pilot.py` Ä‘á»ƒ Ä‘á»c má»™t session áº£nh, Ä‘o Ä‘á»™ nÃ©t/Ä‘á»™ sÃ¡ng, dÃ¹ng perceptual hash Ä‘á»ƒ Æ°u tiÃªn ná»™i dung khÃ¡c nhau vÃ  sao chÃ©p bá»™ áº£nh Ä‘á» xuáº¥t sang má»™t thÆ° má»¥c má»›i mÃ  khÃ´ng ghi Ä‘Ã¨ káº¿t quáº£ cÅ©.
- ÄÃ£ cháº¡y trÃªn 125 áº£nh cá»§a `person01/session02`: ngÆ°á»¡ng Ä‘á»™ nÃ©t thÃ­ch nghi lÃ  `7.52`, cÃ³ 76 áº£nh vÆ°á»£t ngÆ°á»¡ng vÃ  Ä‘Ã£ chá»n Ä‘á»§ 30 áº£nh vÃ o `artifacts/detection_pilot/session02_pilot_30/images/`.
- Táº¡o `annotation_manifest.csv` Ä‘á»ƒ ngÆ°á»i dÃ¹ng ghi `keep/reject` vÃ  `quality_report.csv` Ä‘á»ƒ tra chá»‰ sá»‘ cá»§a toÃ n bá»™ 125 áº£nh. CÃ¡c file nÃ y náº±m trong `artifacts/`, chá»‰ lÆ°u cá»¥c bá»™ vÃ  khÃ´ng push lÃªn GitHub.
- ÄÃ£ kiá»ƒm tra trá»±c quan áº£nh Ä‘áº¡i diá»‡n: bá»™ Ä‘á» xuáº¥t cÃ³ bÃºt hoÃ n chá»‰nh, nhiá»u linh kiá»‡n vÃ  tÃ¬nh huá»‘ng tay cáº§m; tuy nhiÃªn áº£nh webcam cÃ²n má»m vÃ  cÃ³ linh kiá»‡n nhá», nÃªn váº«n báº¯t buá»™c duyá»‡t thá»§ cÃ´ng trÆ°á»›c khi upload CVAT.
- NÃ¢ng validator detector: `Trainable: YES` giá» yÃªu cáº§u train/val Ä‘á»u cÃ³ áº£nh vÃ  box, má»i áº£nh cÃ³ file label, khÃ´ng cÃ³ dÃ²ng label lá»—i vÃ  táº¥t cáº£ class trong `data.yaml` Ä‘á»u cÃ³ box.
- Cáº­p nháº­t thÃ´ng bÃ¡o cháº·n trong `scripts/train_detector.py` Ä‘á»ƒ ngÆ°á»i dÃ¹ng xá»­ lÃ½ danh sÃ¡ch `Training blockers` thay vÃ¬ chá»‰ nháº­n thÃ´ng bÃ¡o chung.
- ThÃªm test cho yÃªu cáº§u validation split, Ä‘á»§ class vÃ  thuáº­t toÃ¡n chá»n áº£nh pilot. Tá»•ng test tÄƒng tá»« 34 lÃªn 38.

Nhá»¯ng pháº§n khÃ´ng tá»± Ä‘á»™ng lÃ m thay ngÆ°á»i dÃ¹ng: xÃ¡c nháº­n áº£nh nÃ o tháº­t sá»± Ä‘áº¡t, váº½/chá»‰nh bounding box trÃªn CVAT, bá»‘ trÃ­ camera vÃ  váº­t tháº­t, thu session/ngÆ°á»i má»›i, quay video thao tÃ¡c vÃ  gÃ¡n timestamp hÃ nh Ä‘á»™ng. Sau khi cÃ³ báº£n CVAT export, cÃ¡c bÆ°á»›c chuáº©n hÃ³a split, validate, train vÃ  Ä‘Ã¡nh giÃ¡ cÃ³ thá»ƒ tiáº¿p tá»¥c trong repository.

## 13. ÄÃ¡nh giÃ¡ tÃ­ch há»£p bÃ i toÃ¡n Láº¯p rÃ¡p Há»™p tai nghe (Earbud) & Cáº­p nháº­t há»‡ thá»‘ng (08/09/2026)

### 13.1. Nhá»¯ng viá»‡c ngÆ°á»i dÃ¹ng Ä‘Ã£ lÃ m
1. **Má»Ÿ rá»™ng bÃ i toÃ¡n má»›i**: Chuyá»ƒn giao logic giÃ¡m sÃ¡t sang bÃ i toÃ¡n láº¯p rÃ¡p há»™p tai nghe khÃ´ng dÃ¢y (`earbud_assembly_v1`).
2. **Chuáº©n bá»‹ cáº¥u hÃ¬nh**:
   - `configs/earbud_fsm_config.json`: Äá»‹nh nghÄ©a quy trÃ¬nh 3 bÆ°á»›c (`pick_case` -> `insert_earbud` -> `close_case`).
   - `configs/camera_earbud_config.json`: Äá»‹nh nghÄ©a detector nháº­n diá»‡n 3 lá»›p (`Case`, `Earbud`, `Empty_Slot`) vÃ  Ã¡nh xáº¡ sang action FSM.
   - `configs/action_earbud_config.json`: Cáº¥u hÃ¬nh mÃ´ hÃ¬nh nháº­n diá»‡n hÃ nh Ä‘á»™ng ViT + BiLSTM cho tai nghe.
3. **ÄÆ°a vÃ o dataset tai nghe**: Cung cáº¥p bá»™ dá»¯ liá»‡u 151 áº£nh vÃ  245 bounding box (nguá»“n Roboflow) chia thÃ nh 3 táº­p `train` (106), `valid` (30), `test` (15).
4. **Huáº¥n luyá»‡n thá»­ nghiá»‡m baseline**: Cháº¡y train YOLOv8 trÃªn dataset nÃ y, táº¡o checkpoint `artifacts/training/earbud_detector/weights/best.pt` (káº¿t quáº£ baseline ghi nháº­n mAP50 Ä‘áº¡t ~0.71 trÃªn táº­p test hiá»‡n táº¡i).

---

### 13.2. ÄÃ¡nh giÃ¡ chuyÃªn sÃ¢u: Háº¡n cháº¿ cá»§a Dataset Tai nghe hiá»‡n táº¡i
> **Káº¿t luáº­n**: Dataset earbud **Ä‘á»§ Ä‘á»ƒ train thá»­ má»™t baseline**, nhÆ°ng **chÆ°a Ä‘á»§ sáº¡ch vÃ  Ä‘á»™c láº­p Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ hoáº·c dÃ¹ng realtime Ä‘Ã¡ng tin cáº­y**. Tráº¡ng thÃ¡i `Trainable: YES` tá»« validator chá»‰ xÃ¡c nháº­n Ä‘á»‹nh dáº¡ng YOLO há»£p lá»‡, khÃ´ng Ä‘áº£m báº£o tÃ­nh Ä‘á»™c láº­p thá»‘ng kÃª hay Ä‘á»™ tin cáº­y khoa há»c.

Báº£ng tá»•ng há»£p cÃ¡c váº¥n Ä‘á» phÃ¡t hiá»‡n:

| Váº¥n Ä‘á» | Chi tiáº¿t káº¿t quáº£ phÃ¡t hiá»‡n | TÃ¡c Ä‘á»™ng |
|---|---|---|
| **RÃ² rá»‰ dá»¯ liá»‡u (Data Leakage)** | CÃ³ **70 cáº·p frame liÃªn tiáº¿p** náº±m khÃ¡c split. Cá»¥ thá»ƒ: Frame 008 náº±m á»Ÿ `test`, Frame 009 náº±m á»Ÿ `train`, Frame 010 náº±m á»Ÿ `validation` trong khi gáº§n nhÆ° cÃ¹ng má»™t gÃ³c mÃ¡y vÃ  cáº£nh quay | Model há»c thuá»™c bá»‘i cáº£nh; Ä‘iá»ƒm mAP trÃªn test set bá»‹ thá»•i phá»“ng giáº£ táº¡o, khÃ´ng pháº£n Ã¡nh kháº£ nÄƒng khÃ¡i quÃ¡t |
| **Nguá»“n dá»¯ liá»‡u Ä‘Æ¡n Ä‘iá»‡u** | ToÃ n bá»™ 151 áº£nh dÆ°á»ng nhÆ° trÃ­ch tá»« cÃ¹ng má»™t video, cÃ¹ng má»™t ngÆ°á»i thá»±c hiá»‡n, cÃ¹ng camera, gÃ³c quay vÃ  ná»n | Model dá»… overfit vÃ o ná»n vÃ  Ã¡nh sÃ¡ng cá»¥ thá»ƒ; Ä‘em sang mÃ´i trÆ°á»ng khÃ¡c sáº½ tá»¥t Ä‘á»™ chÃ­nh xÃ¡c |
| **Label rá»—ng thiáº¿u sÃ³t** | **56/151 áº£nh cÃ³ label rá»—ng**. Kiá»ƒm tra trá»±c quan frame 008â€“010 tháº¥y há»™p sáº¡c (`Case`) xuáº¥t hiá»‡n ráº¥t rÃµ nhÆ°ng hoÃ n toÃ n khÃ´ng Ä‘Æ°á»£c gÃ¡n box | YOLO coi Ä‘Ã¢y lÃ  áº£nh ná»n Ã¢m tÃ­nh, khiáº¿n detector bá»‹ pháº¡t khi nháº­n ra Case tháº­t |
| **Bounding Box lá»—i (Noise)** | CÃ³ 3 box `Earbud` kÃ­ch thÆ°á»›c cá»±c nhá», chá»‰ khoáº£ng **1â€“6 pixel** | Nhiá»…u náº·ng cho anchor/loss calculation cá»§a máº¡ng YOLO |
| **Box trÃ¹ng láº·p** | Frame 013 trÃªn thá»±c táº¿ chá»‰ cÃ³ 2 earbud nhÆ°ng láº¡i cÃ³ **4 box `Earbud`** chá»“ng chÃ©o; tá»•ng cá»™ng cÃ³ 7 áº£nh bá»‹ gÃ¡n trÃªn hai box Earbud | GÃ¢y nháº§m láº«n trong loss phÃ¢n loáº¡i vÃ  Non-Maximum Suppression (NMS) |
| **Test set quÃ¡ má»ng** | Táº­p test chá»‰ cÃ³ **4 `Case`**, **11 `Earbud`**, **1 `Empty_Slot`** | Máº«u kiá»ƒm thá»­ quÃ¡ Ã­t, khÃ´ng Ä‘á»§ Ã½ nghÄ©a thá»‘ng kÃª Ä‘á»ƒ káº¿t luáº­n Ä‘á»™ chÃ­nh xÃ¡c tá»«ng lá»›p |
| **Máº¥t cÃ¢n báº±ng lá»›p nghiÃªm trá»ng** | `Empty_Slot` chá»‰ cÃ³ **38 box** trÃªn toÃ n bá»™ 151 áº£nh, tháº¥p hÆ¡n ráº¥t nhiá»u so vá»›i 144 box `Earbud` | Detector khÃ³ há»c Ä‘Æ°á»£c Ä‘áº·c trÆ°ng cá»§a khe trá»‘ng (`Empty_Slot`) |

**VÃ¬ váº­y: ChÆ°a nÃªn tin tÆ°á»Ÿng cÃ¡c chá»‰ sá»‘ mAP thu Ä‘Æ°á»£c tá»« táº­p test hiá»‡n táº¡i.**

---

### 13.3. Kiá»ƒm tra cáº¥u hÃ¬nh vÃ  luá»“ng FSM
- Cáº£ 3 file JSON cáº¥u hÃ¬nh (`camera_earbud_config.json`, `earbud_fsm_config.json`, `action_earbud_config.json`) Ä‘á»u Ä‘Ãºng cÃº phÃ¡p vÃ  khá»›p schema version 1.
- FSM engine váº­n hÃ nh chÃ­nh xÃ¡c chu trÃ¬nh:
  ```text
  pick_case â†’ insert_earbud â†’ close_case â†’ COMPLETED
  ```
- FSM xá»­ lÃ½ Ä‘Ãºng logic báº¯t lá»—i: BÃ¡o `VIOLATION` ("ChÆ°a Ä‘áº·t há»™p sáº¡c vÃ o vÃ¹ng láº¯p mÃ  Ä‘Ã£ thao tÃ¡c tai nghe") náº¿u xuáº¥t hiá»‡n hÃ nh vi `insert_earbud` trÆ°á»›c `pick_case`.

---

### 13.4. CÃ¡c cáº£i tiáº¿n ká»¹ thuáº­t Ä‘Ã£ thá»±c hiá»‡n ngay trong mÃ£ nguá»“n
Äá»ƒ giáº£i quyáº¿t cÃ¡c lá»—i tÃ­ch há»£p mÃ  khÃ´ng lÃ m giÃ¡n Ä‘oáº¡n bÃ i toÃ¡n cÅ©, ká»¹ sÆ° Ä‘Ã£ triá»ƒn khai cÃ¡c cáº­p nháº­t sau:

1. **Chuáº©n hÃ³a thÆ° má»¥c dá»¯ liá»‡u**:
   - ÄÃ£ Ä‘á»•i tÃªn thÆ° má»¥c tá»« `datasets/earbub_parts` thÃ nh `datasets/earbud_parts` (sá»­a lá»—i chÃ­nh táº£).
   - ÄÃ£ cáº­p nháº­t `path: datasets/earbud_parts` trong `datasets/earbud_parts/data.yaml`. Validator cháº¡y thÃ nh cÃ´ng vÃ  nháº­n diá»‡n chuáº©n xÃ¡c 151 áº£nh / 245 box.
2. **Tá»± Ä‘á»™ng suy luáº­n tÃªn run khi huáº¥n luyá»‡n (`scripts/train_detector.py`)**:
   - ThÃªm hÃ m `_infer_run_name`: Náº¿u dá»¯ liá»‡u truyá»n vÃ o chá»©a tá»« khÃ³a `earbud`, máº·c Ä‘á»‹nh checkpoint sáº½ lÆ°u vÃ o `artifacts/training/earbud_detector` (khá»›p hoÃ n toÃ n vá»›i Ä‘Æ°á»ng dáº«n model trong `camera_earbud_config.json`). Náº¿u chá»©a `pen`, lÆ°u vÃ o `pen_parts_detector`.
3. **Tá»•ng quÃ¡t hÃ³a phÃ­m táº¯t camera (`src/assembly/camera_app.py`)**:
   - TrÆ°á»›c Ä‘Ã¢y cÃ¡c phÃ­m `1â€“5` bá»‹ hardcode cá»‘ Ä‘á»‹nh vÃ o 5 bÆ°á»›c cá»§a quy trÃ¬nh láº¯p bÃºt bi.
   - ÄÃ£ nÃ¢ng cáº¥p Ã¡nh xáº¡ phÃ­m sá»‘ `1..N` tá»± Ä‘á»™ng sinh theo danh sÃ¡ch cÃ¡c bÆ°á»›c (`workflow`) trong báº¥t ká»³ file FSM config nÃ o:
     - Vá»›i tai nghe (3 bÆ°á»›c): PhÃ­m `1` lÃ  `pick_case`, phÃ­m `2` lÃ  `insert_earbud`, phÃ­m `3` lÃ  `close_case`.
     - Vá»›i bÃºt bi (5 bÆ°á»›c): PhÃ­m `1â€“5` giá»¯ nguyÃªn cÃ¡c bÆ°á»›c láº¯p bÃºt.
   - Cáº­p nháº­t dÃ²ng nháº¯c lá»‡nh trÃªn UI thá»i gian thá»±c: `SPACE confirm | 1-N steps | C switch cam | R reset | S shot | Q quit`.
4. **Má»Ÿ rá»™ng script quay video hÃ nh Ä‘á»™ng (`scripts/record_assembly_videos.py`)**:
   - ThÃªm tham sá»‘ `--project {pen,earbud,custom}`: Tá»± Ä‘á»™ng lÆ°u video vÃ o `data/earbud_actions/raw_videos` khi chá»n dá»± Ã¡n tai nghe.
   - Cho phÃ©p truyá»n tÃªn `--scenario` linh hoáº¡t, khÃ´ng cÃ²n gÃ² bÃ³ vÃ o danh sÃ¡ch ká»‹ch báº£n riÃªng cá»§a bÃºt.
5. **Bá»• sung hÃ m kiá»ƒm tra khÃ´ng gian hÃ¬nh há»c (`src/assembly/vision.py`)**:
   - Bá»• sung cÃ¡c phÆ°Æ¡ng thá»©c hÃ¬nh há»c vÃ o class `Detection`: `area`, `contains_point`, `intersection_area`, `overlap_ratio_with`, vÃ  `is_inside(container, threshold=0.5)`.
   - Chuáº©n bá»‹ ná»n táº£ng Ä‘á»ƒ giáº£i quyáº¿t háº¡n cháº¿: Kiá»ƒm tra Earbud thá»±c sá»± náº±m bÃªn trong Case hoáº·c Empty Slot thay vÃ¬ chá»‰ kiá»ƒm tra xuáº¥t hiá»‡n trong WORK ZONE.
6. **Má»Ÿ rá»™ng Unit Test**:
   - ThÃªm test kiá»ƒm tra náº¡p `camera_earbud_config.json` vÃ  kiá»ƒm tra hÃ¬nh há»c (`test_detection_spatial_containment`) trong `tests/test_camera_config.py`.
   - ThÃªm test kiá»ƒm thá»­ FSM tai nghe cáº£ luá»“ng Ä‘Ãºng vÃ  báº¯t lá»—i trong `tests/test_fsm.py`.
   - ToÃ n bá»™ **41/41 unit test** cháº¡y qua thÃ nh cÃ´ng (`Ran 41 tests in 0.14s - OK`).

---

### 13.5. CÃ¡c háº¡n cháº¿ cÃ²n láº¡i cáº§n ngÆ°á»i dÃ¹ng xá»­ lÃ½ & quyáº¿t Ä‘á»‹nh thiáº¿t káº¿
1. **Lá»›p `close_case` trong Object Detection**:
   - Hiá»‡n táº¡i class detector chá»‰ cÃ³ `Case`, `Earbud`, `Empty_Slot`. Detector má»™t khung hÃ¬nh khÃ´ng phÃ¢n biá»‡t Ä‘Æ°á»£c tráº¡ng thÃ¡i má»Ÿ náº¯p vÃ  Ä‘Ã³ng náº¯p náº¿u vá» há»™p chá»‰ Ä‘Æ°á»£c gÃ¡n chung nhÃ£n `Case`.
   - Do Ä‘Ã³, hÃ nh Ä‘á»™ng `close_case` hiá»‡n táº¡i cáº§n Ä‘Æ°á»£c kÃ­ch hoáº¡t qua: phÃ­m táº¯t `3`, hoáº·c mÃ´ hÃ¬nh nháº­n diá»‡n hÃ nh Ä‘á»™ng ViT + BiLSTM theo chuá»—i video thá»i gian, hoáº·c bá»• sung nhÃ£n detection riÃªng biá»‡t (`Case_Open` vÃ  `Case_Closed`).
2. **Vai trÃ² cá»§a `Empty_Slot`**:
   - Trong `camera_earbud_config.json`, lá»›p `Empty_Slot` Ä‘ang Ä‘á»ƒ `"action": null` nÃªn detector chá»‰ váº½ khung bao chá»© chÆ°a tÃ¡c Ä‘á»™ng trá»±c tiáº¿p lÃªn FSM.
   - CÃ³ thá»ƒ khai thÃ¡c: Khi sá»‘ lÆ°á»£ng `Empty_Slot` giáº£m tá»« 2 vá» 0 (káº¿t há»£p `is_inside`), há»‡ thá»‘ng tá»± Ä‘á»™ng xÃ¡c nháº­n tai nghe Ä‘Ã£ vÃ o vá»‹ trÃ­.
3. **Dá»¯ liá»‡u Video HÃ nh Ä‘á»™ng**:
   - `action_earbud_config.json` Ä‘Ã£ sáºµn sÃ ng, nhÆ°ng cÃ¡c áº£nh detection tÄ©nh khÃ´ng thá»ƒ dÃ¹ng Ä‘á»ƒ train BiLSTM. Cáº§n thu tháº­p video thao tÃ¡c theo thá»i gian vÃ  gÃ¡n nhÃ£n timestamp (`annotations.csv`) thÃ¬ má»›i train Ä‘Æ°á»£c máº¡ng nháº­n diá»‡n hÃ nh Ä‘á»™ng.
4. **Quyáº¿t Ä‘á»‹nh thiáº¿t káº¿ then chá»‘t**:
   - **Quy trÃ¬nh láº¯p rÃ¡p cáº§n kiá»ƒm tra 1 tai nghe hay kiá»ƒm tra Ä‘á»§ cáº£ 2 tai nghe (tai trÃ¡i vÃ  tai pháº£i)?**
   - Náº¿u láº¯p 1 tai: Giá»¯ nguyÃªn FSM hiá»‡n táº¡i.
   - Náº¿u cáº§n Ä‘á»§ cáº£ 2 tai: Cáº§n tÃ¡ch tráº¡ng thÃ¡i FSM (vÃ­ dá»¥: `insert_left_earbud`, `insert_right_earbud`) hoáº·c dÃ¹ng bá»™ Ä‘áº¿m slot / tai nghe.


