# Viá»‡c báº¡n cáº§n lÃ m Ä‘á»ƒ hoÃ n thiá»‡n há»‡ thá»‘ng nháº­n diá»‡n vÃ  kiá»ƒm tra quy trÃ¬nh láº¯p rÃ¡p

> Cáº­p nháº­t toÃ n diá»‡n ngÃ y: **08/09/2026**  
> Dá»± Ã¡n Ã¡p dá»¥ng cho cáº£ hai bÃ i toÃ¡n: **Láº¯p rÃ¡p Há»™p tai nghe khÃ´ng dÃ¢y (Earbud Assembly)** vÃ  **Láº¯p rÃ¡p BÃºt bi (Pen Assembly)**.

---

## 0. PhÃ¢n cÃ´ng vÃ  tráº¡ng thÃ¡i thá»±c hiá»‡n

KÃ½ hiá»‡u quy Æ°á»›c:
- **ÄÃƒ LÃ€M (Ká»¹ sÆ°/Há»‡ thá»‘ng)**: Pháº§n ká»¹ thuáº­t, mÃ£ nguá»“n, script, cáº¥u hÃ¬nh vÃ  unit test Ä‘Ã£ hoÃ n thiá»‡n vÃ  kiá»ƒm tra tá»± Ä‘á»™ng trong repository.
- **Báº N ÄÃƒ LÃ€M**: Dá»¯ liá»‡u, cáº¥u hÃ¬nh vÃ  káº¿t quáº£ thá»±c nghiá»‡m báº¡n Ä‘Ã£ Ä‘Æ°a vÃ o dá»± Ã¡n.
- **Báº N Cáº¦N LÃ€M**: Cáº§n camera, váº­t tháº­t, thao tÃ¡c gÃ¡n nhÃ£n trÃªn CVAT/Roboflow hoáº·c quyáº¿t Ä‘á»‹nh nghiá»‡p vá»¥ cá»§a báº¡n; há»‡ thá»‘ng khÃ´ng thá»ƒ tá»± Ä‘oÃ¡n thay báº¡n.
- **MÃŒNH LÃ€M SAU**: MÃ¬nh sáº½ tiáº¿p tá»¥c ngay khi báº¡n Ä‘Æ°a dá»¯ liá»‡u Ä‘Ã£ lÃ m sáº¡ch vÃ o Ä‘Ãºng thÆ° má»¥c.

| Tráº¡ng thÃ¡i | Háº¡ng má»¥c cÃ´ng viá»‡c | Chi tiáº¿t ká»¹ thuáº­t |
|---|---|---|
| **Báº N ÄÃƒ LÃ€M** | Chuáº©n bá»‹ dataset tai nghe | Cung cáº¥p 151 áº£nh, 245 bounding box chia sáºµn 3 split (`datasets/earbud_parts`) |
| **Báº N ÄÃƒ LÃ€M** | Táº¡o cáº¥u hÃ¬nh FSM, camera vÃ  action | Táº¡o `earbud_fsm_config.json`, `camera_earbud_config.json`, `action_earbud_config.json` |
| **Báº N ÄÃƒ LÃ€M** | Huáº¥n luyá»‡n thá»­ baseline detector | Train YOLOv8 thÃ nh cÃ´ng táº¡i `artifacts/training/earbud_detector/weights/best.pt` |
| **ÄÃƒ LÃ€M** | Sá»­a lá»—i chÃ­nh táº£ thÆ° má»¥c | Äá»•i `datasets/earbub_parts` thÃ nh `datasets/earbud_parts`, sá»­a Ä‘Æ°á»ng dáº«n trong `data.yaml` |
| **ÄÃƒ LÃ€M** | Tá»± Ä‘á»™ng Ä‘áº·t tÃªn model detector | NÃ¢ng cáº¥p `scripts/train_detector.py` tá»± nháº­n diá»‡n dá»¯ liá»‡u earbud Ä‘á»ƒ lÆ°u vÃ o `earbud_detector` |
| **ÄÃƒ LÃ€M** | Tá»•ng quÃ¡t hÃ³a phÃ­m táº¯t camera | PhÃ­m sá»‘ `1..N` tá»± Ä‘á»™ng Ã¡nh xáº¡ theo quy trÃ¬nh FSM (tai nghe dÃ¹ng 1â€“3, bÃºt dÃ¹ng 1â€“5) |
| **ÄÃƒ LÃ€M** | Má»Ÿ rá»™ng script quay video hÃ nh Ä‘á»™ng | `scripts/record_assembly_videos.py` há»— trá»£ `--project earbud` vÃ  ká»‹ch báº£n linh hoáº¡t |
| **ÄÃƒ LÃ€M** | Bá»• sung hÃ m kiá»ƒm tra khÃ´ng gian hÃ¬nh há»c | Bá»• sung `is_inside`, `overlap_ratio_with`, `contains_point` trong `src/assembly/vision.py` |
| **ÄÃƒ LÃ€M** | Kiá»ƒm thá»­ tá»± Ä‘á»™ng toÃ n diá»‡n | ToÃ n bá»™ 41/41 unit test Ä‘áº¡t `OK`, kiá»ƒm tra cáº£ FSM tai nghe, FSM bÃºt vÃ  hÃ¬nh há»c |
| **Báº N Cáº¦N LÃ€M** | Sá»­a lá»—i nhÃ£n vÃ  khá»­ rÃ² rá»‰ dataset | Xá»­ lÃ½ 56 áº£nh rá»—ng, xÃ³a box lá»—i/trÃ¹ng, quay session má»›i Ä‘á»™c láº­p cho val vÃ  test |
| **Báº N Cáº¦N LÃ€M** | Chá»‘t quyáº¿t Ä‘á»‹nh thiáº¿t káº¿ FSM | Chá»n láº¯p 1 tai nghe hay kiá»ƒm tra Ä‘á»§ cáº£ 2 tai nghe (trÃ¡i vÃ  pháº£i) |
| **Báº N Cáº¦N LÃ€M** | Quay video hÃ nh Ä‘á»™ng cÃ³ timestamp | Quay cÃ¡c chu trÃ¬nh tai nghe báº±ng `record_assembly_videos.py` vÃ  gÃ¡n nhÃ£n thá»i gian |
| **MÃŒNH LÃ€M SAU** | Huáº¥n luyá»‡n láº¡i detector chuáº©n | Khi cÃ³ dataset sáº¡ch, train láº¡i detector, Ä‘Ã¡nh giÃ¡ mAP chuáº©n xÃ¡c trÃªn test set Ä‘á»™c láº­p |
| **MÃŒNH LÃ€M SAU** | TÃ­ch há»£p xÃ¡c thá»±c hÃ¬nh há»c nÃ¢ng cao | GhÃ©p Ä‘iá»u kiá»‡n Earbud náº±m bÃªn trong Case/Empty Slot vÃ o logic realtime |

---

## 1. ÄÃ¡nh giÃ¡ chuyÃªn sÃ¢u: Háº¡n cháº¿ cá»§a Dataset Tai nghe hiá»‡n táº¡i

> [!WARNING]
> **Káº¿t luáº­n cá»‘t lÃµi:** Dataset earbud **Ä‘á»§ Ä‘á»ƒ train thá»­ má»™t baseline**, nhÆ°ng **chÆ°a Ä‘á»§ sáº¡ch vÃ  Ä‘á»™c láº­p Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ hoáº·c dÃ¹ng realtime Ä‘Ã¡ng tin cáº­y**. Tráº¡ng thÃ¡i `Trainable: YES` chá»‰ xÃ¡c nháº­n Ä‘á»‹nh dáº¡ng file YOLO há»£p lá»‡ vá» máº·t ká»¹ thuáº­t, khÃ´ng Ä‘áº£m báº£o tÃ­nh Ä‘á»™c láº­p hay cháº¥t lÆ°á»£ng dá»¯ liá»‡u.

### 1.1. CÃ¡c váº¥n Ä‘á» phÃ¡t hiá»‡n trong Ä‘á»£t kiá»ƒm tra

| Váº¥n Ä‘á» | Káº¿t quáº£ phÃ¡t hiá»‡n | TÃ¡c Ä‘á»™ng thá»±c táº¿ |
|---|---|---|
| **RÃ² rá»‰ dá»¯ liá»‡u (Data Leakage)** | CÃ³ **70 cáº·p frame liÃªn tiáº¿p** náº±m khÃ¡c split. Cá»¥ thá»ƒ: Frame 008 á»Ÿ `test`, Frame 009 á»Ÿ `train`, Frame 010 á»Ÿ `validation` vÃ  gáº§n nhÆ° cÃ¹ng má»™t cáº£nh quay | Model há»c thuá»™c bá»‘i cáº£nh thay vÃ¬ há»c Ä‘áº·c trÆ°ng tá»•ng quÃ¡t. Äiá»ƒm sá»‘ mAP trÃªn test set hiá»‡n táº¡i bá»‹ thá»•i phá»“ng giáº£ táº¡o |
| **Nguá»“n dá»¯ liá»‡u Ä‘Æ¡n Ä‘iá»‡u** | ToÃ n bá»™ 151 áº£nh dÆ°á»ng nhÆ° trÃ­ch tá»« cÃ¹ng má»™t video, cÃ¹ng má»™t ngÆ°á»i thá»±c hiá»‡n, cÃ¹ng gÃ³c camera vÃ  Ä‘iá»u kiá»‡n Ã¡nh sÃ¡ng | Äem mÃ´ hÃ¬nh sang mÃ¡y khÃ¡c, gÃ³c quay khÃ¡c hoáº·c Ã¡nh sÃ¡ng khÃ¡c sáº½ sá»¥t giáº£m Ä‘á»™ chÃ­nh xÃ¡c |
| **Label rá»—ng thiáº¿u sÃ³t** | **56/151 áº£nh cÃ³ label rá»—ng**. Kiá»ƒm tra frame 008â€“010 tháº¥y há»™p sáº¡c rÃµ rÃ ng nhÆ°ng khÃ´ng Ä‘Æ°á»£c gÃ¡n nhÃ£n `Case` | YOLO coi cÃ¡c áº£nh nÃ y lÃ  background Ã¢m tÃ­nh; Ä‘iá»u nÃ y dáº¡y máº¡ng neuron pháº¡t cÃ¡c dá»± Ä‘oÃ¡n Case Ä‘Ãºng |
| **Bounding box lá»—i** | CÃ³ 3 box `Earbud` chá»‰ khoáº£ng **1â€“6 pixel** | GÃ¢y nhiá»…u anchor vÃ  tÃ­nh toÃ¡n hÃ m máº¥t mÃ¡t (loss) |
| **Bounding box trÃ¹ng láº·p** | Frame 013 chá»‰ tháº¥y 2 earbud nhÆ°ng cÃ³ tá»›i **4 box `Earbud`**; tá»•ng cá»™ng 7 áº£nh cÃ³ trÃªn hai box Earbud | GÃ¢y nháº§m láº«n cho thuáº­t toÃ¡n triá»‡t tiÃªu box trÃ¹ng (NMS) |
| **Test set quÃ¡ má»ng** | Táº­p test chá»‰ cÃ³ **4 `Case`**, **11 `Earbud`**, **1 `Empty_Slot`** | Máº«u kiá»ƒm thá»­ quÃ¡ Ã­t, khÃ´ng Ä‘á»§ cÆ¡ sá»Ÿ thá»‘ng kÃª Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ mAP theo tá»«ng lá»›p |
| **Máº¥t cÃ¢n báº±ng lá»›p** | `Empty_Slot` chá»‰ cÃ³ **38 box**, tháº¥p hÆ¡n ráº¥t nhiá»u so vá»›i 144 box `Earbud` | Detector há»c nháº­n diá»‡n khe trá»‘ng kÃ©m hÆ¡n háº³n so vá»›i tai nghe |

> [!CAUTION]
> **VÃ¬ váº­y: Tuyá»‡t Ä‘á»‘i chÆ°a nÃªn tin cáº­y mAP thu Ä‘Æ°á»£c tá»« táº­p test hiá»‡n táº¡i Ä‘á»ƒ bÃ¡o cÃ¡o hay Ä‘Ã¡nh giÃ¡ sáº£n pháº©m.**

---

## 2. Báº¡n cáº§n sá»­a dataset tai nghe nhÆ° tháº¿ nÃ o?

### 2.1. NÄƒm viá»‡c cáº§n sá»­a ngay trÃªn nhÃ£n hiá»‡n cÃ³
1. **Kiá»ƒm tra láº¡i toÃ n bá»™ 56 label rá»—ng:** Má»Ÿ tá»«ng áº£nh trong 56 áº£nh nÃ y. Náº¿u áº£nh cÃ³ `Case`, `Earbud` hoáº·c `Empty_Slot`, báº¯t buá»™c pháº£i khoanh box Ä‘á»§. Chá»‰ Ä‘á»ƒ label rá»—ng náº¿u áº£nh hoÃ n toÃ n khÃ´ng cÃ³ linh kiá»‡n nÃ o (áº£nh bÃ n trá»‘ng hoáº·c chá»‰ cÃ³ tay).
2. **Chuáº©n hÃ³a sá»‘ lÆ°á»£ng box Earbud:** Má»—i earbud váº­t lÃ½ chá»‰ cÃ³ Ä‘Ãºng má»™t bounding box Ã´m sÃ¡t. Tuyá»‡t Ä‘á»‘i khÃ´ng váº½ thÃªm box to bao cáº£ hai earbud cÃ¹ng lÃºc.
3. **XÃ³a/sá»­a 3 box rÃ¡c:** TÃ¬m vÃ  xÃ³a cÃ¡c box kÃ­ch thÆ°á»›c 1â€“6 pixel trong dá»¯ liá»‡u.
4. **Khá»­ rÃ² rá»‰ dá»¯ liá»‡u (Quan trá»ng nháº¥t):** KhÃ´ng chia ngáº«u nhiÃªn cÃ¡c frame trÃ­ch tá»« cÃ¹ng má»™t video vÃ o train, val, test. ToÃ n bá»™ 151 áº£nh tá»« video Ä‘áº§u tiÃªn nÃ y **pháº£i Ä‘Æ°á»£c gom toÃ n bá»™ vÃ o táº­p `train`**.
5. **Quay session má»›i Ä‘á»™c láº­p cho validation vÃ  test:**
   - Táº­p `validation` pháº£i lÃ  má»™t buá»•i quay riÃªng biá»‡t (thay Ä‘á»•i gÃ³c quay hoáº·c ná»n nháº¹).
   - Táº­p `test` pháº£i lÃ  má»™t buá»•i quay Ä‘á»™c láº­p hoÃ n toÃ n (ngÆ°á»i khÃ¡c lÃ m hoáº·c ngÃ y khÃ¡c) vÃ  Ä‘Æ°á»£c giá»¯ kÃ­n Ä‘á»ƒ cháº¥m Ä‘iá»ƒm.

### 2.2. Má»¥c tiÃªu sá»‘ lÆ°á»£ng cho Ä‘á»£t thu tháº­p tiáº¿p theo
Äá»ƒ mÃ´ hÃ¬nh phÃ¡t hiá»‡n á»•n Ä‘á»‹nh vÃ  Ä‘Ã¡nh giÃ¡ tin cáº­y, bá»™ dá»¯ liá»‡u vÃ²ng tiáº¿p theo nÃªn Ä‘áº¡t:
- **300â€“500 áº£nh** tháº­t sá»± khÃ¡c nhau.
- `Case`: Ãt nháº¥t **150 box**.
- `Earbud`: **250â€“300 box**.
- `Empty_Slot`: Ãt nháº¥t **150 box**.
- Má»—i lá»›p trong táº­p test nÃªn cÃ³ tá»‘i thiá»ƒu **30â€“50 box**.
- **10â€“20% áº£nh Ã¢m tÃ­nh tháº­t:** BÃ n trá»‘ng, chá»‰ cÃ³ tay, hoáº·c cÃ¡c váº­t gÃ¢y nháº§m.

### 2.3. CÃ¡c tÃ¬nh huá»‘ng báº¯t buá»™c pháº£i chá»¥p thÃªm
- Há»™p sáº¡c á»Ÿ cáº£ hai tráº¡ng thÃ¡i: **má»Ÿ náº¯p** vÃ  **Ä‘Ã³ng náº¯p**, xoay á»Ÿ nhiá»u gÃ³c khÃ¡c nhau.
- CÃ¡c tráº¡ng thÃ¡i khe cáº¯m: **má»™t khe trá»‘ng**, **hai khe trá»‘ng**, vÃ  **khÃ´ng cÃ²n khe trá»‘ng nÃ o** (Ä‘Ã£ cáº¯m Ä‘á»§ hai tai).
- Tráº¡ng thÃ¡i tai nghe: Má»™t tai nghe riÃªng láº» vÃ  hai tai nghe cÃ¹ng xuáº¥t hiá»‡n.
- Tay ngÆ°á»i thao tÃ¡c: Äang cáº§m, Ä‘ang che má»™t pháº§n (20â€“40% váº­t thá»ƒ).
- Vá»‹ trÃ­ thao tÃ¡c: Tai nghe náº±m ngoÃ i há»™p, náº±m gáº§n miá»‡ng há»™p, vÃ  Ä‘ang Ä‘Æ°á»£c Ä‘áº·t dá»Ÿ vÃ o khe.
- Äa dáº¡ng mÃ´i trÆ°á»ng: Thay Ä‘á»•i Ã¡nh sÃ¡ng (sÃ¡ng/tá»‘i hÆ¡n), ná»n bÃ n, khoáº£ng cÃ¡ch camera vÃ  ngÆ°á»i thao tÃ¡c khÃ¡c nhau.
- Váº­t gÃ¢y nháº§m láº«n (negative distractors): Chuá»™t mÃ¡y tÃ­nh, há»™p nhá», cá»§ sáº¡c Ä‘iá»‡n thoáº¡i, tai nghe cÃ³ dÃ¢y.

---

## 3. Kiá»ƒm tra cáº¥u hÃ¬nh vÃ  Tráº¡ng thÃ¡i sá»­a lá»—i tÃ­ch há»£p

### 3.1. Káº¿t quáº£ kiá»ƒm tra FSM
Ba file cáº¥u hÃ¬nh JSON trong thÆ° má»¥c `configs/` Ä‘á»u há»£p lá»‡ cÃº phÃ¡p. Logic mÃ¡y tráº¡ng thÃ¡i FSM Ä‘Ã£ Ä‘Æ°á»£c kiá»ƒm thá»­ tá»± Ä‘á»™ng vÃ  cháº¡y chÃ­nh xÃ¡c:
```text
pick_case â†’ insert_earbud â†’ close_case â†’ COMPLETED
```
FSM báº¯t lá»—i thÃ nh cÃ´ng: Náº¿u ngÆ°á»i dÃ¹ng thá»±c hiá»‡n `insert_earbud` trÆ°á»›c khi cÃ³ `pick_case`, há»‡ thá»‘ng láº­p tá»©c bÃ¡o lá»—i `VIOLATION`: *"ChÆ°a Ä‘áº·t há»™p sáº¡c vÃ o vÃ¹ng láº¯p mÃ  Ä‘Ã£ thao tÃ¡c tai nghe."*

### 3.2. Báº£ng theo dÃµi cÃ¡c Ä‘iá»ƒm tÃ­ch há»£p há»‡ thá»‘ng

| Váº¥n Ä‘á» tÃ­ch há»£p phÃ¡t hiá»‡n | Tráº¡ng thÃ¡i ká»¹ thuáº­t | HÆ°á»›ng dáº«n & Giáº£i phÃ¡p Ä‘Ã£ thá»±c hiá»‡n |
|---|---|---|
| **ThÆ° má»¥c sai tÃªn `earbub_parts`** | **ÄÃƒ Sá»¬A** | ÄÃ£ Ä‘á»•i tÃªn thÃ nh `datasets/earbud_parts` vÃ  cáº­p nháº­t file [data.yaml](file:///E:/Professional%20documents/Internship/RBCNN_Demo/datasets/earbud_parts/data.yaml). Validator kiá»ƒm tra Ä‘áº¡t `Trainable: YES` |
| **Checkpoint `earbud_detector`** | **ÄÃƒ Sáº´N SÃ€NG** | Checkpoint `artifacts/training/earbud_detector/weights/best.pt` Ä‘Ã£ cÃ³ sáºµn tá»« láº§n train baseline cá»§a báº¡n, sáºµn sÃ ng cháº¡y camera |
| **Script train máº·c Ä‘á»‹nh tÃªn cÅ©** | **ÄÃƒ Sá»¬A** | [train_detector.py](file:///E:/Professional%20documents/Internship/RBCNN_Demo/scripts/train_detector.py) giá» tá»± Ä‘á»™ng suy luáº­n: náº¿u `--data` lÃ  earbud thÃ¬ tá»± Ä‘á»™ng lÆ°u vÃ o `earbud_detector` |
| **PhÃ­m camera hardcode láº¯p bÃºt** | **ÄÃƒ Sá»¬A** | [camera_app.py](file:///E:/Professional%20documents/Internship/RBCNN_Demo/src/assembly/camera_app.py) Ä‘Ã£ chuyá»ƒn sang phÃ­m Ä‘á»™ng: vá»›i tai nghe, phÃ­m `1` lÃ  `pick_case`, phÃ­m `2` lÃ  `insert_earbud`, phÃ­m `3` lÃ  `close_case` |
| **Script quay video máº·c Ä‘á»‹nh bÃºt** | **ÄÃƒ Sá»¬A** | [record_assembly_videos.py](file:///E:/Professional%20documents/Internship/RBCNN_Demo/scripts/record_assembly_videos.py) Ä‘Ã£ há»— trá»£ `--project earbud` vÃ  cho phÃ©p Ä‘áº·t tÃªn ká»‹ch báº£n (`--scenario`) tá»± do |
| **ChÆ°a kiá»ƒm tra containment hÃ¬nh há»c** | **ÄÃƒ Sá»¬A Ná»€N Táº¢NG** | Class `Detection` trong [vision.py](file:///E:/Professional%20documents/Internship/RBCNN_Demo/src/assembly/vision.py) Ä‘Ã£ cÃ³ sáºµn cÃ¡c phÆ°Æ¡ng thá»©c `is_inside`, `overlap_ratio_with`, `contains_point` |
| **`close_case` chÆ°a cÃ³ detection class** | **Cáº¦N CHá»T GIáº¢I PHÃP** | Xem phÃ¢n tÃ­ch chi tiáº¿t táº¡i Má»¥c 3.3 bÃªn dÆ°á»›i |
| **`Empty_Slot` cÃ³ `action: null`** | **Cáº¦N CHá»T GIáº¢I PHÃP** | Xem phÃ¢n tÃ­ch chi tiáº¿t táº¡i Má»¥c 3.4 bÃªn dÆ°á»›i |
| **Cáº§n video temporal cho BiLSTM** | **Báº N Cáº¦N LÃ€M** | Xem hÆ°á»›ng dáº«n quay video táº¡i Má»¥c 6 |

### 3.3. Xá»­ lÃ½ hÃ nh Ä‘á»™ng `close_case`
Hiá»‡n táº¡i máº¡ng detector chá»‰ nháº­n diá»‡n 3 lá»›p: `Case`, `Earbud`, `Empty_Slot`. Khi náº¯p há»™p Ä‘Ã³ng láº¡i, váº­t thá»ƒ váº«n lÃ  `Case`. Do Ä‘Ã³ má»™t detector áº£nh tÄ©nh khÃ´ng thá»ƒ tá»± phÃ¢n biá»‡t Ä‘Æ°á»£c giá»¯a viá»‡c "há»™p Ä‘ang má»Ÿ" vÃ  "hÃ nh Ä‘á»™ng Ä‘Ã³ng náº¯p vá»«a xáº£y ra".

CÃ¡c giáº£i phÃ¡p:
1. **DÃ¹ng phÃ­m xÃ¡c nháº­n (Hiá»‡n táº¡i):** Sau khi láº¯p xong tai nghe, ngÆ°á»i dÃ¹ng nháº¥n phÃ­m `3` hoáº·c `SPACE` Ä‘á»ƒ xÃ¡c nháº­n Ä‘Ã³ng náº¯p hoÃ n táº¥t.
2. **DÃ¹ng mÃ´ hÃ¬nh Temporal ViT + BiLSTM:** Quay video hÃ nh Ä‘á»™ng Ä‘Ã³ng náº¯p, máº¡ng BiLSTM sáº½ nháº­n diá»‡n cá»­ chá»‰ gáº­p náº¯p theo chuá»—i thá»i gian vÃ  tá»± Ä‘á»™ng phÃ¡t event `close_case`.
3. **Má»Ÿ rá»™ng class Detector (KhuyÃªn dÃ¹ng khi gÃ¡n nhÃ£n láº¡i):** Thay vÃ¬ chá»‰ má»™t nhÃ£n `Case`, gÃ¡n thÃ nh:
   - `Case_Open`: Há»™p sáº¡c Ä‘ang má»Ÿ náº¯p.
   - `Case_Closed`: Há»™p sáº¡c Ä‘Ã£ Ä‘Ã³ng náº¯p.  
   Khi `Case_Closed` xuáº¥t hiá»‡n trong WORK ZONE, detector cÃ³ thá»ƒ kÃ­ch hoáº¡t trá»±c tiáº¿p `close_case`!

### 3.4. Táº­n dá»¥ng lá»›p `Empty_Slot`
Lá»›p `Empty_Slot` hiá»‡n cÃ³ `"action": null` (chá»‰ hiá»ƒn thá»‹ box xanh dÆ°Æ¡ng). Äá»ƒ Ä‘Æ°a vÃ o logic kiá»ƒm tra:
- **NguyÃªn lÃ½:** Khi há»™p sáº¡c má»Ÿ ra, ban Ä‘áº§u sáº½ cÃ³ 2 khe trá»‘ng (`Empty_Slot = 2`). Khi cáº¯m tai nghe vÃ o, khe trá»‘ng bá»‹ che khuáº¥t vÃ  biáº¿n máº¥t.
- Báº±ng phÆ°Æ¡ng thá»©c `is_inside(case)` vá»«a bá»• sung, há»‡ thá»‘ng cÃ³ thá»ƒ Ä‘áº¿m sá»‘ `Empty_Slot` náº±m trong `Case`. Khi sá»‘ khe trá»‘ng giáº£m tá»« 2 vá» 0, há»‡ thá»‘ng tá»± Ä‘á»™ng xÃ¡c nháº­n hoÃ n thÃ nh bÆ°á»›c láº¯p tai nghe.

---

## 4. Quyáº¿t Ä‘á»‹nh thiáº¿t káº¿ báº¡n cáº§n chá»‘t

TrÆ°á»›c khi cáº¥u trÃºc láº¡i FSM vÃ  gÃ¡n nhÃ£n chi tiáº¿t, báº¡n cáº§n tráº£ lá»i cÃ¢u há»i cá»‘t lÃµi sau:

> [!IMPORTANT]
> **Quy trÃ¬nh cá»§a báº¡n cáº§n láº¯p Má»˜T tai nghe hay pháº£i kiá»ƒm tra Ä‘á»§ Cáº¢ HAI tai nghe (tai trÃ¡i vÃ  tai pháº£i)?**

### Lá»±a chá»n A: Quy trÃ¬nh láº¯p 1 tai nghe (Hiá»‡n táº¡i)
- **Chu trÃ¬nh:** `pick_case` -> `insert_earbud` -> `close_case` -> HoÃ n táº¥t.
- **Æ¯u Ä‘iá»ƒm:** ÄÆ¡n giáº£n, FSM hiá»‡n táº¡i trong [earbud_fsm_config.json](file:///E:/Professional%20documents/Internship/RBCNN_Demo/configs/earbud_fsm_config.json) giá»¯ nguyÃªn vÃ  cháº¡y ngay láº­p tá»©c.
- **PhÃ¹ há»£p:** LÃ m demo ban Ä‘áº§u, kiá»ƒm thá»­ nhanh kháº£ nÄƒng nháº­n diá»‡n.

### Lá»±a chá»n B: Quy trÃ¬nh kiá»ƒm tra Ä‘á»§ cáº£ 2 tai nghe (Chuáº©n cÃ´ng nghiá»‡p)
- **Chu trÃ¬nh:** `pick_case` -> `insert_earbud_1` -> `insert_earbud_2` -> `close_case` -> HoÃ n táº¥t.
- **YÃªu cáº§u:** 
  - FSM cáº§n thÃªm má»™t tráº¡ng thÃ¡i trung gian (`S2_FIRST_EARBUD_INSERTED` vÃ  `S3_SECOND_EARBUD_INSERTED`).
  - Hoáº·c FSM giá»¯ nguyÃªn action `insert_earbud` nhÆ°ng yÃªu cáº§u xuáº¥t hiá»‡n 2 láº§n liÃªn tiáº¿p trÆ°á»›c khi cho phÃ©p `close_case`.
  - Náº¿u phÃ¢n biá»‡t khe TrÃ¡i / khe Pháº£i: Cáº§n Ä‘áº·t nhÃ£n `Left_Slot`, `Right_Slot` hoáº·c `Earbud_L`, `Earbud_R`.

*Báº¡n hÃ£y chá»n PhÆ°Æ¡ng Ã¡n A hay B Ä‘á»ƒ mÃ¬nh tinh chá»‰nh cáº¥u hÃ¬nh FSM tÆ°Æ¡ng á»©ng.*

---

## 5. HÆ°á»›ng dáº«n cháº¡y thá»­ nghiá»‡m Há»‡ thá»‘ng Tai nghe ngay hÃ´m nay

Báº¡n cÃ³ thá»ƒ cháº¡y thá»­ há»‡ thá»‘ng thá»i gian thá»±c vá»›i checkpoint baseline hiá»‡n táº¡i:

```powershell
cd "E:\Professional documents\Internship\RBCNN_Demo"

# Khá»Ÿi cháº¡y camera thá»i gian thá»±c vá»›i cáº¥u hÃ¬nh tai nghe:
python scripts/run_camera.py `
  --source 0 `
  --camera-config configs/camera_earbud_config.json `
  --fsm-config configs/earbud_fsm_config.json
```

### CÃ¡c phÃ­m Ä‘iá»u khiá»ƒn trong cá»­a sá»• camera:
- `PhÃ­m 1`: MÃ´ phá»ng hÃ nh Ä‘á»™ng **`pick_case`** (Äáº·t há»™p sáº¡c).
- `PhÃ­m 2`: MÃ´ phá»ng hÃ nh Ä‘á»™ng **`insert_earbud`** (Láº¯p tai nghe).
- `PhÃ­m 3`: MÃ´ phá»ng hÃ nh Ä‘á»™ng **`close_case`** (ÄÃ³ng náº¯p há»™p).
- `PhÃ­m SPACE`: XÃ¡c nháº­n gá»£i Ã½ tá»± Ä‘á»™ng khi detector phÃ¡t hiá»‡n váº­t thá»ƒ náº±m á»•n Ä‘á»‹nh trong vÃ¹ng WORK ZONE.
- `PhÃ­m C`: Chuyá»ƒn Ä‘á»•i giá»¯a cÃ¡c camera káº¿t ná»‘i (webcam laptop / camera USB ngoÃ i).
- `PhÃ­m R`: Reset chu trÃ¬nh vá» tráº¡ng thÃ¡i ban Ä‘áº§u `S0_IDLE`.
- `PhÃ­m S`: LÆ°u áº£nh chá»¥p mÃ n hÃ¬nh vÃ o `artifacts/screenshots/`.
- `PhÃ­m Q` hoáº·c `Esc`: ThoÃ¡t chÆ°Æ¡ng trÃ¬nh.

### Cháº¡y kiá»ƒm thá»­ tá»± Ä‘á»™ng toÃ n bá»™ test case:
```powershell
python -m unittest discover tests
```
*(Káº¿t quáº£ hiá»‡n táº¡i: ToÃ n bá»™ 41 unit test Ä‘á»u cháº¡y thÃ nh cÃ´ng).*

---

## 6. HÆ°á»›ng dáº«n quay video hÃ nh Ä‘á»™ng cho Tai nghe (ViT + BiLSTM)

Náº¿u báº¡n muá»‘n há»‡ thá»‘ng tá»± Ä‘á»™ng nháº­n diá»‡n hÃ nh Ä‘á»™ng láº¯p rÃ¡p (nhÆ° cáº¯m tai nghe, gáº­p náº¯p há»™p) mÃ  khÃ´ng cáº§n nháº¥n phÃ­m:

### 6.1. Lá»‡nh quay má»™t session tai nghe
```powershell
# Quay ká»‹ch báº£n láº¯p Ä‘Ãºng (correct):
python scripts/record_assembly_videos.py `
  --source 0 `
  --person person01 `
  --session session01 `
  --project earbud `
  --scenario correct
```

Video sáº½ tá»± Ä‘á»™ng Ä‘Æ°á»£c lÆ°u vÃ o: `data/earbud_actions/raw_videos/person01/session01/`.

### 6.2. Quay cÃ¡c ká»‹ch báº£n lá»—i Ä‘á»ƒ kiá»ƒm thá»­ cáº£nh bÃ¡o FSM:
```powershell
# Ká»‹ch báº£n quÃªn láº¯p tai nghe mÃ  Ä‘Ã£ Ä‘Ã³ng há»™p:
python scripts/record_assembly_videos.py --source 0 --person person01 --session session02 --project earbud --scenario missing_earbud

# Ká»‹ch báº£n Ä‘Ã³ng náº¯p quÃ¡ sá»›m:
python scripts/record_assembly_videos.py --source 0 --person person01 --session session03 --project earbud --scenario close_case_early

# Ká»‹ch báº£n lÃ m sai thá»© tá»± (cáº§m tai nghe trÆ°á»›c khi Ä‘áº·t há»™p):
python scripts/record_assembly_videos.py --source 0 --person person01 --session session04 --project earbud --scenario wrong_order
```

---

## 7. Checklist viá»‡c báº¡n nÃªn Æ°u tiÃªn lÃ m ngay

Äá»ƒ hoÃ n thiá»‡n há»‡ thá»‘ng, báº¡n hÃ£y thá»±c hiá»‡n theo thá»© tá»± sau:

- [ ] **Viá»‡c 1: Chá»‘t thiáº¿t káº¿ FSM:** BÃ¡o cho mÃ¬nh biáº¿t báº¡n muá»‘n quy trÃ¬nh kiá»ƒm tra 1 tai nghe hay Ä‘á»§ cáº£ 2 tai nghe (trÃ¡i/pháº£i).
- [ ] **Viá»‡c 2: Sá»­a 56 label rá»—ng & xÃ³a 3 box lá»—i:** Má»Ÿ táº­p dá»¯ liá»‡u `datasets/earbud_parts` trÃªn cÃ´ng cá»¥ gÃ¡n nhÃ£n, kiá»ƒm tra 56 file label rá»—ng vÃ  gÃ¡n box cho cÃ¡c áº£nh tháº¥y rÃµ há»™p sáº¡c / tai nghe / khe trá»‘ng; xÃ³a 3 box 1-6 pixel.
- [ ] **Viá»‡c 3: Khá»­ rÃ² rá»‰ dá»¯ liá»‡u:** Chuyá»ƒn toÃ n bá»™ 151 áº£nh hiá»‡n táº¡i vÃ o táº­p `train`.
- [ ] **Viá»‡c 4: Quay session má»›i cho Val vÃ  Test:** Quay 1 session Ä‘á»™c láº­p cho `val` (khoáº£ng 50 áº£nh) vÃ  1 session Ä‘á»™c láº­p cho `test` (khoáº£ng 50 áº£nh) tá»« gÃ³c quay / ngÆ°á»i thao tÃ¡c khÃ¡c.
- [ ] **Viá»‡c 5: Kiá»ƒm tra vÃ  train láº¡i:** Cháº¡y `python scripts/validate_detection_dataset.py --data datasets/earbud_parts/data.yaml`, sau Ä‘Ã³ cháº¡y `python scripts/train_detector.py --data datasets/earbud_parts/data.yaml --epochs 60` Ä‘á»ƒ cÃ³ checkpoint sáº¡ch má»›i.

