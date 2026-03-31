# OCR Augmentation Starter for VS Code + WSL

This starter project helps you do three things:

1. **Create a small word-recognition dataset** (synthetic text images)
2. **Apply OCR-friendly image augmentation**
3. **Train a PaddleOCR recognition model** using a clean folder structure

It is designed for **Ubuntu on WSL** and can be opened directly in **VS Code**.

---

## 1) What this project contains

```text
ocr_aug_starter/
├── README.md
├── requirements.txt
├── configs/
│   └── rec_aug_train.yml
├── data/
│   └── words/
│       └── words.txt
├── scripts/
│   ├── setup_wsl.sh
│   ├── create_dataset.py
│   ├── inspect_samples.py
│   └── train_paddleocr.sh
└── src/
    ├── augment.py
    └── dataset_utils.py
```

---

## 2) Before you start

You need:

- Windows with **WSL** enabled
- Ubuntu installed in WSL
- VS Code with the **WSL** extension
- Python 3.10+ available inside Ubuntu

PaddleOCR’s official installation docs say to install **PaddlePaddle first**, then install PaddleOCR. The docs also show pip-based installation, and the main OCR docs note that `paddleocr[all]` is available for package installation. citeturn104024search1turn104024search10

---

## 3) Step-by-step: open WSL and create the environment

### A. Open Ubuntu (WSL)

On Windows:

- Press the Windows key
- Type **Ubuntu**
- Open it

Or from PowerShell:

```bash
wsl
```

### B. Go to your working folder

```bash
mkdir -p ~/projects
cd ~/projects
```

### C. Copy or unzip this project there

If you downloaded this folder as a zip, unzip it and then:

```bash
cd ~/projects/ocr_aug_starter
```

### D. Create a Python virtual environment

```bash
python3 -m venv .venv
```

### E. Activate the environment

```bash
source .venv/bin/activate
```

When activated, your terminal usually shows `(.venv)` at the start.

### F. Upgrade pip

```bash
python -m pip install --upgrade pip wheel setuptools
```

### G. Install Python packages for the starter project

```bash
pip install -r requirements.txt
```

### H. Install PaddlePaddle and PaddleOCR

For **CPU only** training, use the commands from the official PaddleOCR / Paddle docs as your reference and install the PaddlePaddle build that matches your machine. The PaddleOCR installation docs explicitly say PaddlePaddle should be installed first, and PaddleOCR supports pip installation. citeturn104024search1turn104024search10

Example CPU install:

```bash
python -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install "paddleocr[all]"
```

If you have a CUDA GPU available **inside WSL** and it is configured correctly, install the GPU build that matches your CUDA version from the official PaddlePaddle package index instead of the CPU build. Check the official install page for the matching command. citeturn104024search1

---

## 4) Open the folder in VS Code

From inside WSL, in the project root:

```bash
code .
```

This opens the current WSL folder in VS Code.

---

## 5) Create a toy OCR dataset

This project includes a word list in `data/words/words.txt` and a script that renders text images using PIL.

Run:

```bash
python scripts/create_dataset.py \
  --output_dir ./generated_dataset \
  --num_train 2000 \
  --num_val 300 \
  --image_height 32 \
  --min_width 96 \
  --max_width 256
```

This creates:

```text
generated_dataset/
├── train/
│   ├── images/
│   └── train.txt
└── val/
    ├── images/
    └── val.txt
```

Each line in `train.txt` and `val.txt` is:

```text
relative/path/to/image.png<TAB>TEXT_LABEL
```

That is the label format expected by PaddleOCR recognition training for simple datasets.

---

## 6) Inspect the generated examples

```bash
python scripts/inspect_samples.py --dataset_dir ./generated_dataset --count 12
```

This saves a contact sheet image so you can visually check whether the text is still readable after augmentation.

---

## 7) Train with PaddleOCR

There are two ways to use the config.

### Option A: Use this config template directly from a cloned PaddleOCR repo

Clone PaddleOCR if you want full training from source:

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
```

The official repo contains `ppocr/data/imaug/rec_img_aug.py`, where the built-in recognition augmenters such as `RecAug`, `RecConAug`, `SVTRRecAug`, and `ABINetRecAug` are defined. citeturn104024search0

Copy `configs/rec_aug_train.yml` from this starter project into the PaddleOCR repo and update the dataset paths if needed.

Then train:

```bash
python tools/train.py -c /full/path/to/ocr_aug_starter/configs/rec_aug_train.yml
```

### Option B: Use the helper script

From this project root:

```bash
bash scripts/train_paddleocr.sh /full/path/to/PaddleOCR ./generated_dataset
```

---

## 8) What augmentation is being used

This project gives you **two layers** of augmentation:

### Layer 1: custom Python augmentation in `src/augment.py`

This is useful for understanding the idea clearly. It applies:

- mild rotation
- perspective warp
- horizontal stretch
- Gaussian blur
- Gaussian noise
- brightness / contrast changes

These are OCR-friendly because they change **how the text looks**, not **what the text says**.

### Layer 2: PaddleOCR built-in augmentation

The training config enables `RecAug`, and you can optionally enable `RecConAug`. PaddleOCR’s source shows `RecAug` includes OCR-oriented transforms such as TIA-based distort/stretch/perspective and additional blur, HSV, jitter, and noise, while `RecConAug` can concatenate compatible samples and append labels when length and aspect constraints allow it. citeturn104024search0

The PaddleOCR SVTR docs also explicitly mention `RecConAug` and `RecAug` as optional data augmentation strategies for recognition. citeturn104024search4

---

## 9) Why this structure is good

You asked for a project you can use in VS Code and understand clearly. This setup is good because:

- the **dataset creation** is separate from training
- the **augmentation code** is isolated and easy to modify
- the **PaddleOCR config** is separate and easy to compare
- you can run experiments like:
  - no augmentation
  - custom augmentation only
  - PaddleOCR `RecAug`
  - `RecAug + RecConAug`

---

## 10) Very important rules

### Do:

- augment only the **image**, not the label
- keep validation data clean
- start with mild settings
- inspect examples visually

### Do not:

- rearrange characters in an existing labeled image
- use augmentation so strong that text becomes unreadable
- augment validation/test data

---

## 11) Files to edit first

If you want to customize things quickly, edit these:

- `data/words/words.txt` → word list used for synthetic labels
- `src/augment.py` → probabilities and augmentation strength
- `configs/rec_aug_train.yml` → PaddleOCR training settings
- `scripts/create_dataset.py` → dataset size and font behavior

---

## 12) How the code works, simply

### `scripts/create_dataset.py`
Creates synthetic word images from text labels, applies optional augmentation, and writes PaddleOCR label files.

### `src/augment.py`
Contains the OCR-specific image augmenter.

### `src/dataset_utils.py`
Contains helper functions for rendering text, choosing widths, and saving PaddleOCR-format labels.

### `configs/rec_aug_train.yml`
A ready-to-edit PaddleOCR recognition training config using `RecAug`.

---

## 13) Suggested next experiments

1. Train with no augmentation.
2. Train with custom augmentation only.
3. Train with PaddleOCR `RecAug`.
4. Train with `RecAug + RecConAug`.
5. Compare validation accuracy and error types.

That experiment design matches the purpose of OCR augmentation: making training images closer to the target distribution so recognition generalizes better. STRAug is a strong OCR-specific reference for this idea, and PaddleOCR already ships OCR-oriented augmenters in its recognition pipeline. citeturn104024search0turn104024search4

