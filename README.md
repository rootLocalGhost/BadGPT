# BadGPT

## The Dumbest AI Possible

BadGPT is the dumbest AI you will ever meet. It knows nothing, hallucinates everything, and has the memory of a goldfish. But hey, you can steal this project and maybe learn how LLMs actually work without selling your kidney for an H100.

Try it. I don't know why I kept it under the MIT [LICENSE](LICENSE).

---

## 💻 The Rig

This project is battle-tested on the underdog hardware:

- **GPU:** Intel ARC A770 (16GB VRAM) 🚀
- **RAM:** 16 GB System Memory (Living on the edge)
- **OS:** The **BTW! Operating System**

---

## 🛠️ Setup Guide

Here is how to set it up. Don't mess it up.

### 1. Conda Setup

Install miniconda or miniforge prompt. Do it. I won't tell you how. Google is free.

### 2. Create the Environment

We need Python 3.11 because we like stability (sometimes).

```bash
conda create -n BadGPT python=3.11
conda activate BadGPT
```

### 3. Install the "Intel Nightmare" (Torch + IPEX)
### If you're a NVIDIA or AMD guy, see the [Rich People's Guide](RichGuide.md)

Depending on your rig, run **ONE** of the following blocks. Don't run all of them or your PC might explode.

#### 🪟 Windows (Intel GPU / XPU)

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu

python -m pip install intel-extension-for-pytorch==2.8.10+xpu --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

#### 🐧 Linux / WSL (Intel GPU / XPU)

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu

python -m pip install intel-extension-for-pytorch==2.8.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

python -m pip install oneccl_bind_pt==2.8.0+xpu --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

#### 🐌 Linux / WSL (Intel CPU Only)

```bash
python -m pip install torch==2.8.0+cpu torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

python -m pip install intel-extension-for-pytorch==2.8.0
```

_(I don't know if it works on Windows for CPU. Intel didn't say anything on their website. Good luck.)_

### 4. Test Your Sanity

Run this one-liner. If it prints garbage errors, you messed up Step 3.

**For GPU Users:**

```bash
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
```

**For CPU Peasants:**

```bash
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__);"
```

### 5. Install the Rest

Get the other junk needed for the interface and OCR.

```bash
pip install -r requirements.txt
```

---

## 🧹 System Dependencies (For OCR)

If you plan to feed this thing scanned PDFs (images), you need **Tesseract** and **Poppler**.

- **Arch Linux (BTW):** `sudo pacman -S tesseract tesseract-data-eng poppler`
- **Ubuntu:** `sudo apt install tesseract-ocr tesseract-ocr-eng poppler-utils`
- **Windows:** Download the installers manually and add them to your PATH. You know the drill.

---

## 🏃 How to Run (The Fun Part)

The project is modular now because putting everything in one file was driving me crazy.

### Phase 1: Feed the Beast 📄

Put your PDFs inside `BadGPT/data/raw/`.

- **Clean PDFs:** Run `python create_dataset.py`
- **Scanned/Image PDFs:** Run `python create_dataset_ocr.py` (Go get a coffee, this takes forever).

### Phase 2: Torture the GPU (Training) 🔥

Start the training loop. It has autosave so you can rage-quit anytime.

```bash
python train.py

```

- Checkpoints save to `model/checkpoints/` (~600MB each).

### Phase 3: Put it on a Diet 📉

The checkpoints are fat because they contain optimizer states. Convert them to "Lite" weights (~150MB) for chatting.

```bash
python convert_checkpoint.py

```

### Phase 4: Talk to it 💬

Launch the web interface and realize how bad it actually is.

```bash
python inference.py

```

---

## 📂 Project Structure

```text
BadGPT/
├── data/
│   ├── raw/           <-- Put PDFs here
│   └── token/         <-- Text files appear here
├── model/
│   ├── checkpoints/   <-- Heavy training files (.pt)
│   └── weights/       <-- Lite chat files (.pth)
├── badgpt.py          <-- The Brain (Model Arch)
├── train.py           <-- The Gym
├── inference.py       <-- The Chat Room
└── README.md          <-- You are here

```

Enjoy your dumb AI. 🤡
