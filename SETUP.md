# BadGPT Setup Guide

## Not tested or verified


Welcome to BadGPT. This guide will help you set up the environment, including the tricky OCR tools (Tesseract & Poppler) needed for scanning image-based PDFs. (Not Tested)

---

## 🐍 1. Python Dependencies

First, install the Python libraries.

```bash
pip install -r requirements.txt
```

## 👁️ 1. System Dependencies (OCR Tools)

The OCR script (`create_dataset_ocr.py`) requires **Tesseract** (the text reader) and **Poppler** (the PDF-to-Image converter) to be installed on your OS.

### 🐧 Linux

#### **Arch Linux (Manjaro, EndeavourOS)**

```bash
sudo pacman -S tesseract tesseract-data-eng poppler
```

#### **Ubuntu / Debian / Kali**

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng poppler-utils
```

#### **Fedora / RHEL**

```bash
sudo dnf install tesseract tesseract-langpack-eng poppler-utils
```

---

### 🪟 Windows (The Tricky Part)

Windows does not have a simple package manager. You must install these manually and **add them to your PATH**.

#### **Step A: Install Tesseract OCR**

1. Download the installer from [UB-Mannheim Tesseract](https://www.google.com/search?q=https://github.com/UB-Mannheim/tesseract/wiki).
2. Run the `.exe` installer.
3. **Important:** Note the install path (usually `C:\Program Files\Tesseract-OCR`).
4. **Add to PATH:**
* Press `Win` key, type "Edit the system environment variables".
* Click **Environment Variables**.
* Under **System variables**, find `Path` -> Edit -> New.
* Paste: `C:\Program Files\Tesseract-OCR`
* Click OK.



#### **Step B: Install Poppler**

1. Download the latest Release ZIP from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/).
2. Extract the ZIP to a permanent folder (e.g., `C:\Program Files\poppler`).
3. **Add to PATH:**
* Go back to Environment Variables -> Path -> Edit -> New.
* Paste the `bin` folder path: `C:\Program Files\poppler\Library\bin` (or wherever the `pdftoppm.exe` file is located).
* Click OK.



**Verify Installation:**
Open a **new** Command Prompt or PowerShell and type:

```cmd
tesseract --version
pdftoppm -h
```

If both commands output version info, you are ready!

---

## 📂 3. How to Run

1. **Prepare Data:**
* Put your PDFs in `data/raw/`.
* Run `python create_dataset.py` (for normal PDFs).
* Run `python create_dataset_ocr.py` (for scanned/image PDFs).


2. **Train:**
* Run `python train.py`.
* Models are saved to `model/checkpoints/`.


3. **Chat:**
* Convert your heavy checkpoint: `python convert_checkpoint.py`.
* Launch the chat: `python inference.py`.

# Hey hey. Respect Copyright dude.