# BadGPT: The "I Can't Believe It's Not Smarter" AI

Welcome to BadGPT. It's the AI equivalent of a goldfish with a calculator. Now updated to support literally every piece of hardware you own, from your $2000 RTX 4090 to that Intel Arc your cousin gave you for free.

## 🚀 Features

- **Native Everything**: CUDA, XPU (Intel), DirectML (AMD/Windows), or your sad CPU.
- **Auto-Tune**: A button that looks at your VRAM and decides how hard to push your GPU so you don't smell smoke.
- **Gym Mode**: Real-time graphs for VRAM and Loss. Watch the AI fail in 4K.
- **OCR Tab**: Feed it pictures of books because typing is for nerds.

## 🛠 Setup

1. `pip install torch torchvision torchaudio transformers gradio pdfplumber pytesseract pdf2image psutil tiktoken`
2. If on AMD: `pip install torch-directml`
3. If on Intel: `pip install intel-extension-for-pytorch` (Optional, 2.4+ has native XPU)
4. Launch the chaos: `python badgpt.py`

## 🤡 Parameter Guide

- **Steps**: How many times the AI reads the book. More = smarter (usually).
- **Batch Size**: How many sentences it reads at once. High = Fast, Low = Safe for your GPU.
- **Block Size**: Memory span. 256 is "What happened 5 minutes ago?", 1024 is "I remember lunch."
- **Learning Rate**: 1e-4 is "Stable", 1e-2 is "I have a fever and I'm hallucinating."

# Respect the code. Don't sue me when it starts talking about beans for no reason.
