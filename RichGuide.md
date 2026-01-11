# 💰 Rich Peoples Guide

## So, You Have Money?

Congratulations. You are not suffering with an Intel Arc or an integrated potato. You possess the hardware of the gods named NVIDIA RTX 4090s, AMD RX 7900 XTXs, or maybe a CPU with more cores than my entire vocabulary.

This guide is for you. Don't mess it up; your hardware is too expensive for errors.

---

## 🟢 The NVIDIA Elites (CUDA)

If you have a green sticker on your PC, you are living life on easy mode. PyTorch loves you.

### 1. Install PyTorch (The Normal Way)

You don't need weird "Intel Extensions." You just need raw power.

```bash
# Check your CUDA version first (usually 11.8 or 12.1)
# Run this: nvidia-smi

# For CUDA 12.1 (The latest and greatest, maybe)
# If this doesn't work then check the official pytorch site
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Verify You Are Powerful

Run this Python one-liner to flex your VRAM:

```python
import torch; print(f"🔥 GPU: {torch.cuda.get_device_name(0)}"); print(f"🧠 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB");
```

---

## 🔴 The AMD Rebels (ROCm)

You bought AMD because you hate monopolies (or you couldn't find an NVIDIA card). Respect. But setting up ROCm on Windows is... experimental. On Linux, it flies.

### 🐧 Linux (The Only Way that Matters)

ROCm is native here.

```bash
# Install PyTorch for ROCm 6.0 (check the website for your specific version)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/rocm6.0](https://download.pytorch.org/whl/rocm6.0)
```

### 🪟 Windows (DirectML - The "It Works" Way)

Official ROCm support on Windows is still beta for many things. The safest bet for broad compatibility is **DirectML**.

```bash
pip install torch-directml
```

_Note: You will need to change `device = 'cuda'` to `device = 'privateuseone'` or `import torch_directml; device = torch_directml.device()` in `badgpt.py`. Actually, just buy NVIDIA next time._

---

## 🔵 The CPU Workstations (Threadripper / Xeon)

You have 64 cores and 128GB of RAM. You don't need a GPU; you _are_ the render farm.

### 1. Install CPU-Optimized PyTorch

Standard PyTorch is fine, but let's make it faster.

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```

### 2. Force Multi-Threading

In `train.py` or `badgpt.py`, add this at the top to make sure all 128 threads scream in agony:

```python
import torch
torch.set_num_threads(64) # Set this to your physical core count
```

---

## ⚙️ Tweaking BadGPT for Rich People

Since you have the VRAM, stop training baby models. Go into `badgpt.py` and crank these numbers up until your lights flicker.

**In `badgpt.py`:**

```python
# The "I Have 24GB VRAM" Config
BLOCK_SIZE = 1024       # Context window (Was 256)
N_EMBD = 768            # Embedding dimension (Was 384)
N_HEAD = 12             # Attention heads (Was 6)
N_LAYER = 12            # Layers (Was 6)
DROPOUT = 0.2           # Add some dropout so it actually learns instead of cheating
```

**In `train.py`:**

```python
BATCH_SIZE = 32         # Crank this up. If you hit OOM, buy another GPU.
GRAD_ACCUM_STEPS = 1    # You don't need to fake batch sizes. You're rich.
```

Now go train. And if it's still slow, it's not the code. It's you.

# And remember my face. Respect Copyright.