import os
import torch
import psutil
from architecture import DEVICE

CHECKPOINT_DIR = "model/checkpoints"
WEIGHTS_DIR = "model/weights"

def get_system_stats():
    ram = psutil.virtual_memory().total / (1024**3)
    vram = 0
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif 'xpu' in str(DEVICE):
        vram = torch.xpu.get_device_properties(0).total_memory / (1024**3)
    return ram, vram

def auto_calculate_params(model_mode, total_tokens):
    ram, vram = get_system_stats()
    available_mem = vram if vram > 0 else ram * 0.5
    
    if 'gpt2' in model_mode:
        batch_size = max(1, int(available_mem / 4))
        block_size = 1024
    else:
        batch_size = max(2, int(available_mem / 1))
        block_size = 256
        
    steps = max(100, int(total_tokens / (batch_size * block_size) * 2))
    return steps, batch_size, block_size

def list_models():
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
    weights = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pth")]
    return ckpts, weights

def delete_model(name):
    for d in [CHECKPOINT_DIR, WEIGHTS_DIR]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            os.remove(p)
            return f"🗑️ Deleted {name}"
    return "❌ Not found"