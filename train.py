import torch
import tiktoken
import os
import time
import pandas as pd
import gc
import glob
from architecture import GPTModel, DEVICE, BLOCK_SIZE, N_EMBD

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
INITIAL_LR = 1e-3
CHECKPOINT_DIR = "model/checkpoints"
DATA_DIR = "data/token"

enc = tiktoken.get_encoding("cl100k_base")

def load_dataset():
    files = glob.glob(f"{DATA_DIR}/*.txt")
    if not files: return None, "❌ No .txt files in data/token"
    
    full_text_list = []
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            full_text_list.append(file.read())
            full_text_list.append("\n<EOF>\n")
            
    full_text = "".join(full_text_list)
    data = torch.tensor(enc.encode(full_text), dtype=torch.long)
    return data, f"✅ Loaded {len(data)} tokens from {len(files)} files."

def train_generator(steps_to_add, autosave_freq):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    if DEVICE == 'xpu': torch.xpu.empty_cache()
    gc.collect()
    
    train_data, msg = load_dataset()
    if train_data is None:
        yield msg, None, None
        return

    model = GPTModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    ckpts = glob.glob(f"{CHECKPOINT_DIR}/*.pt")
    start_step = 0
    loss_history = []
    lr_history = []
    
    if ckpts:
        latest = max(ckpts, key=os.path.getmtime)
        try:
            checkpoint = torch.load(latest, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_step = checkpoint.get('total_steps', 0)
            loss_history = checkpoint.get('loss_history', [])
            lr_history = checkpoint.get('lr_history', [])
            yield f"♻️ Resumed from {latest}", pd.DataFrame(loss_history, columns=["Step", "Loss"]), pd.DataFrame(lr_history, columns=["Step", "LR"])
        except Exception as e:
            yield f"⚠️ Corrupt checkpoint {latest}, starting fresh. ({e})", None, None

    if DEVICE == 'xpu':
        import intel_extension_for_pytorch as ipex
        model, optimizer = ipex.optimize(model, optimizer=optimizer)

    model.train()
    target_step = start_step + int(steps_to_add)
    optimizer.zero_grad(set_to_none=True)
    
    t0 = time.time()
    yield f"🚀 Training from step {start_step} to {target_step}...", None, None

    df_loss = None
    df_lr = None

    for i in range(start_step, target_step):
        ix = torch.randint(len(train_data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([train_data[k:k+BLOCK_SIZE] for k in ix]).to(DEVICE)
        y = torch.stack([train_data[k+1:k+BLOCK_SIZE+1] for k in ix]).to(DEVICE)
        
        logits, loss = model(x, y)
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()
        
        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if DEVICE == 'xpu': torch.xpu.synchronize()
            
            current_loss = loss.item() * GRAD_ACCUM_STEPS
            scheduler.step(current_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # UI Update (Every 5 steps)
            if (i + 1) % 5 == 0:
                dt = time.time() - t0
                t0 = time.time()
                speed = (BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE) / (dt + 1e-9)
                loss_history.append([i+1, current_loss])
                lr_history.append([i+1, current_lr])
                
                df_loss = pd.DataFrame(loss_history, columns=["Step", "Loss"])
                df_lr = pd.DataFrame(lr_history, columns=["Step", "LR"])
                
                log_msg = f"Step {i+1} | Loss: {current_loss:.4f} | LR: {current_lr:.6f} | {speed:.0f} tok/s"
                yield log_msg, df_loss, df_lr
        
        # Save Checkpoint Logic: Autosave freq OR Final Step
        if (i + 1) % autosave_freq == 0 or (i + 1) == target_step:
            ckpt_path = f"{CHECKPOINT_DIR}/ckpt_step_{i+1}.pt"
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss_history': loss_history,
                'lr_history': lr_history,
                'total_steps': i+1
            }, ckpt_path)
            yield f"💾 Saved Checkpoint: {ckpt_path}", df_loss, df_lr
                
    yield "✅ Training Complete. Go to 'Utilities' tab to Convert.", df_loss, df_lr