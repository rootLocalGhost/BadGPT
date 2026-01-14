import torch
import os
import time
import pandas as pd
import gc
import glob
from architecture import ScratchGPT, FineTuneGPT, GPTConfig, DEVICE
from data_utils import count_tokens
import tiktoken
from transformers import GPT2Tokenizer

CHECKPOINT_DIR = "model/checkpoints"
DATA_DIR = "data/token"


def train_generator(
    model_mode,
    steps_to_add,
    autosave_freq,
    batch_size,
    grad_accum,
    lr,
    block_size,
    n_embd,
    n_head,
    n_layer,
    dropout,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    gc.collect()
    if "xpu" in str(DEVICE):
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    files = glob.glob(f"{DATA_DIR}/*.txt")
    full_text = ""
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            full_text += file.read() + "\n<EOF>\n"

    if "gpt2" in model_mode:
        enc = GPT2Tokenizer.from_pretrained("gpt2")
        train_data = torch.tensor(enc.encode(full_text), dtype=torch.long)
    else:
        enc = tiktoken.get_encoding("cl100k_base")
        train_data = torch.tensor(enc.encode(full_text), dtype=torch.long)

    config = GPTConfig(
        block_size=1024 if "gpt2" in model_mode else int(block_size),
        n_embd=768 if "gpt2" in model_mode else int(n_embd),
        n_head=12 if "gpt2" in model_mode else int(n_head),
        n_layer=12 if "gpt2" in model_mode else int(n_layer),
        dropout=float(dropout),
        vocab_size=50257 if "gpt2" in model_mode else 100277,
        model_mode=model_mode,
    )

    model = (
        FineTuneGPT(config).to(DEVICE)
        if "gpt2" in model_mode
        else ScratchGPT(config).to(DEVICE)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))

    start_step = 0
    loss_hist, lr_hist, vram_hist = [], [], []

    model.train()
    t0 = time.time()
    for i in range(start_step, start_step + int(steps_to_add)):
        ix = torch.randint(len(train_data) - config.block_size, (int(batch_size),))
        x = torch.stack([train_data[k : k + config.block_size] for k in ix]).to(DEVICE)
        y = torch.stack([train_data[k + 1 : k + config.block_size + 1] for k in ix]).to(
            DEVICE
        )

        logits, loss = model(x, y)
        loss = loss / int(grad_accum)
        loss.backward()

        if (i + 1) % int(grad_accum) == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if (i + 1) % 5 == 0:
                cur_vram = 0
                if torch.cuda.is_available():
                    cur_vram = torch.cuda.memory_reserved(0) / 1024**3
                elif "xpu" in str(DEVICE):
                    cur_vram = torch.xpu.memory_reserved(0) / 1024**3

                loss_hist.append([i + 1, loss.item() * int(grad_accum)])
                lr_hist.append([i + 1, optimizer.param_groups[0]["lr"]])
                vram_hist.append([i + 1, cur_vram])

                yield (
                    f"Step {i+1} | Loss: {loss.item():.4f}",
                    pd.DataFrame(loss_hist, columns=["Step", "Loss"]),
                    pd.DataFrame(lr_hist, columns=["Step", "LR"]),
                    pd.DataFrame(vram_hist, columns=["Step", "VRAM"]),
                )

        if (i + 1) % int(autosave_freq) == 0:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.__dict__,
                    "total_steps": i + 1,
                },
                f"{CHECKPOINT_DIR}/ckpt_{i+1}.pt",
            )
