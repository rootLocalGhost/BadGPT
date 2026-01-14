import torch
import tiktoken
import os
import glob
from transformers import GPT2Tokenizer
from architecture import ScratchGPT, FineTuneGPT, GPTConfig, DEVICE

def load_model_weights(weights_dir="model/weights"):
    files = glob.glob(f"{weights_dir}/*.pth")
    if not files: return None, "⚠️ No weights found."
    latest_file = max(files, key=os.path.getmtime)
    payload = torch.load(latest_file, map_location=DEVICE)
    config_dict = payload.get('config', {})
    if not config_dict:
        config = GPTConfig()
    else:
        config = GPTConfig(**config_dict)
    if 'gpt2' in config.model_mode:
        model = FineTuneGPT(config).to(DEVICE)
    else:
        model = ScratchGPT(config).to(DEVICE)
    model.load_state_dict(payload['model_state'])
    model.eval()
    return model, f"✅ Loaded: {os.path.basename(latest_file)}"

def generate_response(model, question, max_len, temp):
    if 'gpt2' in model.config.model_mode:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        prompt = question
    else:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        prompt = f"Question: {question}\nAnswer:"
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    try:
        res_idx = model.generate(idx, max_new_tokens=int(max_len), temperature=temp)
        decoded = tokenizer.decode(res_idx[0].tolist())
        return decoded.replace(prompt, "").strip() if 'gpt2' not in model.config.model_mode else decoded.strip()
    except Exception as e:
        return f"Error: {e}"