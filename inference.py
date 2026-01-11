import torch
import tiktoken
import os
import glob
from architecture import GPTModel, GPTConfig, DEVICE

enc = tiktoken.get_encoding("cl100k_base")

def load_model_weights(weights_dir="model/weights"):
    files = glob.glob(f"{weights_dir}/*.pth")
    if not files:
        return None, "⚠️ No weights found. Train and export first."
    
    latest_file = max(files, key=os.path.getmtime)
    print(f"🧠 Loading Inference Weights: {latest_file}")
    
    payload = torch.load(latest_file, map_location=DEVICE)
    
    # Handle both old format (state_dict only) and new format (dict with config)
    if 'config' in payload and payload['config'] is not None:
        cfg_dict = payload['config']
        config = GPTConfig(**cfg_dict)
        state_dict = payload['model_state']
    else:
        # Fallback to default architecture if config missing
        print("⚠️ Config not found in weights. Using default architecture.")
        config = GPTConfig() 
        state_dict = payload if 'model_state' not in payload else payload['model_state']

    model = GPTModel(config).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, f"✅ Loaded: {os.path.basename(latest_file)}"

def generate_response(model, question, max_len, temp):
    if model is None:
        return "❌ Error: Model not loaded."
    
    prompt = f"Question: {question}\nAnswer:"
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    try:
        with torch.no_grad():
            response_idx = model.generate(idx, max_new_tokens=int(max_len), temperature=temp)
        
        decoded = enc.decode(response_idx[0].tolist())
        return decoded.replace(prompt, "").strip()
    except Exception as e:
        return f"Gen Error: {e}"