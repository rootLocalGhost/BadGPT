import torch
import tiktoken
import os
import glob
from architecture import GPTModel, DEVICE, BLOCK_SIZE

enc = tiktoken.get_encoding("cl100k_base")

def load_model_weights(weights_dir="model/weights"):
    files = glob.glob(f"{weights_dir}/*.pth")
    if not files:
        return None, "⚠️ No weights found. Train and export first."
    
    latest_file = max(files, key=os.path.getmtime)
    print(f"🧠 Loading Inference Weights: {latest_file}")
    
    model = GPTModel().to(DEVICE)
    model.load_state_dict(torch.load(latest_file, map_location=DEVICE))
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