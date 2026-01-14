import torch
import os
import glob

CHECKPOINT_DIR = "model/checkpoints"
WEIGHTS_DIR = "model/weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def convert():
    files = glob.glob(f"{CHECKPOINT_DIR}/*.pt")
    if not files:
        print("❌ No checkpoints.")
        return
    latest = max(files, key=os.path.getmtime)
    ckpt = torch.load(latest, map_location='cpu')
    save_payload = {
        'model_state': ckpt['model_state'],
        'config': ckpt.get('config', None)
    }
    step = ckpt.get('total_steps', 'unknown')
    mode = ckpt.get('config', {}).get('model_mode', 'scratch')
    out_name = f"badgpt_{mode}_step{step}.pth"
    out_path = os.path.join(WEIGHTS_DIR, out_name)
    torch.save(save_payload, out_path)
    print(f"✅ Exported to {out_path}")

if __name__ == "__main__":
    convert()