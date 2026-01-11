import torch
import os
import glob
import sys

CHECKPOINT_DIR = "model/checkpoints"
WEIGHTS_DIR = "model/weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def convert():
    files = glob.glob(f"{CHECKPOINT_DIR}/*.pt")
    if not files:
        print("❌ No checkpoints found.")
        return

    latest_ckpt = max(files, key=os.path.getmtime)
    print(f"🔧 Converting {latest_ckpt}...")
    
    checkpoint = torch.load(latest_ckpt, map_location='cpu') 
    
    state_dict = checkpoint['model_state']
    step_num = checkpoint.get('total_steps', 'unknown')
    
    output_name = f"badgpt_v1_step{step_num}.pth"
    output_path = os.path.join(WEIGHTS_DIR, output_name)
    
    torch.save(state_dict, output_path)
    
    print(f"✅ Success! Saved Lite Weights to: {output_path}")
    print(f"📉 Size reduced from {os.path.getsize(latest_ckpt)/1024**2:.1f}MB to {os.path.getsize(output_path)/1024**2:.1f}MB")

if __name__ == "__main__":
    convert()