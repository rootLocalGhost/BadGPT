import os
import pdfplumber
import re
from tqdm import tqdm

RAW_DIR = "data/raw"
TOKEN_DIR = "data/token"

os.makedirs(TOKEN_DIR, exist_ok=True)

def clean_text(text):
    if not text: return ""
    lines = text.split('\n')
    cleaned = []
    img_pattern = re.compile(r"^\[Image \d+\]$", re.IGNORECASE)
    
    for line in lines:
        s = line.strip()
        if len(s) < 4: continue
        if img_pattern.match(s): continue
        cleaned.append(s)
    
    return "\n".join(cleaned)

def convert_pdfs():
    if not os.path.exists(RAW_DIR):
        print(f"❌ Error: {RAW_DIR} not found.")
        return

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    print(f"📂 Found {len(files)} PDFs in {RAW_DIR}")

    for filename in files:
        pdf_path = os.path.join(RAW_DIR, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(TOKEN_DIR, txt_filename)
        
        print(f"📖 Processing {filename}...")
        
        with open(txt_path, "w", encoding="utf-8") as f_out:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in tqdm(pdf.pages, desc="Pages", leave=False):
                        text = page.extract_text()
                        if text:
                            f_out.write(clean_text(text) + "\n\n")
                            if hasattr(page, 'flush_cache'): page.flush_cache()
            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")

    print(f"✅ All Done! Check {TOKEN_DIR}")

if __name__ == "__main__":
    convert_pdfs()