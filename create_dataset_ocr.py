import os
import gc
import pytesseract
import re
from pdf2image import convert_from_path
from tqdm import tqdm

RAW_DIR = "data/raw"
TOKEN_DIR = "data/token"
CHUNK_SIZE = 5
os.makedirs(TOKEN_DIR, exist_ok=True)

def clean_ocr_text(text):
    if not text: return ""
    lines = text.split('\n')
    cleaned = []
    noise_pattern = re.compile(r"^[\W_]+$") 
    for line in lines:
        s = line.strip()
        if len(s) < 10: continue
        if noise_pattern.match(s): continue
        cleaned.append(s)
    return "\n".join(cleaned)

def convert_scanned_pdfs():
    if not os.path.exists(RAW_DIR): return
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    for filename in files:
        pdf_path = os.path.join(RAW_DIR, filename)
        txt_path = os.path.join(TOKEN_DIR, os.path.splitext(filename)[0] + ".txt")
        print(f"👁️ OCR: {filename}")
        try:
            images = convert_from_path(pdf_path, dpi=200)
            with open(txt_path, "w", encoding="utf-8") as f:
                for img in tqdm(images, desc="OCR Progress"):
                    raw_text = pytesseract.image_to_string(img)
                    clean = clean_ocr_text(raw_text)
                    if clean: f.write(clean + "\n\n")
            del images
            gc.collect()
        except Exception as e:
            print(f"❌ OCR Error: {e}")

if __name__ == "__main__":
    convert_scanned_pdfs()