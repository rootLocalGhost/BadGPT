import os
import sys
import gc
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm
import re

RAW_DIR = "data/raw"
TOKEN_DIR = "data/token"
CHUNK_SIZE = 10  

os.makedirs(TOKEN_DIR, exist_ok=True)

def clean_ocr_text(text):
    if not text: return ""
    lines = text.split('\n')
    cleaned = []
    noise_pattern = re.compile(r"^[\W_]+$") 
    
    for line in lines:
        s = line.strip()
        if len(s) < 4: continue
        if noise_pattern.match(s): continue
        cleaned.append(s)
    
    return "\n".join(cleaned)

def _process_image_batch(images, output_file):
    batch_text = []
    for img in images:
        raw_text = pytesseract.image_to_string(img)
        clean = clean_ocr_text(raw_text)
        if clean:
            batch_text.append(clean)
    
    if batch_text:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("\n\n".join(batch_text) + "\n\n")

def convert_scanned_pdfs():
    if not os.path.exists(RAW_DIR):
        print(f"❌ Error: {RAW_DIR} not found.")
        return

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    if not files:
        print(f"⚠️ No PDFs found in {RAW_DIR}")
        return

    print(f"👀 Found {len(files)} PDFs. Starting OCR (This will be slow)...")

    for filename in files:
        pdf_path = os.path.join(RAW_DIR, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(TOKEN_DIR, txt_filename)
        
        print(f"📖 OCR Processing: {filename} -> {txt_filename}")
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("")

        try:
            from pypdf import PdfReader
            try:
                total_pages = len(PdfReader(pdf_path).pages)
            except:
                total_pages = "?"
            
            print(f"   ↳ Est. Pages: {total_pages}. processing in chunks of {CHUNK_SIZE}...")
            
            if total_pages == "?":
                print("   ⚠️  Cannot count pages. Using failsafe mode (entire file at once - might use high RAM)")
                images = convert_from_path(pdf_path)
                _process_image_batch(images, txt_path)
                del images
            else:
                for i in tqdm(range(0, total_pages, CHUNK_SIZE), unit="chunk"):
                    images = convert_from_path(
                        pdf_path, 
                        first_page=i+1, 
                        last_page=min(i+CHUNK_SIZE, total_pages)
                    )
                    _process_image_batch(images, txt_path)
                    del images
                    gc.collect()
            
            print(f"✅ Finished: {filename} ({os.path.getsize(txt_path)/1024:.1f} KB)")
        except Exception as e:
            print(f"❌ Critical Error on {filename}: {e}")
            print("👉 Tip: Make sure 'poppler-utils' and 'tesseract' are installed.")

if __name__ == "__main__":
    convert_scanned_pdfs()