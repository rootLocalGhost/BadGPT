import os
import pdfplumber
import re
from tqdm import tqdm

RAW_DIR = "data/raw"
TOKEN_DIR = "data/token"
os.makedirs(TOKEN_DIR, exist_ok=True)

SKIP_START_PAGES = 7
SKIP_END_PAGES = 15
TOP_MARGIN_CUTOFF = 0.10 
BOTTOM_MARGIN_CUTOFF = 0.90

def clean_paragraph(text):
    if not text: return ""
    text = re.sub(r'(\w+)-\s*\n(\w+)', r'\1\2', text)
    text = re.sub(r'(?<![.!?:])\n', ' ', text)
    lines = text.split('\n')
    cleaned_lines = []
    garbage_pattern = re.compile(
        r'^(\d+|[IVXLCDM]+)$|'
        r'^\s*Sapiens\s*$|'
        r'^\s*Chapter\s*\d+|'
        r'^\s*Part\s+(One|Two|Three)|'
        r'Image credits|Acknowledgements|'
        r'ISBN|Copyright|All rights|www\.|http',
        re.IGNORECASE
    )
    for line in lines:
        s = line.strip()
        if len(s) < 20: continue
        if garbage_pattern.search(s): continue
        if s[0].isdigit(): continue
        cleaned_lines.append(s)
    return "\n".join(cleaned_lines)

def convert_pdfs():
    if not os.path.exists(RAW_DIR):
        print(f"❌ Error: {RAW_DIR} not found.")
        return
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    for filename in files:
        pdf_path = os.path.join(RAW_DIR, filename)
        txt_path = os.path.join(TOKEN_DIR, os.path.splitext(filename)[0] + ".txt")
        print(f"📖 Processing {filename}...")
        with open(txt_path, "w", encoding="utf-8") as f_out:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    start_idx = min(SKIP_START_PAGES, len(pdf.pages))
                    end_idx = max(start_idx, len(pdf.pages) - SKIP_END_PAGES)
                    for i in tqdm(range(start_idx, end_idx), desc="Reading"):
                        page = pdf.pages[i]
                        crop_box = (0, page.height * TOP_MARGIN_CUTOFF, page.width, page.height * BOTTOM_MARGIN_CUTOFF)
                        try:
                            text = page.crop(crop_box).extract_text()
                            if text:
                                clean = clean_paragraph(text)
                                if len(clean) > 50: f_out.write(clean + "\n\n")
                        except:
                            continue
            except Exception as e:
                print(f"⚠️ Error: {e}")
    print(f"✅ Done. Data in {TOKEN_DIR}")

if __name__ == "__main__":
    convert_pdfs()