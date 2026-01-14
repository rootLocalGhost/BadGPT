import os
import pdfplumber
import re
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm
import tiktoken
from transformers import GPT2Tokenizer

RAW_DIR = "data/raw"
TOKEN_DIR = "data/token"

def clean_text(text):
    if not text: return ""
    text = re.sub(r'(\w+)-\s*\n(\w+)', r'\1\2', text)
    text = re.sub(r'(?<![.!?:])\n', ' ', text)
    return text

def process_data(ocr=False):
    os.makedirs(TOKEN_DIR, exist_ok=True)
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    yield f"🔎 Found {len(files)} files."
    for filename in files:
        pdf_path = os.path.join(RAW_DIR, filename)
        txt_path = os.path.join(TOKEN_DIR, os.path.splitext(filename)[0] + ".txt")
        if ocr:
            yield f"👁️ OCR-ing {filename}..."
            images = convert_from_path(pdf_path, dpi=150)
            with open(txt_path, "w", encoding="utf-8") as f:
                for img in images:
                    f.write(clean_text(pytesseract.image_to_string(img)) + "\n\n")
        else:
            yield f"📖 Extracting {filename}..."
            with pdfplumber.open(pdf_path) as pdf:
                with open(txt_path, "w", encoding="utf-8") as f:
                    for page in pdf.pages:
                        f.write(clean_text(page.extract_text()) + "\n\n")
    yield "✅ Data processing complete."

def count_tokens(mode='scratch'):
    files = [os.path.join(TOKEN_DIR, f) for f in os.listdir(TOKEN_DIR) if f.endswith(".txt")]
    total = 0
    enc = GPT2Tokenizer.from_pretrained('gpt2') if 'gpt2' in mode else tiktoken.get_encoding("cl100k_base")
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            total += len(enc.encode(file.read()))
    return total