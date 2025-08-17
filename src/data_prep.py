import re, os, math, json
from pathlib import Path
from typing import List, Dict, Tuple
import pdfplumber
import fitz  # pymupdf
from bs4 import BeautifulSoup
import pandas as pd
from unidecode import unidecode
from nltk.corpus import stopwords

STOPWORDS = set()
try:
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    # NLTK not downloaded; proceed with empty stopword set
    STOPWORDS = set()

def load_texts_from_dir(dir_path: Path) -> Dict[str, str]:
    texts = {}
    for p in dir_path.glob("*"):
        if p.suffix.lower() in ['.txt']:
            texts[p.stem] = p.read_text(errors='ignore')
        elif p.suffix.lower() in ['.pdf']:
            texts[p.stem] = extract_text_pdf(p)
        elif p.suffix.lower() in ['.html', '.htm']:
            texts[p.stem] = extract_text_html(p)
        elif p.suffix.lower() in ['.xlsx', '.xls', '.csv']:
            texts[p.stem] = extract_text_table(p)
    return texts

def extract_text_pdf(pdf_path: Path) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages)
    except Exception:
        # fallback to pymupdf
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
    return clean_text(text)

def extract_text_html(html_path: Path) -> str:
    html = html_path.read_text(errors='ignore')
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator="\n")
    return clean_text(text)

def extract_text_table(path: Path) -> str:
    try:
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        return clean_text(df.to_string(index=False))
    except Exception as e:
        return ""

def clean_text(text: str) -> str:
    # remove headers/footers-like patterns: page numbers, repeated lines
    text = unidecode(text)
    # remove page numbers: "Page x of y" or solitary numbers
    text = re.sub(r'\bPage\s+\d+(\s+of\s+\d+)?\b', ' ', text, flags=re.I)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # collapse spaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()

def segment_into_sections(text: str) -> Dict[str, str]:
    # naive segmentation by common financial headers
    sections = {}
    patterns = [
        ("income_statement", r"(income statement|statement of operations)", re.I),
        ("balance_sheet", r"(balance sheet|statement of financial position)", re.I),
        ("cash_flow", r"(cash flow|cashflow|statement of cash flows)", re.I),
        ("equity", r"(shareholders' equity|equity)", re.I),
        ("management_discussion", r"(management discussion|md&a|management's discussion)", re.I),
    ]
    # split into lines to search regions
    lines = text.splitlines()
    current = "general"
    sections[current] = []
    for line in lines:
        matched = False
        for name, pat, flags in patterns:
            if re.search(pat, line, flags):
                current = name
                if current not in sections:
                    sections[current] = []
                matched = True
                break
        sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}

def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\w+(?:'\w+)?", text)

def preprocess_query(q: str) -> str:
    q = q.lower()
    tokens = [t for t in tokenize_words(q) if t not in STOPWORDS]
    return " ".join(tokens)

def make_chunks(text: str, chunk_words: int, overlap: int, meta: Dict) -> List[Dict]:
    words = tokenize_words(text)
    chunks = []
    i = 0
    uid_base = f"{meta.get('doc_id','doc')}:{meta.get('section','section')}"
    while i < len(words):
        window = words[i:i+chunk_words]
        if not window:
            break
        chunk_text = " ".join(window)
        chunk = {
            "id": f"{uid_base}:{i}",
            "text": chunk_text,
            "meta": meta
        }
        chunks.append(chunk)
        i += max(1, chunk_words - overlap)
    return chunks
