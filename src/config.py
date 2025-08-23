from pathlib import Path

DATA_DIR = Path('data')
COMPANY_DIR = DATA_DIR / 'financial_reports_clean'   # change to your_company
CACHE_DIR = Path('.cache')

# Chunking
CHUNK_SIZES = [100, 400]   # tokens-ish (we count words as an approximation)
CHUNK_OVERLAP = 20

# Retrieval
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DENSE_TOP_K = 6
SPARSE_TOP_K = 6
FUSION_ALPHA = 0.5  # weight for dense vs sparse when score-normalized


# Generation model for RAG
GEN_MODEL = 'distilgpt2'
MAX_CONTEXT_DOCS = 1200  # max tokens of context concatenated

# Guardrails
ENABLE_INPUT_GUARDRAIL = True
ENABLE_OUTPUT_NUMERIC_CHECK = True

# Fine-tuning
FT_BASE_MODEL = 'distilgpt2'
FT_METHOD = 'lora'  # 'lora' or 'full'

# UI
DEFAULT_MODE = 'RAG'  # or 'FT'
