from pathlib import Path

DATA_DIR = Path('data')
COMPANY_DIR = DATA_DIR / 'sample_company'   # change to your_company
CACHE_DIR = Path('.cache')

# Chunking
CHUNK_SIZES = [100, 400]   # tokens-ish (we count words as an approximation)
CHUNK_OVERLAP = 20

# Retrieval
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DENSE_TOP_K = 6
SPARSE_TOP_K = 6
FUSION_ALPHA = 0.5  # weight for dense vs sparse when score-normalized

# Advanced RAG technique by group_number % 5
# 1: Multi-Stage Retrieval, 2: Adaptive Chunk Merging, 3: Cross-Encoder Re-ranking,
# 4: Hybrid Search (Sparse+Dense), 0: Memory-Augmented Retrieval
ADVANCED_RAG_TECHNIQUE = 4

# Generation model for RAG
GEN_MODEL = 'distilgpt2'
MAX_CONTEXT_DOCS = 1200  # max tokens of context concatenated

# Guardrails
ENABLE_INPUT_GUARDRAIL = True
ENABLE_OUTPUT_NUMERIC_CHECK = True

# Fine-tuning
FT_BASE_MODEL = 'distilgpt2'
FT_METHOD = 'lora'  # 'lora' or 'full'
ADVANCED_FT_TECHNIQUE = 2  # 1: SFT, 2: Adapters/PEFT, 3: MoE, 4: Retrieval-Augmented FT, 0: Continual

# UI
DEFAULT_MODE = 'RAG'  # or 'FT'
