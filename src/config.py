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
SOFT_INPUT_GUARDRAIL = True  # if True, still answer but flag when query seems out of scope

# Fine-tuning
FT_BASE_MODEL = 'distilgpt2'
FT_METHOD = 'lora'  # 'lora' or 'full'

# UI
DEFAULT_MODE = 'RAG'  # or 'FT'

# ---------------------------------
# Advanced techniques & caching
# ---------------------------------
# Group number 24 → 24 % 5 == 4
# Advanced RAG technique (option 4): Hybrid Search (Sparse + Dense)
ADVANCED_RAG_TECHNIQUE = 'hybrid_search'

# Advanced FT technique (option 4): Retrieval-Augmented Fine-Tuning (RAFT)
ADVANCED_FT_TECHNIQUE = 'retrieval_augmented_finetuning'

# Caching controls
PERSIST_INDICES = True
INDEX_CACHE_DIR = CACHE_DIR / 'rag_indices'
MODEL_CACHE_DIR = CACHE_DIR / 'models'

# Generation caching
CACHE_GENERATOR_IN_MEMORY = True

# Token limits and safety margins for generator prompting
GEN_MAX_INPUT_TOKENS = 1024
GEN_SAFETY_MARGIN_TOKENS = 32

# Reranking (optional enhancement over hybrid search)
USE_RERANKER = True
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
RERANK_CANDIDATES = 20

# Formatting
ADD_CURRENCY_WHEN_MISSING = True
DEFAULT_CURRENCY_SYMBOL = '£'
