# Group_24_RAG_vs_FT — Comparative Financial QA (RAG vs Fine-Tuning)

This repo scaffolds your Assignment 2 end-to-end:
- Data collection & preprocessing (PDF/HTML/Excel → clean text → sections → chunks)
- RAG system (hybrid sparse+dense, multi-chunk sizes, guardrails, UI)
- Fine-tuned model system (adapter/LoRA for a small open-source LM), guardrails, UI
- Unified Streamlit app to switch modes, show confidence, latency, and method
- Evaluation scripts + report template

> **Replace `XX` with your group number** and drop your real company reports into `data/your_company/`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# (Optional) NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

# Launch the app
streamlit run app/streamlit_app.py
```

## How to use with your own data

1. Put your last-two-years financial statements (PDF/HTML/Excel/TXT) under `data/your_company/`.
2. Run `scripts/run_evaluation.py` once to build/load indices (FAISS + BM25) and cache embeddings.
3. Edit `qa/qa_seed.csv` to use your ~50 real Q/A pairs.
4. Re-run the evaluation and take screenshots for the report.

## Advanced Techniques

- **RAG (Group 24 → 24 % 5 = 4): Hybrid Search (Sparse + Dense)** implemented with BM25 + SentenceTransformer + FAISS and score fusion; cached indices in `.cache/rag_indices`.
- **Fine-Tuning (Group 24 → 24 % 5 = 4): Retrieval-Augmented Fine-Tuning (RAFT)** — training prompts include retrieved contexts; inference optionally uses retrieval too.

Engineering improvements:
- Cached indices and generator with strict token budgeting.
- Robust numeric guardrail and evaluation numeric comparison.

## Submission
- Zip the folder as `Group_<Number>_RAG_vs_FT.zip` (script auto-creates one as `reports/submission.zip`).
- Include 3 screenshots (UI with query, method, confidence, time).
- Export `reports/report.md` to PDF.
- Provide a hosted link (Streamlit Community Cloud / local tunnel).

## Models (all open-source)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Generator (RAG): `distilgpt2`
- Fine-tune target: `distilgpt2` with LoRA + RAFT

No proprietary APIs used.
