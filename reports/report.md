# Comparative Financial QA: RAG vs Fine-Tuning

**Group Number:** XX  
**Company:** Replace with your company  
**Data Years:** 2023, 2024

## 1. Data Collection & Preprocessing
- Sources used (PDF/HTML/Excel): _fill here_
- Cleaning rules: header/footer removal, page numbers stripped, unicode normalized.
- Sectioning: Income Statement, Balance Sheet, Cash Flow, Equity, MD&A.

## 2. RAG System
- Chunk sizes tried: 100, 400 words (20 overlap)
- Indices: FAISS (dense) + BM25 (sparse); Hybrid fusion alpha=0.5
- Advanced Technique (Group mod 5): _fill here_
- Guardrails:
  - Input filter for out-of-scope queries
  - Output numeric check against retrieved contexts
- Generator: distilgpt2 (open-source)

## 3. Fine-Tuned System
- Base: distilgpt2
- Advanced FT Technique: _fill here_ (e.g., LoRA adapters)
- Hyperparams: lr=2e-4, bs=4, epochs=1–3, hardware: _fill here_

## 4. Evaluation
Insert 3 screenshots from the UI showing:
- Relevant high-confidence, low-confidence, and irrelevant queries.

### Extended Results (sample)
Attach `reports/results.csv` and paste a summary table here.

| Question | Method | Answer | Confidence | Time (s) | Correct (Y/N) |
|---|---|---|---|---|---|
| Revenue in 2024? | RAG | $4.50B | 0.92 | 0.50 | Y |
| Revenue in 2024? | Fine-Tune | $4.50B | — | 0.41 | Y |

## 5. Analysis
- Accuracy vs Speed: _observations_
- Strengths of RAG: factual grounding, adaptability.
- Strengths of FT: fluent, fast once trained.
- Robustness to irrelevant questions: RAG guardrail returns "out of scope"; FT may generate but needs filtering.
- Practical trade-offs: infra, latency, data freshness.

## 6. Submission
- Provide hosted app link
- Confirm `Group_XX_RAG_vs_FT.zip` attached
