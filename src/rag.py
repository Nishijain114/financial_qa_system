import os, time, math, json, re
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

from .config import (COMPANY_DIR, CHUNK_SIZES, CHUNK_OVERLAP, EMBEDDING_MODEL,
                     DENSE_TOP_K, SPARSE_TOP_K, FUSION_ALPHA, GEN_MODEL,
                     MAX_CONTEXT_DOCS, ENABLE_INPUT_GUARDRAIL, ENABLE_OUTPUT_NUMERIC_CHECK)
from .data_prep import load_texts_from_dir, segment_into_sections, make_chunks, preprocess_query

from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Corpus & Indexing
# -------------------------
def build_corpus():
    texts = load_texts_from_dir(COMPANY_DIR)
    corpus = []
    for doc_id, raw in texts.items():
        sections = segment_into_sections(raw)
        for sec_name, sec_text in sections.items():
            for sz in CHUNK_SIZES:
                chunks = make_chunks(sec_text, sz, CHUNK_OVERLAP, {"doc_id": doc_id, "section": sec_name, "chunk_size": sz})
                corpus.extend(chunks)
    return corpus

def fit_indices(corpus: List[Dict]):
    docs = [c["text"] for c in corpus]
    # Sparse: BM25
    bm25 = BM25Okapi([d.split() for d in docs])
    # Sparse: TF-IDF (for normalization)
    tfidf = TfidfVectorizer().fit(docs)

    # Dense: Sentence embeddings + FAISS
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    embs = encoder.encode(docs, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    return {
        "bm25": bm25,
        "tfidf": tfidf,
        "dense_index": index,
        "dense_encoder": encoder,
        "docs": docs,
        "corpus": corpus
    }

# -------------------------
# Guardrails
# -------------------------
def input_guardrail(query: str) -> Tuple[bool, str]:
    if not ENABLE_INPUT_GUARDRAIL:
        return True, ""
    allowed_keywords = ['revenue','income','profit','loss','assets','liabilities','equity','cash','expenses','cogs','eps','segment','balance','income statement','balance sheet','cash flow','headcount','guidance','margin','capex','free cash flow']
    q = query.lower()
    if any(k in q for k in allowed_keywords):
        return True, ""
    return False, "Query appears out of scope for financial statements."

def numeric_output_guardrail(answer: str, contexts: List[str]) -> Dict:
    if not ENABLE_OUTPUT_NUMERIC_CHECK:
        return {"flag": False, "reason": ""}
    nums_ans = re.findall(r"\$?\d+(?:\.\d+)?", answer)
    joined = " ".join(contexts)
    flags = [n for n in nums_ans if n not in joined]
    if flags:
        return {"flag": True, "reason": f"Numbers {flags} not found in retrieved context. Low confidence."}
    return {"flag": False, "reason": ""}

# -------------------------
# Retrieval
# -------------------------
# -------------------------
# Retrieval (Hybrid Search: Sparse + Dense)
# -------------------------
def retrieve(state, query: str, top_k_dense=DENSE_TOP_K, top_k_sparse=SPARSE_TOP_K, alpha=FUSION_ALPHA) -> list[tuple[float,int]]:
    """
    Hybrid Retrieval:
    - Dense: Sentence embeddings + FAISS
    - Sparse: BM25
    - Fusion: Weighted combination using alpha
    """
    q_proc = preprocess_query(query)
    
    # ----- Dense Retrieval -----
    q_emb = state["dense_encoder"].encode([q_proc], normalize_embeddings=True)[0]
    D, I = state["dense_index"].search(np.array([q_emb]), top_k_dense)
    dense_hits = list(zip(D[0].tolist(), I[0].tolist()))
    
    # ----- Sparse Retrieval -----
    bm25 = state["bm25"]
    scores = bm25.get_scores(q_proc.split())
    sparse_idxs = np.argsort(scores)[::-1][:top_k_sparse]
    sparse_hits = [(scores[i], int(i)) for i in sparse_idxs]
    
    # ----- Fusion -----
    fused = {}
    if dense_hits:
        d_scores = np.array([h[0] for h in dense_hits])
        d_norm = (d_scores - d_scores.min()) / (np.ptp(d_scores)+1e-6)
        for (ds, di), dn in zip(dense_hits, d_norm):
            fused[di] = fused.get(di, 0.0) + float(alpha * dn)  # alpha weight for dense

    if sparse_hits:
        s_scores = np.array([h[0] for h in sparse_hits])
        s_norm = (s_scores - s_scores.min()) / (np.ptp(s_scores)+1e-6)
        for (ss, si), sn in zip(sparse_hits, s_norm):
            fused[si] = fused.get(si, 0.0) + float((1-alpha) * sn)  # 1-alpha weight for sparse

    ranked = sorted([(score, idx) for idx, score in fused.items()], key=lambda x: x[0], reverse=True)
    return ranked

# -------------------------
# Generator
# -------------------------
def load_generator():
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model

# -------------------------
# Query Answering
# -------------------------
def answer_query(state, query: str, max_new_tokens=64, temperature=0.7) -> Dict:
    ok, msg = input_guardrail(query)
    t0 = time.time()
    if not ok:
        return {"answer": "Out of scope: " + msg, "confidence": 0.2, "latency": time.time()-t0, "method": "RAG", "contexts": []}

    ranked = retrieve(state, query)
    # Top 6 contexts
    picked = ranked[:6]
    contexts = [state["docs"][idx] for (_, idx) in picked]
    context_text = "\n".join(contexts)
    # Truncate context to max tokens
    context_text = " ".join(context_text.split()[:MAX_CONTEXT_DOCS])

    prompt = f"Answer the question based on the context.\n\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer:"

    tok, model = load_generator()
    input_ids = tok.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)

    # Proper generation
    gen_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=tok.eos_token_id
    )

    out = tok.decode(gen_ids[0], skip_special_tokens=True)
    # Extract answer after "Answer:"
    answer = out.split("Answer:")[-1].strip()
    # Remove repeated sentences
    answer = re.sub(r'(\b.+?\b)( \1)+', r'\1', answer)

    # Numeric guardrail
    guard = numeric_output_guardrail(answer, contexts)
    conf = 0.9 if not guard["flag"] else 0.5
    latency = time.time() - t0

    return {"answer": answer, "confidence": conf, "latency": latency, "method": "RAG", "contexts": contexts, "guardrail": guard}
