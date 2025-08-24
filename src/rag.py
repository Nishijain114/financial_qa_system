import os, time, math, json, re, pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import os, warnings, transformers

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()

from .config import (COMPANY_DIR, CHUNK_SIZES, CHUNK_OVERLAP, EMBEDDING_MODEL,
                     DENSE_TOP_K, SPARSE_TOP_K, FUSION_ALPHA, GEN_MODEL,
                     MAX_CONTEXT_DOCS, ENABLE_INPUT_GUARDRAIL, ENABLE_OUTPUT_NUMERIC_CHECK,
                     PERSIST_INDICES, INDEX_CACHE_DIR, CACHE_GENERATOR_IN_MEMORY,
                     GEN_MAX_INPUT_TOKENS, GEN_SAFETY_MARGIN_TOKENS,
                     USE_RERANKER, CROSS_ENCODER_MODEL, RERANK_CANDIDATES)
from .config import ADD_CURRENCY_WHEN_MISSING, DEFAULT_CURRENCY_SYMBOL, SOFT_INPUT_GUARDRAIL
from .data_prep import load_texts_from_dir, segment_into_sections, make_chunks, preprocess_query

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
import torch

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
    # Dense: Sentence embeddings + FAISS
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    embs = encoder.encode(docs, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    state = {
        "bm25": bm25,
        "dense_index": index,
        "dense_encoder": encoder,
        "docs": docs,
        "corpus": corpus,
        "metas": [c.get("meta", {}) for c in corpus]
    }

    if PERSIST_INDICES:
        INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Save docs
        (INDEX_CACHE_DIR / "docs.json").write_text(json.dumps(docs))
        # Save FAISS index
        faiss.write_index(index, str(INDEX_CACHE_DIR / "dense.index"))
        # Save BM25 via pickle of tokenized docs
        try:
            with open(INDEX_CACHE_DIR / "bm25.pkl", "wb") as f:
                pickle.dump(bm25, f)
        except Exception:
            # Fallback: store tokens list and rebuild quickly on load
            tokens = [d.split() for d in docs]
            with open(INDEX_CACHE_DIR / "bm25_tokens.pkl", "wb") as f:
                pickle.dump(tokens, f)
        # Save metas
        (INDEX_CACHE_DIR / "metas.json").write_text(json.dumps(state["metas"]))

    return state


def load_indices_if_available():
    if not PERSIST_INDICES:
        return None
    docs_fp = INDEX_CACHE_DIR / "docs.json"
    faiss_fp = INDEX_CACHE_DIR / "dense.index"
    bm25_fp = INDEX_CACHE_DIR / "bm25.pkl"
    bm25_tokens_fp = INDEX_CACHE_DIR / "bm25_tokens.pkl"
    metas_fp = INDEX_CACHE_DIR / "metas.json"
    if not (docs_fp.exists() and faiss_fp.exists() and (bm25_fp.exists() or bm25_tokens_fp.exists())):
        return None
    try:
        docs = json.loads(docs_fp.read_text())
        index = faiss.read_index(str(faiss_fp))
        if bm25_fp.exists():
            with open(bm25_fp, "rb") as f:
                bm25 = pickle.load(f)
        else:
            with open(bm25_tokens_fp, "rb") as f:
                tokens = pickle.load(f)
            bm25 = BM25Okapi(tokens)
        encoder = SentenceTransformer(EMBEDDING_MODEL)
        metas = json.loads(metas_fp.read_text()) if metas_fp.exists() else [{} for _ in docs]
        state = {
            "bm25": bm25,
            "dense_index": index,
            "dense_encoder": encoder,
            "docs": docs,
            "corpus": None,
            "metas": metas
        }
        return state
    except Exception:
        return None


def initialize_state():
    state = load_indices_if_available()
    if state is not None:
        return state
    corpus = build_corpus()
    return fit_indices(corpus)

# -------------------------
# Guardrails
# -------------------------
def input_guardrail(query: str) -> Tuple[bool, str]:
    if not ENABLE_INPUT_GUARDRAIL:
        return True, ""
    allowed_keywords = [
        'revenue','income','profit','loss','assets','liabilities','equity','cash','expenses','expenditure','total expenditure','total expenses',
        'cogs','eps','segment','balance','income statement','balance sheet','cash flow','headcount','guidance','margin','capex','free cash flow',
        'spending','cost','costs','operating expenses','opex','donated services','donation','investment income','dividends','interest',
        'management fees','charitable activities','audit fees','net income','net movement in funds','net gains on investments','net loss on investments',
        'net assets','unrestricted funds','debtors','creditors','market value'
    ]
    q = query.lower()
    if any(k in q for k in allowed_keywords):
        return True, ""
    return False, "Query appears out of scope for financial statements."

def _normalize_numbers(text: str) -> List[str]:
    # Normalize numbers by removing currency and commas, and handling negatives
    nums = re.findall(r"[\-\(\$£€]?\d[\d,\.]*\)?", text)
    normalized = []
    for n in nums:
        n = n.replace(",", "").replace("$", "").replace("£", "").replace("€", "")
        if n.startswith("(") and n.endswith(")"):
            n = "-" + n[1:-1]
        normalized.append(n)
    return normalized


def numeric_output_guardrail(answer: str, contexts: List[str]) -> Dict:
    if not ENABLE_OUTPUT_NUMERIC_CHECK:
        return {"flag": False, "reason": ""}
    nums_ans = _normalize_numbers(answer)
    joined = " ".join(contexts)
    ctx_nums = set(_normalize_numbers(joined))
    flags = [n for n in nums_ans if n not in ctx_nums]
    if flags:
        return {"flag": True, "reason": f"Numbers {flags} not found in retrieved context. Low confidence."}
    return {"flag": False, "reason": ""}

# Heuristics for numeric extraction fallback
_NUM_PAT = re.compile(r"\(?-?[£$€]?\d[\d,\.]*\)?")

def _is_numeric_question(query: str) -> bool:
    q = query.lower()
    intent_keywords = [
        'revenue','income','profit','loss','assets','liabilities','equity','cash','expenses','cogs','eps',
        'segment','balance','cash flow','guidance','margin','capex','free cash flow','dividend','interest',
        'audit','fees','grants','donated','donation','investment','net assets','creditors','debtors','funds'
    ]
    return any(k in q for k in intent_keywords)

def extract_number_from_contexts(query: str, contexts: List[str]) -> Optional[str]:
    q = query.lower()
    keywords = [w for w in re.findall(r"[a-zA-Z]+", q) if len(w) >= 4]
    best = None
    best_score = -1
    for ctx in contexts:
        if not isinstance(ctx, str):
            continue
        for line in ctx.splitlines():
            ll = line.lower()
            score = sum(1 for k in keywords if k in ll)
            if score <= 0:
                continue
            m = _NUM_PAT.search(line)
            if m:
                if score > best_score:
                    best_score = score
                    best = m.group(0)
    if best is None:
        joined = "\n".join([c for c in contexts if isinstance(c, str)])
        m = _NUM_PAT.search(joined)
        if m:
            best = m.group(0)
    return best

# Domain metric detection and precise extraction
METRIC_SYNONYMS = {
    "total_expenditure": ["total expenditure", "total expenses", "expenditure total", "total costs"],
    "total_income_endowments": ["total income and endowments", "total income"],
    "investment_income": ["investment income", "dividends", "interest on cash"],
    "donated_services": ["donated services"],
    "management_fees": ["management fees", "investment management costs fees", "management fee"],
    "charitable_activities": ["charitable activities", "charitable grants", "grants"],
    "audit_fees": ["audit fees", "auditor"],
    "net_income": ["net income", "net income expenditure", "net income/(expenditure)", "surplus", "deficit"],
    "net_gains": ["net gains on investments", "net gain overall"],
    "net_losses": ["net loss on investments", "net losses on investments"],
    "net_movement": ["net movement in funds"],
    "net_assets": ["total net assets", "net assets"],
    "unrestricted_funds": ["unrestricted funds"],
    "investments_year_end": ["market value at 31 march", "market value at year end", "market value at 31 march"],
    "debtors": ["debtors"],
    "creditors": ["creditors"],
}
_NEXT_LABEL_TOKENS = sorted({t for syns in METRIC_SYNONYMS.values() for t in syns} | {"refunds received"}, key=len, reverse=True)

def _detect_metric(query: str) -> Optional[str]:
    q = query.lower()
    best_key = None
    best_hits = 0
    for key, syns in METRIC_SYNONYMS.items():
        hits = sum(1 for s in syns if s in q)
        if hits > best_hits:
            best_hits = hits
            best_key = key
    return best_key if best_hits > 0 else None

def _parse_year_tokens(query: str) -> List[str]:
    q = query
    # match 2022/23 or 2022 23 or single year like 2024
    m = re.search(r"(20\d{2})\s*[/\- ]\s*(\d{2})", q)
    tokens: List[str] = []
    if m:
        y1 = m.group(1)
        y2 = m.group(2)
        tokens.append(f"{y1} {y2}")
        tokens.append(f"{y1}/{y2}")
    else:
        m2 = re.search(r"(20\d{2})", q)
        if m2:
            y = int(m2.group(1))
            tokens.append(str(y))
            # Assume fiscal year ending Mar next year
            if y >= 2010:
                prev = y - 1
                tokens.append(f"{prev} {str(y)[2:]}")
    return tokens

def _format_number(nstr: str) -> str:
    raw = nstr.replace(",", "").replace(" ", "").replace("£", "").replace("$", "").replace("€", "")
    sign = 1
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1]
        sign = -1
    if raw.startswith("-"):
        raw = raw[1:]
        sign = -1
    try:
        if "." in raw:
            val = float(raw) * sign
            return f"{val:,.2f}".rstrip("0").rstrip(".")
        else:
            val = int(raw) * sign
            return f"{val:,}"
    except Exception:
        return nstr

_YEAR_TOKEN = re.compile(r"\b20\d{2}(?:\s*[\-/]\s*\d{2})?\b")

def _extract_numbers_excluding_years(text: str) -> List[str]:
    # Find all numeric/currency-like strings but drop year tokens such as 2023 or 2023/24
    candidates = re.findall(r"\(?-?[£$€]?\d{1,3}(?:[ ,]\d{3})*(?:\.\d+)?\)?|\(?-?[£$€]?\d+\)?", text)
    results: List[str] = []
    for c in candidates:
        c_clean = c.replace(",", " ")  # normalize for year token check like 2023 24
        if _YEAR_TOKEN.search(c_clean):
            continue
        results.append(c)
    return results

def extract_metric_value(query: str, contexts: List[str]) -> Optional[str]:
    metric = _detect_metric(query)
    if not metric:
        return None
    year_toks = _parse_year_tokens(query)
    candidates: List[str] = []
    for ctx in contexts:
        if not isinstance(ctx, str):
            continue
        for line in ctx.splitlines():
            ll = line.lower()
            labels = METRIC_SYNONYMS.get(metric, [])
            # require metric synonym match
            if not any(s in ll for s in labels):
                continue
            # if year tokens specified, require a year token
            if year_toks and not any(t.replace("/", " ") in ll for t in year_toks):
                continue
            # extract first numeric after the matched phrase, excluding year tokens
            label = next(s for s in labels if s in ll)
            idx = ll.find(label)
            segment = line[idx+len(label):]
            seg_lower = segment.lower()
            # Cut at the next label token if it appears, so we don't overrun into another metric
            cut_idx = None
            for tok in _NEXT_LABEL_TOKENS:
                p = seg_lower.find(tok)
                if p != -1:
                    cut_idx = p if cut_idx is None else min(cut_idx, p)
            if cut_idx is not None and cut_idx > 0:
                segment = segment[:cut_idx]
                seg_lower = segment.lower()
            # Prefer number after the word 'total' if present (e.g., Dividends 330, Interest 169, Total 499)
            if 'total income and endowments' in seg_lower and metric != 'total_income_endowments':
                # Exclude the unrelated 'total income and endowments' tail
                cut = seg_lower.find('total income and endowments')
                trimmed = segment[:cut]
                nums = _extract_numbers_excluding_years(trimmed)
                if nums:
                    candidates.append(_format_number(nums[0]))
                    continue
            if 'total' in seg_lower:
                t_idx = seg_lower.find('total')
                after_total = segment[t_idx:]
                nums = _extract_numbers_excluding_years(after_total)
                if nums:
                    candidates.append(_format_number(nums[0]))
                    continue
            # Otherwise take the first numeric immediately following the label segment
            nums = _extract_numbers_excluding_years(segment)
            if nums:
                candidates.append(_format_number(nums[0]))
    # If nothing on the same line, look in the following line for numeric after the label
    if not candidates:
        for ctx in contexts:
            if not isinstance(ctx, str):
                continue
            lines = ctx.splitlines()
            for i, line in enumerate(lines[:-1]):
                ll = line.lower()
                if not any(s in ll for s in METRIC_SYNONYMS.get(metric, [])):
                    continue
                nxt = lines[i+1]
                nums = _extract_numbers_excluding_years(nxt)
                if nums:
                    candidates.append(_format_number(nums[0]))
    return candidates[0] if candidates else None

# -------------------------
# Retrieval
# -------------------------
# -------------------------
# Retrieval (Hybrid Search: Sparse + Dense)
# -------------------------
def retrieve(state, query: str, top_k_dense=DENSE_TOP_K, top_k_sparse=SPARSE_TOP_K, alpha=FUSION_ALPHA, use_reranker: Optional[bool] = None) -> list[tuple[float,int]]:
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
    # Optional cross-encoder reranking of top candidates
    if (USE_RERANKER if use_reranker is None else use_reranker) and ranked:
        rerank_k = min(RERANK_CANDIDATES, len(ranked))
        candidates = ranked[:rerank_k]
        docs = state["docs"]
        # Lazy-load cross-encoder
        if "cross_encoder" not in state:
            tok_ce = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)
            ce = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_MODEL)
            state["cross_encoder_tok"] = tok_ce
            device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
            state["device"] = device
            state["cross_encoder"] = ce.to(device)
        tok_ce = state["cross_encoder_tok"]
        ce = state["cross_encoder"]
        pairs = [(query, docs[idx]) for (_, idx) in candidates]
        with torch.no_grad():
            inputs = tok_ce([p[0] for p in pairs], [p[1] for p in pairs], padding=True, truncation=True, return_tensors="pt").to(state["device"])
            scores = ce(**inputs).logits.squeeze(-1).detach().cpu()
            scores = scores.tolist() if isinstance(scores, torch.Tensor) else scores
        reranked = sorted([(float(s), candidates[i][1]) for i, s in enumerate(scores)], key=lambda x: x[0], reverse=True)
        # Merge reranked head with remaining tail
        tail = ranked[rerank_k:]
        ranked = reranked + tail
    return ranked

# -------------------------
# Generator
# -------------------------
_GEN_CACHE = {"tok": None, "model": None, "device": None}

def load_generator():
    if CACHE_GENERATOR_IN_MEMORY and _GEN_CACHE["tok"] is not None and _GEN_CACHE["model"] is not None:
        return _GEN_CACHE["tok"], _GEN_CACHE["model"], _GEN_CACHE["device"]
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if CACHE_GENERATOR_IN_MEMORY:
        _GEN_CACHE["tok"], _GEN_CACHE["model"], _GEN_CACHE["device"] = tok, model, device
    return tok, model, device

# -------------------------
# Query Answering
# -------------------------
def answer_query(state, query: str, max_new_tokens=64, temperature=0.7, top_k_dense: Optional[int] = None, top_k_sparse: Optional[int] = None, alpha: Optional[float] = None, use_reranker: Optional[bool] = None) -> Dict:
    ok, msg = input_guardrail(query)
    t0 = time.time()
    if not ok and not SOFT_INPUT_GUARDRAIL:
        return {"answer": "Out of scope: " + msg, "confidence": 0.2, "latency": time.time()-t0, "method": "RAG", "contexts": []}

    ranked = retrieve(state, query, top_k_dense or DENSE_TOP_K, top_k_sparse or SPARSE_TOP_K, alpha if alpha is not None else FUSION_ALPHA, use_reranker)
    # Top 6 contexts
    picked = ranked[:6]
    contexts = [state["docs"][idx] for (_, idx) in picked]
    # Build prompt with precise token budgeting
    tok, model, device = load_generator()
    # Reserve room for new tokens and safety margin
    max_input = GEN_MAX_INPUT_TOKENS - max_new_tokens - GEN_SAFETY_MARGIN_TOKENS
    preamble = "Answer the question based on the context.\n\nContext:\n"
    question_part = f"\n\nQuestion: {query}\nAnswer:"
    # Add contexts until we run out of token budget
    selected_contexts = []
    for ctx in contexts:
        trial_context = "\n".join(selected_contexts + [ctx])
        trial_prompt = preamble + trial_context + question_part
        if len(tok.encode(trial_prompt)) <= max_input:
            selected_contexts.append(ctx)
        else:
            break
    context_text = "\n".join(selected_contexts)

    prompt = f"{preamble}{context_text}{question_part}"
    input_ids = tok.encode(prompt, return_tensors="pt", truncation=True, max_length=GEN_MAX_INPUT_TOKENS).to(device)
    attention_mask = input_ids.ne(tok.pad_token_id).long().to(device)

    # Proper generation
    gen_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=tok.eos_token_id
    )

    out = tok.decode(gen_ids[0], skip_special_tokens=True)
    # Extract answer after "Answer:"
    answer = out.split("Answer:")[-1].strip()
    # Remove repeated sentences
    answer = re.sub(r'(\b.+?\b)( \1)+', r'\1', answer)

    # Numeric guardrail and precise metric extraction fallback
    guard = numeric_output_guardrail(answer, contexts)
    conf = 0.9 if not guard["flag"] else 0.5
    # If the metric is explicitly "total_expenditure", bypass generator output and return extracted value
    metric_key = _detect_metric(query)
    if metric_key == 'total_expenditure':
        precise_te = extract_metric_value(query, contexts)
        if precise_te:
            if ADD_CURRENCY_WHEN_MISSING and not any(c in precise_te for c in ['£','$','€']):
                answer = f"{DEFAULT_CURRENCY_SYMBOL}{precise_te}"
            else:
                answer = precise_te
            conf = 0.98
    if _is_numeric_question(query):
        has_num = bool(_NUM_PAT.search(answer))
        if guard["flag"] or not has_num:
            # Try precise metric extraction first
            precise = extract_metric_value(query, contexts)
            if precise:
                if ADD_CURRENCY_WHEN_MISSING and not any(c in precise for c in ['£','$','€']):
                    answer = f"{DEFAULT_CURRENCY_SYMBOL}{precise}"
                else:
                    answer = precise
                conf = 0.98
            else:
                extracted = extract_number_from_contexts(query, contexts)
                if extracted:
                    if ADD_CURRENCY_WHEN_MISSING and not any(c in extracted for c in ['£','$','€']):
                        answer = f"{DEFAULT_CURRENCY_SYMBOL}{extracted}"
                    else:
                        answer = extracted
                    conf = 0.95
        else:
            # If generator produced non-numeric text but we have a precise metric, replace with it
            precise = extract_metric_value(query, contexts)
            if precise:
                if ADD_CURRENCY_WHEN_MISSING and not any(c in precise for c in ['£','$','€']):
                    answer = f"{DEFAULT_CURRENCY_SYMBOL}{precise}"
                else:
                    answer = precise
                conf = max(conf, 0.95)

    latency = time.time() - t0

    # Sources metadata parallel to contexts
    sources = []
    if "metas" in state and state["metas"]:
        for (_, idx) in picked:
            sources.append(state["metas"][idx])

    if not ok and SOFT_INPUT_GUARDRAIL:
        # annotate low confidence but still return answer
        conf = min(conf, 0.35)
        guard = {"flag": True, "reason": "Query appears out of scope based on keywords; answer provided with low confidence."}

    return {"answer": answer, "confidence": conf, "latency": latency, "method": "RAG", "contexts": contexts, "sources": sources, "guardrail": guard}
