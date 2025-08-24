import streamlit as st
from pathlib import Path
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag import initialize_state, answer_query, retrieve, extract_metric_value
from src.ft import generate_answer_ft
from src.config import DEFAULT_MODE

st.set_page_config(page_title="Financial QA: RAG vs Fine-Tune", layout="wide")

st.title("📊 Comparative Financial QA — RAG vs Fine-Tuning")
st.caption("Open-source only • Same data • Confidence, latency, robustness")

@st.cache_resource(show_spinner=True)
def get_state():
    return initialize_state()

mode = st.sidebar.selectbox("Mode", ["RAG","FT"], index=0 if DEFAULT_MODE=="RAG" else 1)
use_raft = st.sidebar.checkbox("Use retrieval for FT (RAFT)", value=True)
use_reranker = st.sidebar.checkbox("Use cross-encoder reranker", value=True)
top_k_dense = st.sidebar.slider("Dense top_k", 1, 20, 6)
top_k_sparse = st.sidebar.slider("Sparse top_k", 1, 20, 6)
alpha = st.sidebar.slider("Fusion alpha (dense weight)", 0.0, 1.0, 0.5, 0.05)
query = st.text_input("Enter your question about the financial statements:", "What was the company’s revenue in 2024?")

col1, col2 = st.columns(2)
with col1:
    if st.button("Run"):
        if mode == "RAG":
            state = get_state()
            out = answer_query(state, query, top_k_dense=top_k_dense, top_k_sparse=top_k_sparse, alpha=alpha, use_reranker=use_reranker)
            st.subheader("Answer (RAG)")
            st.write(out["answer"])
            st.metric("Confidence", out["confidence"])
            st.metric("Latency (s)", round(out["latency"],3))
            with st.expander("Retrieved Context (top)"):
                for ctx in out["contexts"]:
                    st.text(ctx[:1000] + ("..." if len(ctx)>1000 else ""))
            if out.get("sources"):
                with st.expander("Sources"):
                    st.json(out["sources"])            
            if out.get("guardrail",{}).get("flag"):
                st.warning(out["guardrail"]["reason"])
        else:
            model_dir = Path("reports/ft_model")
            if not model_dir.exists():
                st.error("Fine-tuned model not found. Please run fine-tuning first (see README).")
            else:
                import time
                start_time = time.time()
                state = get_state() if use_raft else None
                ans, conf = generate_answer_ft(model_dir, query, return_confidence=True, state=state)
                # If RAFT is enabled but the model returned a long text or no currency, try precise extraction from contexts
                if use_raft and state is not None:
                    ranked = retrieve(state, query)
                    contexts = [state["docs"][idx] for (_, idx) in ranked[:6]]
                    extracted = extract_metric_value(query, contexts)
                    if extracted and (not isinstance(ans, str) or len(ans) > 16 or not any(c in ans for c in ['£','$','€'])):
                        if not any(c in extracted for c in ['£','$','€']):
                            ans = f"£{extracted}"
                        else:
                            ans = extracted
                latency = time.time() - start_time

                st.subheader("Answer (Fine-Tuned)")
                st.write(ans)
                st.metric("Confidence", f"{conf:.2f}")
                st.metric("Latency (s)", f"{latency:.2f}")
                if use_raft and state is not None:
                    ranked = retrieve(state, query)
                    contexts = [state["docs"][idx] for (_, idx) in ranked[:6]]
                    with st.expander("Retrieved Context (FT / RAFT)"):
                        for ctx in contexts:
                            st.text(ctx[:1000] + ("..." if len(ctx)>1000 else ""))
with col2:
    st.info("Use the sidebar to switch between **RAG** and **FT**. The RAG mode shows retrieved context and applies guardrails. FT supports RAFT (retrieval-augmented) when enabled and requires a fine-tuned model at `reports/ft_model/`.")