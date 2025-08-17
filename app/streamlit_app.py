import streamlit as st
from pathlib import Path
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag import build_corpus, fit_indices, answer_query
from src.ft import generate_answer_ft
from src.config import DEFAULT_MODE

st.set_page_config(page_title="Financial QA: RAG vs Fine-Tune", layout="wide")

st.title("📊 Comparative Financial QA — RAG vs Fine-Tuning")
st.caption("Open-source only • Same data • Confidence, latency, robustness")

@st.cache_resource(show_spinner=True)
def get_state():
    corpus = build_corpus()
    state = fit_indices(corpus)
    return state

mode = st.sidebar.selectbox("Mode", ["RAG","FT"], index=0 if DEFAULT_MODE=="RAG" else 1)
query = st.text_input("Enter your question about the financial statements:", "What was the company’s revenue in 2024?")

col1, col2 = st.columns(2)
with col1:
    if st.button("Run"):
        if mode == "RAG":
            state = get_state()
            out = answer_query(state, query)
            st.subheader("Answer (RAG)")
            st.write(out["answer"])
            st.metric("Confidence", out["confidence"])
            st.metric("Latency (s)", round(out["latency"],3))
            with st.expander("Retrieved Context (top)"):
                for ctx in out["contexts"]:
                    st.text(ctx[:1000] + ("..." if len(ctx)>1000 else ""))
            if out.get("guardrail",{}).get("flag"):
                st.warning(out["guardrail"]["reason"])
        else:
            model_dir = Path("reports/ft_model")
            if not model_dir.exists():
                st.error("Fine-tuned model not found. Please run fine-tuning first (see README).")
            else:
                ans = generate_answer_ft(model_dir, query)
                st.subheader("Answer (Fine-Tuned)")
                st.write(ans)
                st.metric("Confidence", "—")
                st.metric("Latency (s)", "—")
with col2:
    st.info("Use the sidebar to switch between **RAG** and **FT**. The RAG mode shows retrieved context and applies guardrails. FT requires you to run the fine-tuning script first to produce a model at `reports/ft_model/`.")
