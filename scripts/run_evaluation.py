import time
from pathlib import Path
import pandas as pd
import sys
import os
import re

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag import build_corpus, fit_indices, answer_query
from src.ft import finetune_lora, load_qa_csv, generate_answer_ft
from src.eval import evaluate_system


QA_PATH = Path("qa/qa_seed.csv")
FT_OUT = Path("reports/ft_model")


def extract_number_from_context(contexts):
    """
    Look through retrieved context lines and extract the first number (integer/decimal).
    Returns as integer if possible.
    """
    for ctx in contexts:
        # Ensure ctx is string
        if not isinstance(ctx, str):
            continue
        match = re.search(r'\b\d[\d,\.]*\b', ctx)  # matches 1,499 or 596.12
        if match:
            num = match.group(0).replace(",", "")
            try:
                return int(num)
            except ValueError:
                return float(num)
    return None


def main():
    qa_df = load_qa_csv(QA_PATH)

    # Build RAG state
    print("Building corpus and indices...")
    corpus = build_corpus()
    state = fit_indices(corpus)

    # Prepare inference functions
    def rag_infer(q):
        r = answer_query(state, q)
        r_num = extract_number_from_context(r["contexts"])
        if r_num is None:
            r_num = "0"  # default if no number found
        return {"answer": str(r_num), "confidence": r.get("confidence", 1.0), "latency": r.get("latency", 0)}

    # Fine-tune FT model
    print("Fine-tuning FT model with LoRA...")
    FT_OUT.mkdir(parents=True, exist_ok=True)
    finetune_lora(qa_df, FT_OUT, epochs=1, lr=2e-4, bs=4)

    def ft_infer(q):
        a = generate_answer_ft(FT_OUT, q, return_confidence=True)
        # a may now be a tuple (answer, confidence)
        if isinstance(a, tuple):
            ans_text, conf = a
        else:
            ans_text, conf = a, 0.5
        a_num = extract_number_from_context([ans_text])
        if a_num is None:
            a_num = "0"  # default if no number found
        return {"answer": str(a_num), "confidence": conf, "latency": 0}

    # Mandatory 3 official tests
    official = [
        ("Relevant, high-confidence", "What was the company’s revenue in 2024?"),
        ("Relevant, low-confidence", "How many unique products were sold?"),
        ("Irrelevant", "What is the capital of France?"),
    ]
    print("\n=== Official Tests ===")
    for label, q in official:
        r = rag_infer(q)
        print(f"[RAG] {label} :: {q}\n -> {r['answer']} (conf={r['confidence']:.2f}, {r['latency']:.2f}s)")
        a = ft_infer(q)
        print(f"[FT ] {label} :: {q}\n -> {a['answer']} (conf={a['confidence']:.2f}, {a['latency']:.2f}s)")

    # Extended evaluation (10+ questions)
    print("\nRunning extended evaluation...")
    eval_rag = evaluate_system(qa_df.head(20), rag_infer, "RAG")
    eval_ft = evaluate_system(qa_df.head(20), ft_infer, "Fine-Tune")
    out = pd.concat([eval_rag, eval_ft], axis=0).reset_index(drop=True)
    out.to_csv("reports/results.csv", index=False)
    print("Saved: reports/results.csv")
    print(out.head())


if __name__ == "__main__":
    main()
