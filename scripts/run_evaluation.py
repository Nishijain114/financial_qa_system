from pathlib import Path
import pandas as pd
import sys
import os
import re

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag import initialize_state, answer_query, retrieve
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

    # Build or load RAG state (cached)
    print("\n" + "="*60)
    print("Loading/Building Corpus and Indices (with cache)...")
    print("="*60)
    state = initialize_state()

    # Prepare inference functions
    def rag_infer(q):
        r = answer_query(state, q)
        return {"answer": r.get("answer", ""), "confidence": r.get("confidence", 1.0), "latency": r.get("latency", 0)}

    # Fine-tune FT model
    print("\n" + "="*60)
    print("Fine-tuning FT Model with LoRA (RAFT enabled)...")
    print("="*60)
    FT_OUT.mkdir(parents=True, exist_ok=True)
    finetune_lora(qa_df, FT_OUT, epochs=1, lr=2e-4, bs=4, state=state)

    def ft_infer(q):
        ans_text, conf = generate_answer_ft(FT_OUT, q, return_confidence=True, state=state)
        return {"answer": ans_text, "confidence": conf, "latency": 0}

    # Mandatory 3 official tests
    print("\n" + "="*60)
    print("Official Sanity Tests")
    print("="*60)
    official = [
        ("Relevant, high-confidence", "What was the company’s revenue in 2024?"),
        ("Relevant, low-confidence", "How many unique products were sold?"),
        ("Irrelevant", "What is the capital of France?"),
    ]
    for label, q in official:
        r = rag_infer(q)
        a = ft_infer(q)
        print(f"\n{label}\nQ: {q}")
        print(f"   [RAG] -> {r['answer']}   (conf={r['confidence']:.2f}, {r['latency']:.2f}s)")
        print(f"   [FT ] -> {a['answer']}   (conf={a['confidence']:.2f}, {a['latency']:.2f}s)")

    # Extended evaluation
    print("\n" + "="*60)
    print("Extended Evaluation (20 samples)")
    print("="*60)
    eval_rag = evaluate_system(qa_df.head(20), rag_infer, "RAG")
    eval_ft = evaluate_system(qa_df.head(20), ft_infer, "Fine-Tune (RAFT)")

    out = pd.concat([eval_rag, eval_ft], axis=0).reset_index(drop=True)
    out.to_csv("reports/results.csv", index=False)

    print("\n" + "="*60)
    print("Results saved to: reports/results.csv")
    print("="*60)

    # Final leaderboard
    rag_acc = (eval_rag["Correct (Y/N)"] == "Y").mean() * 100
    ft_acc = (eval_ft["Correct (Y/N)"] == "Y").mean() * 100
    print("\nLeaderboard (Accuracy)")
    print("-"*60)
    print(f"RAG        : {rag_acc:.2f}%")
    print(f"Fine-Tuned : {ft_acc:.2f}%")
    print("-"*60)

if __name__ == "__main__":
    main()
