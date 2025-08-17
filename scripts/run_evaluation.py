import time
from pathlib import Path
import pandas as pd
from src.rag import build_corpus, fit_indices, answer_query
from src.ft import finetune_lora, load_qa_csv, generate_answer_ft
from src.eval import evaluate_system

QA_PATH = Path("qa/qa_seed.csv")
FT_OUT = Path("reports/ft_model")

def main():
    qa_df = load_qa_csv(QA_PATH)

    # Build RAG state
    print("Building corpus and indices...")
    corpus = build_corpus()
    state = fit_indices(corpus)

    # Prepare inference functions
    def rag_infer(q):
        return answer_query(state, q)

    # Baseline FT evaluation (pre-fine-tune) — optional
    # Fine-tune
    print("Fine-tuning FT model with LoRA...")
    FT_OUT.mkdir(parents=True, exist_ok=True)
    finetune_lora(qa_df, FT_OUT, epochs=1, lr=2e-4, bs=4)

    def ft_infer(q):
        return generate_answer_ft(FT_OUT, q)

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
        print(f"[FT ] {label} :: {q}\n -> {a}")

    # Extended evaluation (10+)
    print("\nRunning extended evaluation...")
    eval_rag = evaluate_system(qa_df.head(20), rag_infer, "RAG")
    eval_ft = evaluate_system(qa_df.head(20), ft_infer, "Fine-Tune")
    out = pd.concat([eval_rag, eval_ft], axis=0).reset_index(drop=True)
    out.to_csv("reports/results.csv", index=False)
    print("Saved: reports/results.csv")
    print(out.head())

if __name__ == "__main__":
    main()
