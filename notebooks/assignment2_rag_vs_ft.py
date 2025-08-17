# Assignment 2 – RAG vs Fine-Tuning (Executable Notebook as .py)
# Run cells with: `ipython notebooks/assignment2_rag_vs_ft.py`

# %% Imports & Setup
from pathlib import Path
import pandas as pd
from src.rag import build_corpus, fit_indices, answer_query
from src.ft import load_qa_csv, finetune_lora, generate_answer_ft
from src.eval import evaluate_system

# %% Build indices for RAG
corpus = build_corpus()
state = fit_indices(corpus)

# %% Try RAG on a sample question
out = answer_query(state, "What was the company’s revenue in 2024?")
print(out)

# %% Fine-tune (LoRA) and test
qa_df = load_qa_csv(Path("qa/qa_seed.csv"))
model_dir = Path("reports/ft_model")
model_dir.mkdir(parents=True, exist_ok=True)
finetune_lora(qa_df, model_dir, epochs=1)
ans = generate_answer_ft(model_dir, "What was the company’s revenue in 2024?")
print(ans)

# %% Evaluate both on 10 questions
def rag_infer(q): return answer_query(state, q)
def ft_infer(q): return generate_answer_ft(model_dir, q)
res_rag = evaluate_system(qa_df.head(10), rag_infer, "RAG")
res_ft = evaluate_system(qa_df.head(10), ft_infer, "Fine-Tune")
res = pd.concat([res_rag, res_ft], axis=0)
res.to_csv("reports/results.csv", index=False)
print(res.head())
