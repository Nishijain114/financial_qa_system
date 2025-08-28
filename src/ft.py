import time
import re
from pathlib import Path
from typing import Optional, List
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
try:
    # Newer PEFT provides AutoPeftModelForCausalLM
    from peft import AutoPeftModelForCausalLM
    _HAS_AUTO_PEFT = True
except Exception:
    from peft import PeftModel
    _HAS_AUTO_PEFT = False
import torch

# Optional retrieval imports
from .rag import retrieve, extract_metric_value
from .config import GEN_MAX_INPUT_TOKENS, GEN_SAFETY_MARGIN_TOKENS
from .config import ADD_CURRENCY_WHEN_MISSING, DEFAULT_CURRENCY_SYMBOL

# -------------------------
# Data loading / formatting
# -------------------------
def load_qa_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    q_col = cols.get("question") or "question"
    a_col = cols.get("answer") or "answer"
    df = df.rename(columns={q_col: "question", a_col: "answer"})
    return df[["question", "answer"]].dropna().reset_index(drop=True)

def format_instruction(row):
    return f"Instruction: Answer the financial question.\nQuestion: {row['question']}\nAnswer: {row['answer']}"

def build_raft_prompt(question: str, contexts: List[str], answer: Optional[str] = None) -> str:
    ctx_text = "\n".join(contexts)
    base = f"Instruction: Answer the financial question based on the provided context.\nContext:\n{ctx_text}\n\nQuestion: {question}\nAnswer:"
    if answer is not None:
        return base + f" {answer}"
    return base

def prepare_dataset(df: pd.DataFrame) -> Dataset:
    df = df.copy()
    df["text"] = df.apply(format_instruction, axis=1)
    return Dataset.from_pandas(df[["text"]], preserve_index=False)

def prepare_raft_dataset(df: pd.DataFrame, state, tok, max_input_tokens: int = GEN_MAX_INPUT_TOKENS) -> Dataset:
    rows = []
    # Reserve a small margin because we will not generate during training
    max_len = max_input_tokens
    for _, row in df.iterrows():
        q = row["question"]
        a = row["answer"]
        ranked = retrieve(state, q)
        docs = [state["docs"][idx] for (_, idx) in ranked[:6]]
        # budget contexts to fit
        preamble = "Instruction: Answer the financial question based on the provided context.\nContext:\n"
        question_part = f"\n\nQuestion: {q}\nAnswer: {a}"
        selected = []
        for ctx in docs:
            trial = preamble + "\n".join(selected + [ctx]) + question_part
            if len(tok.encode(trial)) <= max_len:
                selected.append(ctx)
            else:
                break
        text = build_raft_prompt(q, selected, answer=a)
        rows.append({"text": text})
    return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)

# -------------------------
# Model / Tokenizer
# -------------------------
def base_model_and_tokenizer(base_model='distilgpt2'):
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model

# -------------------------
# Tokenization with labels
# -------------------------
def tokenize_function(examples, tok, max_length=512):
    enc = tok(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    labels = []
    for ids, attn in zip(enc["input_ids"], enc["attention_mask"]):
        lab = ids.copy()
        for i, m in enumerate(attn):
            if m == 0:
                lab[i] = -100
        labels.append(lab)
    enc["labels"] = labels
    return enc

# -------------------------
# Fine-tuning (LoRA)
# -------------------------
def finetune_lora(train_df: pd.DataFrame, output_dir: Path, epochs=3, lr=2e-4, bs=4, state=None):
    tok, model = base_model_and_tokenizer()
    if state is None:
        dataset = prepare_dataset(train_df)
    else:
        dataset = prepare_raft_dataset(train_df, state, tok, max_input_tokens=GEN_MAX_INPUT_TOKENS)
    tokenized = dataset.map(lambda ex: tokenize_function(ex, tok), batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=bs,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=10,
        save_steps=200,
        fp16=False,
        report_to=[],
        gradient_accumulation_steps=1,
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"]
    )

    model = get_peft_model(model, peft_config)

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    return output_dir

# -------------------------
# Retrieval-Augmented FT Inference
# -------------------------
def _load_ft_model_and_tokenizer(model_dir: Path, base_model: str = 'distilgpt2'):
    import torch
    # Try to load adapter-aware model first
    tok = None; model = None
    if _HAS_AUTO_PEFT:
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(str(model_dir))
            tok = AutoTokenizer.from_pretrained(str(model_dir))
        except Exception:
            model = None
    if model is None:
        # Fallback: load base then attach adapters
        tok = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForCausalLM.from_pretrained(base_model)
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, str(model_dir))
        except Exception:
            # As a last resort, use base model only
            model = base
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model


def generate_answer_ft(model_dir: Path, question: str, max_new_tokens=64, temperature=0.2, return_confidence=True, state=None):
    import torch
    tok, model = _load_ft_model_and_tokenizer(Path(model_dir))

    contexts = []
    if state is not None:
        ranked = retrieve(state, question)
        contexts = [state["docs"][idx] for (_, idx) in ranked[:6]]
        # Try precise extraction from contexts first
        precise = extract_metric_value(question, contexts)
        if precise:
            if ADD_CURRENCY_WHEN_MISSING and not any(c in precise for c in ['£','$','€']):
                return f"{DEFAULT_CURRENCY_SYMBOL}{precise}", 0.98 if return_confidence else f"{DEFAULT_CURRENCY_SYMBOL}{precise}"
            return precise, 0.98 if return_confidence else precise
        # Build RAFT prompt with contexts
        max_input = GEN_MAX_INPUT_TOKENS - max_new_tokens - GEN_SAFETY_MARGIN_TOKENS
        preamble = "Instruction: Answer with a single numeric value and units if present, based on context.\nContext:\n"
        q_part = f"\n\nQuestion: {question}\nAnswer (number only):"
        selected = []
        for ctx in contexts:
            trial = preamble + "\n".join(selected + [ctx]) + q_part
            if len(tok.encode(trial)) <= max_input:
                selected.append(ctx)
            else:
                break
        prompt = preamble + "\n".join(selected) + q_part
    else:
        prompt = f"Instruction: Answer with a single numeric value if applicable.\nQuestion: {question}\nAnswer:"

    input_ids = tok.encode(prompt, return_tensors="pt", truncation=True, max_length=GEN_MAX_INPUT_TOKENS)
    attention_mask = (input_ids != tok.pad_token_id).long()

    gen_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
        repetition_penalty=1.2,
        eos_token_id=tok.eos_token_id
    )

    out = tok.decode(gen_ids[0], skip_special_tokens=True)
    # Robustly extract after the answer prefix
    answer = out
    for key in ["Answer (number only):", "Answer:"]:
        if key in out:
            answer = out.split(key)[-1].strip()
            break
    # If the model still returned long text, fall back to the first number
    import re as _re
    nums = _re.findall(r"\(?-?[£$€]?\d[\d,\.]*\)?", answer)
    if nums:
        answer = nums[0]
        if ADD_CURRENCY_WHEN_MISSING and not any(c in answer for c in ['£','$','€']):
            answer = f"{DEFAULT_CURRENCY_SYMBOL}{answer}"

    if return_confidence:
        with torch.no_grad():
            outputs = model(input_ids=gen_ids)
            logits = outputs.logits  # [1, seq_len, vocab]
            seq = gen_ids[0]
            input_len = input_ids.shape[1]
            total_len = seq.shape[0]
            gen_len = total_len - input_len
            if gen_len > 0:
                # probs for generated tokens positions
                probs = torch.softmax(logits[0, input_len-1:total_len-1, :], dim=-1)
                chosen = seq[input_len:total_len]
                tok_probs = probs[range(gen_len), chosen]
                confidence = float(tok_probs.mean())
            else:
                confidence = 0.5
        return answer, confidence

    return answer, 0.5
