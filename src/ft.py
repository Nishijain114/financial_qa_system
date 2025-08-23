import time
import re
from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

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

def prepare_dataset(df: pd.DataFrame) -> Dataset:
    df = df.copy()
    df["text"] = df.apply(format_instruction, axis=1)
    return Dataset.from_pandas(df[["text"]], preserve_index=False)

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
def tokenize_function(examples, tok, max_length=256):
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
def finetune_lora(train_df: pd.DataFrame, output_dir: Path, epochs=3, lr=2e-4, bs=4):
    tok, model = base_model_and_tokenizer()
    dataset = prepare_dataset(train_df)
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
def generate_answer_ft(model_dir: Path, question: str, max_new_tokens=64, temperature=0.2, return_confidence=True):
    import torch
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompt = f"Instruction: Answer the financial question.\nQuestion: {question}\nAnswer:"
    input_ids = tok.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    attention_mask = torch.ones_like(input_ids)

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
    answer = out.split("Answer:")[-1].strip()

    # Compute confidence
    if return_confidence:
        with torch.no_grad():
            outputs = model(input_ids=gen_ids)
            logits = outputs.logits[0]
            gen_tokens = gen_ids[0][input_ids.shape[1]:]
            token_probs = torch.softmax(logits[input_ids.shape[1]-1:-1, :], dim=-1)
            confidence = float(token_probs[range(len(gen_tokens)), gen_tokens].mean())
        return answer, confidence

    return answer, 0.5  # always return two values
