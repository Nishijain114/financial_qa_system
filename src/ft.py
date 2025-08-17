import os, time, math, json, re
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .config import FT_BASE_MODEL, FT_METHOD, ADVANCED_FT_TECHNIQUE

def load_qa_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def format_instruction(row):
    return f"Instruction: Answer the financial question.\nQuestion: {row['question']}\nAnswer: {row['answer']}"

def prepare_dataset(df: pd.DataFrame):
    df = df.copy()
    df["text"] = df.apply(format_instruction, axis=1)
    return Dataset.from_pandas(df[["text"]])

def base_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(FT_BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(FT_BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model

def tokenize_function(examples, tok, max_length=256):
    return tok(examples["text"], truncation=True, padding="max_length", max_length=max_length)

def finetune_lora(train_df: pd.DataFrame, output_dir: Path, epochs=3, lr=2e-4, bs=4):
    tok, model = base_model_and_tokenizer()
    dataset = prepare_dataset(train_df)
    tokenized = dataset.map(lambda ex: tokenize_function(ex, tok))
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=bs,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=10,
        save_steps=200,
        fp16=False,
        report_to=[]
    )
    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    return output_dir

def generate_answer_ft(model_dir: Path, question: str, max_new_tokens=64, temperature=0.2):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt = f"Instruction: Answer the financial question.\nQuestion: {question}\nAnswer:"
    input_ids = tok.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    gen_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False)
    out = tok.decode(gen_ids[0], skip_special_tokens=True)
    answer = out.split("Answer:")[-1].strip()
    return answer
