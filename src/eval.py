import time, re
from typing import Dict, List
import pandas as pd
from pathlib import Path

_CURRENCY = r"[\$£€]?"

def _clean_text(s: str) -> str:
    # remove encoding artifacts and commas
    return s.replace("Â", "").replace(",", "").strip()

def _norm_num(s: str) -> List[float]:
    s = _clean_text(s)
    # handle parentheses as negative
    parts = re.findall(rf"{_CURRENCY}\(?-?\d+(?:\.\d+)?\)?", s)
    out: List[float] = []
    for p in parts:
        neg = p.startswith("-") or (p.startswith("(") and p.endswith(")"))
        p = p.strip("$£€()")
        try:
            v = float(p)
            out.append(-v if neg else v)
        except:
            continue
    return out

def numeric_close(a: str, b: str, tol=0.02) -> bool:
    a = _clean_text(a)
    b = _clean_text(b)
    nums_a = _norm_num(a)
    nums_b = _norm_num(b)

    if nums_a and nums_b:
        # check all combinations of numbers
        for va in nums_a:
            for vb in nums_b:
                if vb == 0 and va == 0:
                    return True
                if vb != 0 and abs(va - vb) / max(1.0, abs(vb)) <= tol:
                    return True
        return False

    # fallback: compare only digits (ignoring words, spaces, punctuation)
    return re.sub(r"\D", "", a) == re.sub(r"\D", "", b)

def evaluate_system(qa_df: pd.DataFrame, infer_fn, system_name: str) -> pd.DataFrame:
    rows = []
    print(f"\n=== Running evaluation for {system_name} ===")
    for i, row in qa_df.iterrows():
        q = row["question"]
        gt = row["answer"]
        t0 = time.time()
        out = infer_fn(q)
        latency = time.time() - t0
        if isinstance(out, dict):
            ans = out.get("answer", "")
            conf = out.get("confidence", None)
        else:
            ans = str(out); conf = None
        correct = numeric_close(ans, gt)
        rows.append({
            "Question": q,
            "Method": system_name,
            "Answer": ans,
            "Confidence": conf if conf is not None else "",
            "Time (s)": round(latency, 3),
            "Correct (Y/N)": "Y" if correct else "N",
            "GroundTruth": gt
        })

        # --- Live print for this question ---
        print(f"[{system_name}] Q{i+1}: {q}")
        print(f"  Pred: {ans}")
        print(f"  GT:   {gt}")
        print(f"  Time: {latency:.2f}s | Correct: {'Y' if correct else 'N'}\n")

    df = pd.DataFrame(rows)
    
    # ---- Accuracy summary ----
    total = len(df)
    correct = (df["Correct (Y/N)"] == "Y").sum()
    acc = 100.0 * correct / max(1, total)
    print(f"\n=== {system_name} Summary ===")
    print(f"Total: {total} | Correct: {correct} | Accuracy: {acc:.2f}%")

    return df
