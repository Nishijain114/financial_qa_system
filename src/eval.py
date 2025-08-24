import time, re, math, json
from typing import Dict, List
import pandas as pd
from pathlib import Path

_CURRENCY = r"[\$£€]?"

def _norm_num(s: str) -> List[float]:
    s = s.replace(",", "")
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
    nums_a = _norm_num(a)
    nums_b = _norm_num(b)
    if not nums_a or not nums_b:
        return a.strip().lower() == b.strip().lower()
    va, vb = nums_a[0], nums_b[0]
    if vb == 0:
        return va == 0
    return abs(va - vb)/max(1.0, abs(vb)) <= tol

def evaluate_system(qa_df: pd.DataFrame, infer_fn, system_name: str) -> pd.DataFrame:
    rows = []
    for _, row in qa_df.iterrows():
        q = row["question"]
        gt = row["answer"]
        t0 = time.time()
        out = infer_fn(q)
        latency = time.time() - t0
        if isinstance(out, dict):
            ans = out.get("answer","")
            conf = out.get("confidence", None)
        else:
            ans = str(out); conf = None
        correct = numeric_close(ans, gt)
        rows.append({"Question": q, "Method": system_name, "Answer": ans, "Confidence": conf if conf is not None else "", "Time (s)": round(latency,3), "Correct (Y/N)": "Y" if correct else "N", "GroundTruth": gt})
    return pd.DataFrame(rows)
