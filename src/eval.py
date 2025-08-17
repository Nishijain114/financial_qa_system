import time, re, math, json
from typing import Dict, List
import pandas as pd
from pathlib import Path

def numeric_close(a: str, b: str, tol=0.02) -> bool:
    # tolerate small numeric differences; extract first numeric
    import re
    nums_a = re.findall(r"\d+(?:\.\d+)?", a)
    nums_b = re.findall(r"\d+(?:\.\d+)?", b)
    if not nums_a or not nums_b:
        return a.strip().lower() == b.strip().lower()
    try:
        va = float(nums_a[0]); vb = float(nums_b[0])
        if vb == 0:
            return va == 0
        return abs(va - vb)/max(1.0, abs(vb)) <= tol
    except:
        return a.strip().lower() == b.strip().lower()

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
