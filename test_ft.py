import sys
import os
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ft import generate_answer_ft

FT_MODEL = Path("reports/ft_model")

# Example questions to test
questions = [
    "What was the total income and endowments in 2022/23?",
    "What were the donated services in 2022/23?",
    "How much was the investment income in 2022/23?",
]

for q in questions:
    ans = generate_answer_ft(FT_MODEL, q)
    print(f"Q: {q}\nA: {ans}\n")
