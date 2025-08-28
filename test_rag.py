from src import rag

state = rag.initialize_state()
result = rag.answer_query(state, "What was the total income and endowments in 2022/23?")

assert isinstance(result, dict)
assert "answer" in result
assert "contexts" in result
print("RAG smoke test OK:", {k: result[k] for k in ["answer","confidence","method"]})