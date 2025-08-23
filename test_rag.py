from src import rag
state = rag.fit_indices(rag.build_corpus())
result = rag.answer_query(state, "What was the total income and endowments in 2022/23?")
print(result)