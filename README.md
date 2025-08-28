# Financial QA System: RAG vs Fine-Tuning

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![License](https://img.shields.io/badge/License-Open--Source-orange.svg)

This project implements a **Comparative Financial Question Answering (QA) System** using two approaches:

1. **Retrieval-Augmented Generation (RAG)**
2. **Retrieval-Augmented Fine-Tuning (RAFT)**

The system allows users to query financial reports and receive accurate, explainable answers. It also provides an interactive **Streamlit web interface** for testing both approaches.

🔗 **Live Demo:** [Financial QA System](https://financialappsystem-hekrv9pd5cqs6fhcs2a3cx.streamlit.app/)

---

## 📂 Project Structure

```
financial_qa_system/
├── app/                    # Streamlit web interface
├── data/                   # Financial reports + parser
├── qa/                     # Seed QA dataset
├── scripts/                # Evaluation pipeline
├── src/                    # Core system logic (RAG & FT)
├── test_ft.py              
├── test_rag.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate    # Windows
   ```

2. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Build Corpus & Initialize Models**
   ```bash
   python -m scripts.run_evaluation --build_corpus
   ```

   This step:
   - Builds RAG document corpus
   - Fine-tunes model with LoRA (RAFT)
   - Runs evaluation & saves results in `reports/results.csv`

4. **Run the Streamlit App**
   ```bash
   streamlit run app/streamlit_app.py --server.port 9000
   ```

   Or access hosted demo: [Streamlit App](https://financialappsystem-hekrv9pd5cqs6fhcs2a3cx.streamlit.app/)

---

## 📊 Results Summary

| Approach        | Accuracy (%) | Avg Latency (ms) |
|-----------------|--------------|------------------|
| RAG             | 80.0         | 612.3            |
| Fine-Tune (RAFT)| 85.0         | 1338.7           |

- **RAFT** achieved **5% higher accuracy** than RAG.
- **RAG** was **~2x faster** than RAFT.

**Conclusion:**  
Use **RAG** for **speed & factual grounding**, and **RAFT** for **accuracy on structured/numeric queries**. A hybrid strategy provides the best balance.

---

## 🔬 Key Methodologies

- **Hybrid Retrieval (RAG):** Combines BM25 sparse retrieval + dense embeddings (FAISS) with optional re-ranking.
- **RAFT Fine-Tuning:** DistilGPT2 fine-tuned with LoRA (r=8, alpha=16, dropout=0.05), context-aware training with retrieval augmentation.
- **Guardrails:** Input rejection for out-of-domain queries & numeric consistency checks.
- **Efficiency:** Cached FAISS indices, token budgets for generation, lightweight model usage.

---

## 👨‍💻 Contributors

- Jaspreet Monga  
- Nishi Jain  
- Anusha Sinha  
- Bharat Goyal  
- Vishal Yadav  


