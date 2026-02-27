# University Academic Regulations RAG  
**Retrieval-Augmented Question Answering System**

This project implements a **Retrieval-Augmented Generation (RAG)** system designed to **accurately answer questions about university academic regulations** (quy định học vụ) using **official policy documents** as grounded knowledge sources.

The system focuses on **faithful, document-grounded responses**, reducing hallucinations commonly found in standalone LLMs, and is suitable for real-world deployment in **student support and academic advisory systems**.

---

## Key Features

- **Hybrid document retrieval** over academic regulation texts  
- **LLM-powered answer generation** grounded in retrieved contexts  
- **Source-based reasoning** over official policy documents  
- **Evaluation with RAG metrics** (faithfulness, context precision/recall, answer correctness)  
- Designed for **Vietnamese university academic regulations**

---

## Evaluation Strategy

Evaluation was conducted using **RAGAS metrics** on a curated QA dataset derived from academic regulations:

- **Faithfulness** – Is the answer supported by retrieved documents?
- **Context Precision** – Are retrieved documents relevant?
- **Context Recall** – Are all necessary documents retrieved?
- **Answer Correctness** – Does the answer match the ground truth?

---

### Model Comparison (Overall Scores)

| Model          | Faithfulness | Context Precision | Context Recall | Answer Correctness |
|----------------|-------------|-------------------|----------------|--------------------|
| **DeepSeek v3.1** | 0.8682 | 0.7550 | **0.8733** | **0.7992** |
| **GPT-OSS**       | **0.8990** | **0.7683** | 0.8533 | 0.7157 |
| **Qwen3 Coder**   | 0.8962 | 0.7533 | **0.8733** | 0.7417 |

---

## Analysis & Insights

- **GPT-OSS** achieves the highest **faithfulness**, indicating strong grounding when relevant context is retrieved.
- **DeepSeek v3.1** delivers the **best answer correctness**, suggesting superior reasoning and synthesis over retrieved academic regulations.
- **Qwen3 Coder** shows balanced performance, particularly strong **context recall**, but slightly weaker answer accuracy.
- Overall, results highlight a **trade-off between grounding strength and final answer correctness**, reinforcing the importance of retrieval quality and model selection in RAG systems.

These findings demonstrate that **evaluation-driven model comparison** is critical for building reliable, domain-specific RAG applications.

---

## Knowledge Base

- **Source:** Official university academic regulation documents  
- **Processing:**  
  - Text normalization  
  - Semantic chunking  
  - Metadata-aware indexing  

> All answers are generated **only from retrieved policy content**, ensuring verifiability.

---

## Future Improvements

- **Deployment & Accessibility**  
  Deploy the system as a web-based or API-driven service to support real-time student queries and integration with university platforms.

- **Metadata-Enriched Knowledge Base**  
  Introduce more structured and domain-specific metadata, especially for **temporal hierarchies**, this enables time-aware retrieval and prevents outdated policy answers.

- **Retrieval Optimization**  
  Improve retrieval quality by:
  - Applying reranking models (e.g., cross-encoders)
  - Fine-tuning embeddings on academic regulation data
  - Experimenting with adaptive chunking strategies based on document structure

- **Answer Traceability**  
  Add explicit citation mapping from answers to source documents to enhance transparency and trustworthiness.


