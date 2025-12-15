# Antibot-AI-Infectious-Disease-Assistant
Final Year Project: Antibot AI – A healthcare assistant that delivers evidence-based antibiotic information using Large Language Models and RAG pipelines.

## 📌 Project Overview

An AI-powered infectious disease assistant using RAG architecture to deliver reliable, document-grounded answers about antibiotics and infectious diseases.
**Antibot AI** is an AI-powered infectious disease assistant designed to provide accurate, evidence-based responses related to antibiotics and infectious diseases. The system combines **Large Language Models (LLMs)** with a **Retrieval-Augmented Generation (RAG)** architecture to ensure that answers are grounded in trusted medical literature rather than relying solely on pretrained model knowledge.

This project was developed as a **Final Year Project (FYP)** to demonstrate the real-world application of **Artificial Intelligence, Natural Language Processing (NLP), and semantic search** in the healthcare domain, with a strong emphasis on reliability, explainability, and responsible AI usage.

---

## How Antibot AI Works (Architecture Overview)

Antibot AI follows a **Retrieval-Augmented Generation (RAG)** pipeline:

1. **Document Ingestion** – Authoritative medical documents are uploaded and processed.
2. **Text Chunking & Embedding** – Documents are split into chunks and converted into vector embeddings.
3. **Vector Storage** – Embeddings are stored in a FAISS vector database for efficient semantic search.
4. **Query Processing** – User questions are embedded and matched against relevant document chunks.
5. **LLM Response Generation** – A Mistral-based LLM generates answers using retrieved context.
6. **Source-Grounded Output** – Responses are based strictly on retrieved medical content, reducing hallucinations.

---

## Key Features

* Retrieval-Augmented Generation (RAG) for factual accuracy
* Document-based question answering
* Mistral LLM via Hugging Face
* FAISS-powered semantic search
* Streamlit-based interactive UI
* Reduced hallucination through source grounding
* Upload and query custom medical documents

---

## Tech Stack

* **Programming Language:** Python
* **Frontend:** Streamlit
* **LLM Framework:** LangChain
* **Language Model:** Mistral (via Hugging Face)
* **Vector Database:** FAISS
* **Embeddings:** Hugging Face Embedding Models
* **Development Tools:** VS Code, Virtual Environment

---

## Project Structure

```bash
Antibot-AI/
│── app.py                # Streamlit application
│── data/                 # Medical documents
│── embeddings/           # Vector storage (FAISS)
│── prompts/              # Custom prompts
│── requirements.txt      # Project dependencies
│── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

```bash
## Clone the repository
git clone https://github.com/your-username/antibot-ai.git
cd antibot-ai

## Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

## Install dependencies
pip install -r requirements.txt

## Run the application
streamlit run app.py
```

---

## Medical Disclaimer

**Antibot AI is intended for educational and research purposes only.**

* This system does **not** provide medical advice, diagnosis, or treatment.
* It should **not** be used as a substitute for professional medical judgment.
* Always consult qualified healthcare professionals for clinical decisions.

---

## Academic Context

This project was developed as part of a **Bachelor of Computer Science Final Year Project**, focusing on:

* Applied AI in healthcare
* Natural Language Processing (NLP)
* Information Retrieval Systems
* Responsible and explainable AI design

---

## Future Enhancements

* Integration of clinical guidelines databases
* Multi-language medical support
* Citation-based answer highlighting
* Role-based access (student / clinician modes)
* Deployment on cloud platforms

---

## Contributions

Contributions, suggestions, and improvements are welcome. Feel free to fork the repository and submit a pull request.

---

