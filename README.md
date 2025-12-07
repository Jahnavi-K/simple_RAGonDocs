# PDF Search RAG — Minimal End-to-End Pipeline

This project implements a lightweight Retrieval-Augmented Generation (RAG) system for querying PDF documents using fully open-source tools.  
The workflow includes PDF ingestion, text chunking, vector embedding using Sentence-Transformers, FAISS-based retrieval, and FLAN-T5 generation — all wrapped in a simple Streamlit interface.

### Key Features
- PDF loading and structured text extraction  
- Chunking with recursive character splitting  
- Embedding via `sentence-transformers/all-MiniLM-L6-v2`  
- Vector search using FAISS  
- Response generation with a HuggingFace FLAN-T5 model  
- Streamlit front-end for interactive querying  

### How to Run
```bash
# create and activate environment
python3 -m venv .rag-env
source .rag-env/bin/activate

# install dependencies
pip install -r requirements.txt

# launch the app
streamlit run 1_RAG_pdf_OS_LangChain.py

### Acknowledgment:
This repository is an extension of the ideas discussed in “Building a RAG Model with Open-Source Tools – A Journey into PDF Search” by Srinivasan Ramanujam — available on LinkedIn: https://www.linkedin.com/pulse/building-rag-model-open-source-tools-journey-pdf-search-ramanujam-duxvc/
