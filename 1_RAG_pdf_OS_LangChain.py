import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from transformers import pipeline

# New recommended package for HF integrations
# Make sure you have: pip install -U langchain-huggingface
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)

st.title("PDF Query with RAG")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


@st.cache_resource
def get_embeddings_model():
    # Sentence-transformers embeddings, built once
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_llm():
    # Use a smaller FLAN variant to keep things reasonable on CPU
    hf_pipeline = pipeline(
        task="text2text-generation",
        model="google/flan-t5-small",   # IMPORTANT: smaller than base
        device=-1,                      # CPU
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)


@st.cache_resource
def build_vector_store(pdf_bytes: bytes):
    # Build FAISS index from PDF; cached per file content
    temp_pdf_path = "temp.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # slightly larger chunks -> fewer chunks
        chunk_overlap=100,
    )
    texts = text_splitter.split_documents(documents)

    # OPTIONAL: for debugging, cap number of chunks
    # texts = texts[:50]

    embeddings = get_embeddings_model()
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store


if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()

    with st.spinner("Building vector store from PDF (first time can take a while)..."):
        vector_store = build_vector_store(pdf_bytes)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = get_llm()

    system_template = """
You are an assistant that answers questions based only on the provided PDF context.
If the answer is not contained in the context, say you do not know.

Context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Querying the PDF..."):
            result = rag_chain.invoke({"input": query})
        st.write("**Answer:**")
        st.write(result["answer"])
else:
    st.info("Please upload a PDF to begin.")
