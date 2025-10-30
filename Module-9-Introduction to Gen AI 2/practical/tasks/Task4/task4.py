import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader



def load_pdfs(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())

        # Remove temp file
        os.remove(tmp_path)

    return docs


# --------------------
# Build Vector Store
# --------------------
def build_vector_store(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

# --------------------
# Streamlit App
# --------------------
st.title("📚 Chat with Multiple PDFs")
st.write("Upload PDFs and ask questions. If the answer isn’t found in the PDFs, I’ll tell you instead of guessing.")

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Correct usage
if uploaded_files:
    with st.spinner("Processing PDFs..."):
        docs = load_pdfs(uploaded_files)   # ✅ pass UploadedFile objects, not .name
        vectorstore = build_vector_store(docs)


    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Load a small Hugging Face model for Q&A
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Question input
    question = st.text_input("Ask a question based on the PDFs:")

    if question:
        result = qa_chain.invoke(question)

        # Check if answer is relevant (based on retrieved docs)
        if result["source_documents"]:
            st.subheader("Answer:")
            st.write(result["result"])

            with st.expander("Sources"):
                for doc in result["source_documents"]:
                    st.write(doc.metadata, "…", doc.page_content[:200])
        else:
            st.subheader("Answer:")
            st.write("❌ Sorry, I could not find the answer in the provided PDFs.")
