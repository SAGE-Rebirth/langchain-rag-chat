import os
import asyncio
import sys
import streamlit as st
from dotenv import load_dotenv

# Fix event loop issues
if sys.platform == "win32" and (3, 8, 0) <= sys.version_info < (3, 9, 0):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Disable problematic watcher
from streamlit import config
config.set_option("server.fileWatcherType", "none")

# Rest of your imports
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
except ImportError:
    HuggingFaceEmbeddings = None
    HuggingFaceEndpoint = None
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import login
from PyPDF2 import PdfReader
from docx import Document


# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Authenticate with Hugging Face
if HUGGINGFACEHUB_API_TOKEN:
    login(token=HUGGINGFACEHUB_API_TOKEN)
else:
    st.error("Please add your Hugging Face API token to the .env file.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="LangChain RAG Chat", layout="wide")
st.sidebar.title("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

st.title("Chat with your Documents")

# Helper functions for file reading (unchanged)
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_txt(file):
    return file.read().decode("utf-8")

def extract_text(files):
    all_text = ""
    for file in files:
        if file.type == "application/pdf":
            all_text += read_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            all_text += read_docx(file)
        elif file.type == "text/plain":
            all_text += read_txt(file)
    return all_text

# Process uploaded files
if uploaded_files:
    with st.spinner("Processing documents and creating embeddings..."):
        raw_text = extract_text(uploaded_files)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_text(raw_text)
        
        if HuggingFaceEmbeddings is None or HuggingFaceEndpoint is None:
            st.error("Please install langchain-huggingface: pip install -U langchain-huggingface")
            st.stop()
            
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        try:
            vectorstore = FAISS.from_texts(texts, embeddings)
        except ImportError:
            st.error("Please install faiss-cpu: pip install faiss-cpu")
            st.stop()
            
        retriever = vectorstore.as_retriever()
        
        # SOLUTION 1: Use a different model that supports text-generation
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            max_new_tokens=512,
            do_sample=False,
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )
        st.session_state["qa_chain"] = qa_chain
        st.success("Documents processed and ready for chat!")
else:
    st.info("Please upload documents to begin.")

# Chat interface
if "qa_chain" in st.session_state:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    user_input = st.text_input("Ask a question about your documents:")
    
    if user_input:
        with st.spinner("Generating answer..."):
            # For conversational chain
            if isinstance(st.session_state["qa_chain"], ConversationalRetrievalChain):
                result = st.session_state["qa_chain"]({
                    "question": user_input,
                    "chat_history": st.session_state["chat_history"]
                })
                answer = result["answer"]
            # For regular QA chain
            else:
                answer = st.session_state["qa_chain"]({"query": user_input})["result"]
            
            st.session_state["chat_history"].append((user_input, answer))
    
    for i, (q, a) in enumerate(reversed(st.session_state["chat_history"])):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {a}")