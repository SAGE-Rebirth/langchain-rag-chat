# LangChain RAG Chat

A Streamlit-powered web application that lets you chat with your own documents using Retrieval-Augmented Generation (RAG) and Hugging Face models via LangChain. Upload PDFs, DOCX, or TXT files, and ask questions to get AI-generated answers grounded in your content.

---

## Table of Contents
- [Overview](#overview)
- [Key Concepts Used](#key-concepts-used)
  - [LangChain](#langchain)
  - [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [Hugging Face](#hugging-face)
  - [Streamlit](#streamlit)
- [How It Works](#how-it-works)
- [Setup & Usage](#setup--usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**LangChain RAG Chat** enables users to interact with their own documents using advanced language models. It combines document retrieval and generative AI to provide accurate, context-aware answers to your questions.

## Key Concepts Used

### LangChain
- Used for chaining together document retrieval and language model inference.
- Handles splitting documents, embedding, and conversational retrieval.

### Retrieval-Augmented Generation (RAG)
- RAG combines information retrieval with generative models.
- The app retrieves relevant document chunks and feeds them to the language model for grounded answers.

### Hugging Face
- Utilizes Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- Uses Hugging Face Inference Endpoints (e.g., Llama-3.1-8B-Instruct) for text generation.
- Requires a Hugging Face API token for authentication.

### Streamlit
- Provides the interactive web UI for uploading files and chatting.
- Handles user input, file uploads, and displays answers in real time.

## How It Works
1. **Upload Documents:**
   - Upload PDF, DOCX, or TXT files via the sidebar.
2. **Text Extraction & Chunking:**
   - Extracts text from uploaded files and splits it into manageable chunks.
3. **Embeddings & Vector Store:**
   - Generates embeddings for each chunk using Hugging Face models.
   - Stores embeddings in a FAISS vector store for efficient retrieval.
4. **Conversational Retrieval:**
   - When you ask a question, the app retrieves relevant chunks and passes them to a Hugging Face LLM for answer generation.
5. **Chat Interface:**
   - Displays the conversation history and AI responses.

## Setup & Usage

### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd langchain-rag-chat
```

### 2. Create a Virtual Environment (Recommended)
```sh
python -m venv rag-env
rag-env\Scripts\activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Configure Environment Variables
- Copy `sample.env` to `.env` and add your Hugging Face API token:
```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

### 5. Run the App
```sh
streamlit run app.py
```

### 6. Usage
- Open the Streamlit app in your browser.
- Upload your documents.
- Ask questions in the chat box and get answers based on your content!

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License.
