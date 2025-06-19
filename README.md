# Fine-Tuned RAG Chatbot with Streaming Responses

**Author:** Chanchal Alam  
**Assignment for:** Amlgo Labs

---

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot that allows users to interactively query the contents of a PDF document. The system leverages semantic search and a local large language model (LLM) to provide accurate, context-grounded answers with streaming responses via a user-friendly web interface.

## Features
- Extracts and preprocesses text from PDF documents
- Splits text into manageable, semantically meaningful chunks
- Embeds chunks using a transformer model (`all-MiniLM-L6-v2`)
- Builds a FAISS vector database for efficient semantic retrieval
- Integrates with a local LLM (Mistral via Ollama) for answer generation
- Streamlit web app for interactive Q&A with streaming responses
- Displays sources for each answer

## Tech Stack
- **Python**
- **Streamlit** (web UI)
- **FAISS** (vector database)
- **Sentence Transformers** (embeddings)
- **Ollama** (Mistral LLM)
- **pdfplumber, nltk** (preprocessing)

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download and place your PDF in the `data/` directory.**
4. **Preprocess and chunk the document:**
   ```bash
   python src/preprocess.py
   ```
5. **Embed and index the chunks:**
   ```bash
   python src/embed_and_index.py
   ```
6. **Start the Ollama server with the Mistral model running locally.**
7. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Usage
- Open the web app in your browser.
- Ask questions about the document, the chatbot will retrieve relevant context and generate grounded answers.
- Sources for each answer are displayed for transparency.

## Acknowledgments
- Assignment for Amlgo Labs
- Built by Chanchal Alam
- Powered by open-source LLMs and vector search libraries
