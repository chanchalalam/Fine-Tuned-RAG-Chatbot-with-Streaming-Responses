import streamlit as st
from src.rag_pipeline import Retriever, format_prompt, INDEX_PATH, MAPPING_PATH, EMBEDDING_MODEL
from src.llm import call_ollama
import time

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Initialize retriever
@st.cache_resource
def get_retriever():
    return Retriever(INDEX_PATH, MAPPING_PATH, EMBEDDING_MODEL)

retriever = get_retriever()

# Session state for chat
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.title("📄 RAG Chatbot with Streaming Responses")
st.write("Ask a question about the document. Responses are grounded in the source text.")

# Sidebar info
with st.sidebar:
    st.markdown("### Model Info")
    st.write(f"**Embedding Model:** {EMBEDDING_MODEL}")
    st.write(f"**Vector DB:** FAISS")
    st.write(f"**Chunks Indexed:** {len(retriever.mapping)}")
    st.write(f"**LLM in Use:** Mistral (Ollama)")
    st.button("Clear Chat", on_click=lambda: st.session_state['messages'].clear())

# Chat input
user_input = st.text_input("Your question:", key="user_input")

if st.button("Send") and user_input.strip():
    # Retrieve relevant chunks
    retrieved = retriever.retrieve(user_input)
    prompt = format_prompt(user_input, retrieved)
    answer = call_ollama(prompt, model="mistral")
    st.session_state['messages'].append({
        'role': 'user', 'content': user_input
    })
    st.session_state['messages'].append({
        'role': 'assistant', 'content': answer
    })
    st.rerun()

for i, msg in enumerate(st.session_state['messages']):
    if msg['role'] == 'user':
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
        if i == len(st.session_state['messages']) - 1:
            retrieved = retriever.retrieve(st.session_state['messages'][-2]['content'])
            st.markdown("**Sources:**")
            for j, chunk in enumerate(retrieved):
                st.markdown(f"[Source {j+1}] {chunk['text'][:300]}...")