import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

VECTORDB_DIR = 'vectordb'
INDEX_PATH = os.path.join(VECTORDB_DIR, 'faiss.index')
MAPPING_PATH = os.path.join(VECTORDB_DIR, 'chunk_mapping.json')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TOP_K = 3

class Retriever:
    def __init__(self, index_path, mapping_path, embedding_model):
        self.index = faiss.read_index(index_path)
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mapping = json.load(f)
        self.model = SentenceTransformer(embedding_model)

    def retrieve(self, query, top_k=TOP_K):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_emb, top_k)
        results = []
        for idx in I[0]:
            chunk_info = self.mapping[str(idx)]
            results.append(chunk_info)
        return results

def format_prompt(query, retrieved_chunks):
    context = '\n\n'.join([f"[Source {i+1}] {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)])
    prompt = f"""You are an AI assistant. Use the following context to answer the user's question.\n\nContext:\n{context}\n\nUser Question: {query}\n\nAnswer (grounded in the context above):"""
    return prompt

def main():
    retriever = Retriever(INDEX_PATH, MAPPING_PATH, EMBEDDING_MODEL)
    user_query = input("Enter your question: ")
    retrieved = retriever.retrieve(user_query)
    print("Retrieved Chunks:")
    for i, chunk in enumerate(retrieved):
        print(f"[Source {i+1}] {chunk['text']}\n")



if __name__ == '__main__':
    main() 