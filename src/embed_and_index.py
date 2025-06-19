import os
import glob
import json
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = 'chunks'
VECTORDB_DIR = 'vectordb'
INDEX_PATH = os.path.join(VECTORDB_DIR, 'faiss.index')
MAPPING_PATH = os.path.join(VECTORDB_DIR, 'chunk_mapping.json')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


def load_chunks(chunks_dir):
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, 'chunk_*.txt')))
    chunks = []
    for file in chunk_files:
        with open(file, 'r', encoding='utf-8') as f:
            chunks.append(f.read())
    return chunks, chunk_files


def main():
    os.makedirs(VECTORDB_DIR, exist_ok=True)
    print('Loading chunks...')
    chunks, chunk_files = load_chunks(CHUNKS_DIR)
    print(f'Loaded {len(chunks)} chunks.')

    print('Loading embedding model...')
    model = SentenceTransformer(EMBEDDING_MODEL)
    print('Generating embeddings...')
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    print('Building FAISS index...')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f'Saved FAISS index to {INDEX_PATH}')

    mapping = {str(i): {"file": os.path.basename(chunk_files[i]), "text": chunks[i]} for i in range(len(chunks))}
    with open(MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f'Saved chunk mapping to {MAPPING_PATH}')


if __name__ == '__main__':
    main() 