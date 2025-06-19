import os
import re
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

PDF_PATH = 'data/AI Training Document.pdf'
CHUNKS_DIR = 'chunks'
CHUNK_MIN_WORDS = 100
CHUNK_MAX_WORDS = 300


def extract_text_from_pdf(pdf_path):
    all_text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + '\n'
    return all_text


def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def chunk_text(text, min_words=100, max_words=300):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ''
    current_word_count = 0
    for sent in sentences:
        words = sent.split()
        if current_word_count + len(words) > max_words and current_word_count >= min_words:
            chunks.append(current_chunk.strip())
            current_chunk = ''
            current_word_count = 0
        current_chunk += ' ' + sent
        current_word_count += len(words)
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def save_chunks(chunks, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(os.path.join(out_dir, f'chunk_{i+1:04d}.txt'), 'w', encoding='utf-8') as f:
            f.write(chunk)
    print(f'Saved {len(chunks)} chunks to {out_dir}')


def main():
    print('Extracting text from PDF...')
    raw_text = extract_text_from_pdf(PDF_PATH)
    print('Cleaning text...')
    cleaned_text = clean_text(raw_text)
    print('Chunking text...')
    chunks = chunk_text(cleaned_text, CHUNK_MIN_WORDS, CHUNK_MAX_WORDS)
    print(f'Number of chunks: {len(chunks)}')
    print('Saving chunks...')
    save_chunks(chunks, CHUNKS_DIR)
    print('Done.')


if __name__ == '__main__':
    main() 