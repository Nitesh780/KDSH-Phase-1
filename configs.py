import os
BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
CHUNKS_DIR = os.path.join(BASE, "outputs", "chunks")
INDEX_DIR = os.path.join(BASE, "outputs", "faiss_index")
DOSSIER_DIR = os.path.join(BASE, "outputs", "dossiers")
RESULTS_CSV = os.path.join(BASE, "outputs", "results.csv")

# chunking
CHUNK_SIZE_WORDS = 800    # ~500-1200 words, tune for your LLM/tokenizer
CHUNK_OVERLAP = 150

# retrieval
TOP_K = 10

# embedder model
EMBED_MODEL = "all-MiniLM-L6-v2"

# LLM config (OpenAI optional)
OPENAI_MODEL = "gpt-4o-mini"  # change if you'd like
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

# make directories on import (safe)
for p in [CHUNKS_DIR, INDEX_DIR, DOSSIER_DIR]:
    os.makedirs(p, exist_ok=True)
