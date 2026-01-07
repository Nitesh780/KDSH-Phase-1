import json
import os
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from configs import INDEX_DIR, EMBED_MODEL

CHUNKS_FILE = "outputs/chunks/chunks.jsonl"

def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)

    texts = []
    metadata = []

    print("📥 Loading chunks...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
            metadata.append(record)

    print(f"🔢 Total chunks loaded: {len(texts)}")

    print("🧠 Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("⚙️ Creating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))

    with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    print("✅ FAISS index built successfully")

if __name__ == "__main__":
    build_index()
