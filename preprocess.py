import os
import json
import pandas as pd
from tqdm import tqdm
from configs import CHUNKS_DIR
from utils import stream_chunks_from_file

BOOK_MAP = {
    "The Count of Monte Cristo": "data/novels/The Count of Monte Cristo.txt",
    "In Search of the Castaways": "data/novels/In search of the castaways.txt",
    "In search of the castaways": "data/novels/In search of the castaways.txt"
}

def prepare_chunks(dataset_csv_path):
    df = pd.read_csv(dataset_csv_path)

    unique_books = df["book_name"].unique()
    out_path = os.path.join(CHUNKS_DIR, "chunks.jsonl")

    os.makedirs(CHUNKS_DIR, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out:
        for book_name in tqdm(unique_books, desc="Books"):
            if book_name not in BOOK_MAP:
                raise ValueError(f"Unknown book_name: {book_name}")

            novel_path = BOOK_MAP[book_name]
            print(f"\n📖 Processing book: {book_name}")

            for i, chunk in enumerate(stream_chunks_from_file(novel_path)):
                record = {
                    "book_name": book_name,
                    "chunk_id": f"{book_name.replace(' ', '_')}_chunk_{i}",
                    "chunk_index": i,
                    "text": chunk
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Chunks written to {out_path}")

if __name__ == "__main__":
    prepare_chunks("data/dataset.csv")
