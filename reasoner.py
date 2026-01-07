import os, json, faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from configs import INDEX_DIR, TOP_K, RESULTS_CSV, DOSSIER_DIR

EMBED_MODEL = "all-MiniLM-L6-v2"

def rule_based_analyze(claims, passages, embedder):
    results = []
    if not passages:
        return [{"claim": c, "label": 1, "excerpts": [], "analysis": "No contradicting evidence found."} for c in claims]

    passage_texts = [p["text"] for p in passages]
    passage_embs = embedder.encode(passage_texts, convert_to_numpy=True)
    claim_embs = embedder.encode(claims, convert_to_numpy=True)

    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(claim_embs, passage_embs)

    for i, claim in enumerate(claims):
        top_idx = sim[i].argsort()[-3:][::-1]
        top_passages = [passages[j] for j in top_idx]

        label = 1
        for p in top_passages:
            txt = p["text"].lower()
            if any(x in txt for x in ["did not", "never", "no ", "not "]):
                label = 0
                break

        results.append({
            "claim": claim,
            "label": label,
            "excerpts": top_passages,
            "analysis": "Heuristic semantic match with negation check."
        })

    return results


def run_reasoner(dataset_csv):
    # ---- Load dataset ----
    df = pd.read_csv(dataset_csv)

    # ---- Load FAISS index ----
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))

    with open(os.path.join(INDEX_DIR, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    embedder = SentenceTransformer(EMBED_MODEL)

    os.makedirs(DOSSIER_DIR, exist_ok=True)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Examples"):
        story_id = int(row["id"])
        book_name = row["book_name"]
        backstory = str(row["content"])

        # ---- Split backstory into claims ----
        claims = [c.strip() for c in backstory.split(".") if len(c.strip()) > 20]
        if not claims:
            claims = [backstory]

        # ---- Embed claims ----
        claim_embs = embedder.encode(claims, convert_to_numpy=True)
        faiss.normalize_L2(claim_embs)

        retrieved = []
        for emb in claim_embs:
            _, idxs = index.search(emb.reshape(1, -1), TOP_K)
            for idx in idxs[0]:
                rec = metadata[idx]
                if rec["book_name"] == book_name:  # 🔑 CRITICAL FILTER
                    retrieved.append(rec)

        # Deduplicate
        seen = set()
        retrieved = [r for r in retrieved if not (r["chunk_id"] in seen or seen.add(r["chunk_id"]))]

        analysis = rule_based_analyze(claims, retrieved, embedder)

        story_label = 0 if any(a["label"] == 0 for a in analysis) else 1

        dossier = {
            "story_id": story_id,
            "book_name": book_name,
            "backstory": backstory,
            "analysis": analysis,
            "evidence_chunks": retrieved[:20]
        }

        dossier_path = os.path.join(DOSSIER_DIR, f"story_{story_id}_dossier.json")
        with open(dossier_path, "w", encoding="utf-8") as f:
            json.dump(dossier, f, indent=2, ensure_ascii=False)

        results.append({
            "id": story_id,
            "label": story_label,
            "evidence_file": os.path.basename(dossier_path)
        })

    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    print(f"\n✅ Results written to {RESULTS_CSV}")


if __name__ == "__main__":
    run_reasoner("data/dataset.csv")
