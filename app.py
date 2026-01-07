import streamlit as st
import json
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = "outputs/faiss_index/faiss.index"
META_PATH = "outputs/faiss_index/metadata.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

@st.cache_resource
def load_resources():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = SentenceTransformer(EMBED_MODEL)
    return index, metadata, model

index, metadata, embedder = load_resources()


st.set_page_config(page_title="Narrative Consistency Checker", layout="wide")
st.title("📖 Narrative Consistency Checker")
st.write("Verify if a backstory contradicts a novel using evidence-based reasoning.")

book_name = st.selectbox(
    "Select Book",
    sorted(list(set(m["book_name"] for m in metadata)))
)

backstory = st.text_area(
    "Paste Backstory Here",
    height=200,
    placeholder="Enter character backstory or narrative claim..."
)

def analyze(backstory, book_name):
    claims = [c.strip() for c in backstory.split(".") if len(c.strip()) > 20]
    if not claims:
        claims = [backstory]

    claim_embs = embedder.encode(claims, convert_to_numpy=True)
    faiss.normalize_L2(claim_embs)

    retrieved = []
    for emb in claim_embs:
        _, idxs = index.search(emb.reshape(1, -1), TOP_K)
        for idx in idxs[0]:
            rec = metadata[idx]
            if rec["book_name"] == book_name:
                retrieved.append(rec)


    seen = set()
    retrieved = [r for r in retrieved if not (r["chunk_id"] in seen or seen.add(r["chunk_id"]))]

    label = "✅ CONSISTENT"
    for r in retrieved:
        if any(x in r["text"].lower() for x in ["did not", "never", "not "]):
            label = "❌ CONTRADICT"
            break

    return label, retrieved[:5]

if st.button("Check Consistency"):
    if not backstory.strip():
        st.warning("Please enter a backstory.")
    else:
        label, evidence = analyze(backstory, book_name)

        st.subheader("Result")
        st.markdown(f"### {label}")

        st.subheader("Evidence")
        for i, e in enumerate(evidence, 1):
            st.markdown(f"**Passage {i}:**")
            st.write(e["text"])
