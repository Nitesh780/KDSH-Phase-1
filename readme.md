# Narrative Consistency Checker (KDSH Phase-1)

This project implements an end-to-end system to check whether a given **backstory** is **consistent** or **contradictory** with a full-length **novel**.

The system retrieves relevant passages from the novel and provides **evidence-backed reasoning** for each decision.  
It is designed to handle **very long texts (100k+ words)** in a memory-safe manner.

---

## Problem Description

Given:
- A novel (e.g. *The Count of Monte Cristo*)
- A backstory describing characters or events from that novel

The task is to determine:
- **CONSISTENT (1)** — the backstory does not conflict with the novel
- **CONTRADICT (0)** — at least one claim in the backstory conflicts with the novel

Each decision must be supported by **verbatim excerpts** from the novel.

---

## Project Overview

The project follows a retrieval-based reasoning pipeline:

1. **Novel preprocessing**
   - Large novels are streamed from disk
   - Text is split into overlapping chunks without loading everything into memory

2. **Semantic indexing**
   - Each chunk is embedded using a sentence transformer
   - A FAISS vector index is built for efficient similarity search

3. **Reasoning**
   - Backstories are split into individual claims
   - Relevant passages are retrieved from the novel
   - Claims are checked for contradiction using heuristic semantic analysis
   - Evidence is collected and saved

---

## Project Structure

```text
KDSH-Phase-1/
│
├── preprocess.py        # Memory-safe chunking of novels
├── indexer.py           # FAISS index construction
├── reasoner.py          # Consistency reasoning and evidence generation
├── utils.py             # Helper utilities
├── configs.py           # Configuration values
│
├── data/
│   ├── novels/          # Full novel text files
│   └── dataset.csv     # Backstories to be verified
│
├── outputs/
│   ├── chunks/          # Chunked novel text
│   ├── faiss_index/     # FAISS index and metadata
│   ├── dossiers/        # Evidence dossiers (JSON)
│   └── results.csv     # Final predictions
│
├── requirements.txt     # Runtime dependencies
├── requirements-dev.txt # Development dependencies
└── README.md

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Nitesh780/KDSH-Phase-1.git
cd KDSH-Phase-1

python -m venv .venv
source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

python preprocess.py

python indexer.py

python reasoner.py

streamlit run app.py
