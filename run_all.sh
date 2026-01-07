#!/usr/bin/env bash
set -euo pipefail

# 1. preprocess chunks
python3 preprocess.py --dataset data/dataset.csv

# 2. index
python3 indexer.py --chunks outputs/chunks/chunks.jsonl

# 3. reason (LLM optional; set OPENAI_API_KEY in env)
python3 reasoner.py --dataset data/dataset.csv --chunks outputs/chunks/chunks.jsonl
