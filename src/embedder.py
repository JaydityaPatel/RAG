"""Generates embeddings with FastEmbed's ONNX MiniLM model."""

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks" / "chunks.json"
EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "chunk_embeddings.npy"
IDS_PATH = EMBEDDINGS_DIR / "chunk_ids.json"


def load_chunks():
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def embeddings_are_current():
    return EMBEDDINGS_PATH.exists() and EMBEDDINGS_PATH.stat().st_mtime >= CHUNKS_PATH.stat().st_mtime


@lru_cache(maxsize=1)
def get_embedding_model():
    """Reuse one small ONNX model instance across query operations."""
    return TextEmbedding(model_name=MODEL_NAME, threads=2)


def generate_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.embed(texts, batch_size=32)
    return np.asarray(list(vectors), dtype=np.float32)


def main():
    started = time.perf_counter()
    chunks = load_chunks()
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if embeddings_are_current() and IDS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        with IDS_PATH.open("r", encoding="utf-8") as file:
            chunk_ids = json.load(file)
        print("Embeddings are current; skipped regeneration.")
    else:
        embeddings = generate_embeddings(chunks, get_embedding_model())
        chunk_ids = [chunk["id"] for chunk in chunks]
        np.save(EMBEDDINGS_PATH, embeddings)
        with IDS_PATH.open("w", encoding="utf-8") as file:
            json.dump(chunk_ids, file, ensure_ascii=False, indent=2)
    print(f"Model: {MODEL_NAME}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Total chunks embedded: {len(chunks)}")
    print(f"Time taken: {time.perf_counter() - started:.2f} seconds")
    print(f"Saved array shape: {embeddings.shape}")
    print(f"Row count == ID count == chunk count: {len(embeddings) == len(chunk_ids) == len(chunks)}")


if __name__ == "__main__":
    main()
