"""Wraps FAISS index creation, persistence, and similarity search."""

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

try:
    from src.embedder import get_embedding_model
except ModuleNotFoundError:
    from embedder import get_embedding_model

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = ROOT / "embeddings" / "chunk_embeddings.npy"
IDS_PATH = ROOT / "embeddings" / "chunk_ids.json"
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.json"
INDEX_DIR = ROOT / "indexes" / "faiss_index"
FLAT_PATH = INDEX_DIR / "flat.index"
IVF_PATH = INDEX_DIR / "ivf.index"
MODEL_NAME = "all-MiniLM-L6-v2"


class FaissBackend:
    def __init__(self):
        with IDS_PATH.open("r", encoding="utf-8") as file:
            self.chunk_ids = json.load(file)
        with CHUNKS_PATH.open("r", encoding="utf-8") as file:
            chunks = json.load(file)
        self.id_lookup = {chunk["id"]: chunk for chunk in chunks}

    @staticmethod
    def _normalized_embeddings():
        vectors = np.load(EMBEDDINGS_PATH).astype(np.float32, copy=True)
        faiss.normalize_L2(vectors)
        return vectors

    def build_flat_index(self):
        vectors = self._normalized_embeddings()
        # Unit-length vectors make inner product equal cosine similarity: dot(a, b) / (|a||b|) becomes dot(a, b).
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(FLAT_PATH))
        return index

    def build_ivf_index(self, nlist=50):
        vectors = self._normalized_embeddings()
        # nlist is the number of coarse clusters; training learns their centers before vectors are added.
        quantizer = faiss.IndexFlatIP(vectors.shape[1])
        index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.add(vectors)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(IVF_PATH))
        return index

    @staticmethod
    def load_flat_index():
        return faiss.read_index(str(FLAT_PATH))

    @staticmethod
    def load_ivf_index():
        return faiss.read_index(str(IVF_PATH))

    def search(self, index, query_text, embedding_model, top_k=5, nprobe=10):
        query_vector = np.asarray(
            next(iter(embedding_model.embed([query_text]))),
            dtype=np.float32,
        ).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        if hasattr(index, "nprobe"):
            index.nprobe = nprobe
        scores, positions = index.search(query_vector, top_k)
        results = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0 or position >= len(self.chunk_ids):
                continue
            chunk_id = self.chunk_ids[int(position)]
            chunk = self.id_lookup[chunk_id]
            results.append({
                "score": float(score),
                "id": chunk_id,
                "title": chunk.get("metadata", {}).get("title"),
                "text": chunk.get("text", "")[:200],
            })
        return results


def print_results(label, results):
    print(label)
    for rank, result in enumerate(results, 1):
        preview = result["text"].replace("\n", " ")[:150]
        print(f"{rank}. score={result['score']:.4f} | title={result['title']!r} | preview={preview}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    backend = FaissBackend()
    if not FLAT_PATH.exists():
        print("Building flat index...")
        backend.build_flat_index()
    if not IVF_PATH.exists():
        print("Building IVF index...")
        backend.build_ivf_index()

    flat_index = backend.load_flat_index()
    ivf_index = backend.load_ivf_index()
    model = get_embedding_model()
    queries = [
        "What have I learned about AWS?",
        "Tips for saving water",
        "cybersecurity team names",
    ]
    flat_results_by_query = []
    ivf_results_by_query = []
    flat_time = 0.0
    ivf_time = 0.0

    for query in queries:
        flat_started = time.perf_counter()
        flat_results = backend.search(flat_index, query, model)
        flat_time += time.perf_counter() - flat_started
        ivf_started = time.perf_counter()
        ivf_results = backend.search(ivf_index, query, model, nprobe=10)
        ivf_time += time.perf_counter() - ivf_started
        flat_results_by_query.append(flat_results)
        ivf_results_by_query.append(ivf_results)

        print(f"\n{'=' * 80}\nQUERY: {query}\n{'=' * 80}")
        print_results("FLAT RESULTS", flat_results)
        print_results("IVF RESULTS", ivf_results)

    print("\nOVERLAP COMPARISON")
    for query, flat_results, ivf_results in zip(queries, flat_results_by_query, ivf_results_by_query):
        flat_ids = {result["id"] for result in flat_results}
        overlap = sum(result["id"] in flat_ids for result in ivf_results)
        print(f"{query}: {overlap}/5 overlap")
    print(f"\nTOTAL FLAT SEARCH TIME: {flat_time:.4f} seconds")
    print(f"TOTAL IVF SEARCH TIME: {ivf_time:.4f} seconds")
