"""Benchmark FAISS Flat, Chroma, and Pinecone retrieval.

The report measures agreement with the FAISS Flat index as a reference, not
ground-truth accuracy: this dataset has no human-labeled correct answers.
"""

import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backends.chroma_backend import ChromaBackend
from src.backends.faiss_backend import FaissBackend
from src.backends.pinecone_backend import PineconeBackend


QUERIES = [
    "What have I learned about AWS?",
    "Tips for saving water",
    "cybersecurity team names",
    "What have I learned about DevOps?",
    "What problems did I face while learning OCPP?",
    "What was my RAG learning plan?",
    "Find conversations where I discussed AWS.",
    "What decisions did I make about my monitor purchase?",
]


def timed_search(search_function):
    started = time.perf_counter()
    result = search_function()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return result, elapsed_ms


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_backend = FaissBackend()
    flat_index = faiss_backend.load_flat_index()
    chroma_backend = ChromaBackend()
    chroma_backend.build_collection()
    pinecone_backend = PineconeBackend()
    pinecone_backend.create_index_if_not_exists()
    pinecone_count = pinecone_backend.get_namespace_vector_count()
    if pinecone_count == 0:
        print("Pinecone re-upload needed: yes")
        pinecone_backend.upload_minimized_chunks()
    else:
        print(f"Pinecone re-upload needed: no ({pinecone_count} vectors already present)")

    records = []
    for query in QUERIES:
        query_vector = model.encode(query, convert_to_numpy=True).astype(np.float32).reshape(1, -1)
        faiss_vector = query_vector.copy()
        faiss.normalize_L2(faiss_vector)

        flat, flat_ms = timed_search(lambda: flat_index.search(faiss_vector, 5))
        chroma, chroma_ms = timed_search(
            lambda: chroma_backend.collection.query(query_embeddings=query_vector.tolist(), n_results=5)
        )
        pinecone, pinecone_ms = timed_search(
            lambda: pinecone_backend.index.query(
                vector=query_vector[0].tolist(), namespace=pinecone_backend.namespace,
                top_k=5, include_metadata=True
            )
        )

        flat_ids = [faiss_backend.chunk_ids[int(position)] for position in flat[1][0] if position >= 0]
        chroma_ids = chroma["ids"][0]
        pinecone_ids = [match["id"] for match in pinecone.get("matches", [])]
        records.append({
            "query": query,
            "flat_ids": flat_ids,
            "chroma_ids": chroma_ids,
            "pinecone_ids": pinecone_ids,
            "flat_ms": flat_ms,
            "chroma_ms": chroma_ms,
            "pinecone_ms": pinecone_ms,
        })

    latencies = {
        name: [record[f"{name}_ms"] for record in records]
        for name in ("flat", "chroma", "pinecone")
    }
    overlaps = {
        "flat_chroma": [len(set(r["flat_ids"]) & set(r["chroma_ids"])) for r in records],
        "flat_pinecone": [len(set(r["flat_ids"]) & set(r["pinecone_ids"])) for r in records],
        "chroma_pinecone": [len(set(r["chroma_ids"]) & set(r["pinecone_ids"])) for r in records],
    }
    rank1_agreement = sum(
        bool(r["flat_ids"] and r["chroma_ids"] and r["pinecone_ids"]
             and r["flat_ids"][0] == r["chroma_ids"][0] == r["pinecone_ids"][0])
        for r in records
    )

    print("\n" + "=" * 60)
    print("BENCHMARK REPORT — FAISS (Flat, reference) vs Chroma vs Pinecone")
    print("=" * 60)
    print("Note: 'accuracy' below means agreement with FAISS Flat's exact brute-force results, not ground-truth correctness (no human-labeled correct answers exist for this dataset).")
    print("\nLATENCY (ms, average across 8 queries):")
    for name, label in (("flat", "FAISS"), ("chroma", "Chroma"), ("pinecone", "Pinecone")):
        values = latencies[name]
        print(f"  {label:8} {sum(values) / len(values):.2f} (min {min(values):.2f} / max {max(values):.2f})")
    print("\nTOP-5 AGREEMENT (average overlap out of 5, across 8 queries):")
    print(f"  FAISS vs Chroma:   {sum(overlaps['flat_chroma']) / 8:.2f}/5")
    print(f"  FAISS vs Pinecone: {sum(overlaps['flat_pinecone']) / 8:.2f}/5")
    print(f"  Chroma vs Pinecone: {sum(overlaps['chroma_pinecone']) / 8:.2f}/5")
    print(f"\nRANK-1 AGREEMENT: {rank1_agreement} out of 8 queries had all three backends agree on the top result.")
    print("\nPER-QUERY BREAKDOWN:")
    for index, record in enumerate(records, 1):
        print(f"\n{index}. {record['query']}")
        print(f"   FAISS: {record['flat_ms']:.2f} ms | Chroma: {record['chroma_ms']:.2f} ms | Pinecone: {record['pinecone_ms']:.2f} ms")
        print(f"   Overlap: FAISS/Chroma {len(set(record['flat_ids']) & set(record['chroma_ids']))}/5 | FAISS/Pinecone {len(set(record['flat_ids']) & set(record['pinecone_ids']))}/5 | Chroma/Pinecone {len(set(record['chroma_ids']) & set(record['pinecone_ids']))}/5")

    max_latency = max(max(values) for values in latencies.values())
    print("\nRELATIVE LATENCY")
    for name, label in (("flat", "FAISS"), ("chroma", "Chroma"), ("pinecone", "Pinecone")):
        average = sum(latencies[name]) / len(latencies[name])
        bar = "#" * max(1, round(20 * average / max_latency))
        print(f"{label:8} [{bar:<20}] {average:.2f} ms")
    print("\nNOTE: Pinecone data is still uploaded in namespace 'chatgpt-history'. Run pinecone_backend.py's delete step manually if you want to clean up.")


if __name__ == "__main__":
    main()
