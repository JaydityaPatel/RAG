"""Teaching example of cosine similarity search using only NumPy for the math."""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = ROOT / "embeddings" / "chunk_embeddings.npy"
IDS_PATH = ROOT / "embeddings" / "chunk_ids.json"
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.json"


def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)  # Multiply matching dimensions and add the results.
    norm_a = np.linalg.norm(vec_a)  # Compute the length of the first vector.
    norm_b = np.linalg.norm(vec_b)  # Compute the length of the second vector.
    denominator = norm_a * norm_b  # Combine both vector lengths for normalization.
    return float(dot_product / denominator) if denominator else 0.0  # Divide by both vector lengths.


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    embeddings = np.load(EMBEDDINGS_PATH)
    with IDS_PATH.open("r", encoding="utf-8") as file:
        chunk_ids = json.load(file)
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
    rng = np.random.default_rng(42)
    selected_indices = rng.choice(len(embeddings), size=100, replace=False)
    subset_embeddings = embeddings[selected_indices]
    subset_chunks = [chunks_by_id[chunk_ids[index]] for index in selected_indices]

    title_terms = ("save water", "cybersuraksha", "team names")
    matching_ids = [
        chunk["id"]
        for chunk in subset_chunks
        if any(term in chunk.get("metadata", {}).get("title", "").lower() for term in title_terms)
    ]
    print(f"Target title present in random subset: {'yes' if matching_ids else 'no'}")
    print(f"Matching chunk ids: {matching_ids}")

    model = SentenceTransformer(MODEL_NAME)
    queries = [
        "What have I learned about AWS?",
        "Tips for saving water",
        "cybersecurity team names",
    ]
    all_results = []
    for query in queries:
        query_embedding = model.encode(query, convert_to_numpy=True)
        results = []
        for index in range(len(subset_embeddings)):
            score = cosine_similarity(query_embedding, subset_embeddings[index])
            results.append((score, subset_chunks[index]))
        results.sort(key=lambda item: item[0], reverse=True)
        all_results.append(results)

        print(f"\nQuery: {query}")
        for rank, (score, chunk) in enumerate(results[:5], 1):
            title = chunk.get("metadata", {}).get("title")
            preview = chunk["text"][:150].replace("\n", " ")
            print(f"{rank}. score={score:.4f} | title={title!r} | preview={preview}")

    lowest_score, lowest_chunk = all_results[0][-1]
    print("\nLowest-scoring chunk for the first query:")
    print(f"score={lowest_score:.4f} | title={lowest_chunk.get('metadata', {}).get('title')!r}")
    print(f"preview={lowest_chunk['text'][:150].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
