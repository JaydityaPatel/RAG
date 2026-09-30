"""Wraps Pinecone managed vector DB operations using privacy-minimized metadata only."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

try:
    from src.embedder import get_embedding_model
except ModuleNotFoundError:
    from embedder import get_embedding_model

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = ROOT / "embeddings" / "chunk_embeddings.npy"
IDS_PATH = ROOT / "embeddings" / "chunk_ids.json"
MINIMIZED_CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks_minimized.json"
LOCAL_CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.json"


class PineconeBackend:
    def __init__(self, index_name="chatgpt-rag-learning", namespace="chatgpt-history"):
        load_dotenv(ROOT / ".env")
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is missing from .env")
        self.client = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.namespace = namespace
        self.index = None

    def create_index_if_not_exists(self, dimension=384, metric="cosine"):
        names = {index.name for index in self.client.list_indexes()}
        if self.index_name not in names:
            self.client.create_index(name=self.index_name, dimension=dimension, metric=metric,
                                     spec=ServerlessSpec(cloud="aws", region="us-east-1"))
            print(f"Created Pinecone index: {self.index_name}")
        else:
            print(f"Pinecone index already exists: {self.index_name}")
        while not self.client.describe_index(self.index_name).status.get("ready", False):
            time.sleep(2)
        self.index = self.client.Index(self.index_name)
        print("Pinecone index is ready.")
        return self.index

    def upload_minimized_chunks(self, batch_size=100):
        if not MINIMIZED_CHUNKS_PATH.exists():
            raise FileNotFoundError(f"Required privacy-minimized input is missing: {MINIMIZED_CHUNKS_PATH}")
        vectors = np.load(EMBEDDINGS_PATH)
        with IDS_PATH.open("r", encoding="utf-8") as file:
            chunk_ids = json.load(file)
        with MINIMIZED_CHUNKS_PATH.open("r", encoding="utf-8") as file:
            chunks = json.load(file)
        embedding_by_id = {chunk_id: vectors[index].tolist() for index, chunk_id in enumerate(chunk_ids)}
        started = time.perf_counter()
        uploaded = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            records = []
            for chunk in batch:
                metadata = {key: ("unknown" if value is None else value)
                            for key, value in chunk.get("metadata", {}).items()}
                records.append({"id": chunk["id"], "values": embedding_by_id[chunk["id"]], "metadata": metadata})
            self.index.upsert(vectors=records, namespace=self.namespace)
            uploaded += len(records)
        print(f"Uploaded {uploaded} vectors in {time.perf_counter() - started:.2f} seconds.")
        return uploaded

    def search(self, query_text, embedding_model, top_k=5):
        query_vector = next(iter(embedding_model.embed([query_text]))).tolist()
        response = self.index.query(vector=query_vector, namespace=self.namespace, top_k=top_k, include_metadata=True)
        return [{"id": match["id"], "score": float(match["score"]), "metadata": match.get("metadata", {})}
                for match in response.get("matches", [])]

    def enrich_with_local_text(self, pinecone_results, local_chunks_path=LOCAL_CHUNKS_PATH):
        with Path(local_chunks_path).open("r", encoding="utf-8") as file:
            local_chunks = {chunk["id"]: chunk for chunk in json.load(file)}
        # This is the privacy boundary in action — Pinecone only returned an id and a similarity score; all readable content is being resolved locally, never from the cloud.
        enriched = []
        for result in pinecone_results:
            chunk = local_chunks[result["id"]]
            metadata = chunk.get("metadata", {})
            enriched.append({"score": result["score"], "id": result["id"], "title": metadata.get("title"),
                             "text": chunk.get("text", "")[:200], "conversation_id": metadata.get("conversation_id"),
                             "create_date": metadata.get("create_date")})
        return enriched

    def delete_all_in_namespace(self):
        self.index.delete(delete_all=True, namespace=self.namespace)
        print(f"Deleted all vectors in namespace: {self.namespace}")

    def get_namespace_vector_count(self):
        stats = self.index.describe_index_stats()
        return stats.get("namespaces", {}).get(self.namespace, {}).get("vector_count", 0)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    backend = PineconeBackend()
    print("\nSTEP 1: create_index_if_not_exists()")
    backend.create_index_if_not_exists()
    print("\nSTEP 2: vector count before upload")
    print(backend.get_namespace_vector_count())
    print("\nSTEP 3: upload_minimized_chunks()")
    backend.upload_minimized_chunks()
    print("\nSTEP 4: vector count after upload")
    time.sleep(5)
    print(backend.get_namespace_vector_count())
    model = get_embedding_model()
    for query in ["What have I learned about AWS?", "Tips for saving water", "cybersecurity team names"]:
        print(f"\nSTEP 5: {query}")
        for result in backend.enrich_with_local_text(backend.search(query, model)):
            print(f"score={result['score']:.4f} | title={result['title']!r} | preview={result['text'][:150]}")
    print("\nSTEP 6: delete_all_in_namespace()")
    backend.delete_all_in_namespace()
    print("\nSTEP 7: vector count after deletion")
    time.sleep(5)
    print(backend.get_namespace_vector_count())
