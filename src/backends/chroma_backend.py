"""Wraps Chroma local vector DB collection creation, persistence, and querying with metadata filtering."""

import json
import os
import sys
from pathlib import Path

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

try:
    from src.embedder import MODEL_NAME, get_embedding_model
except ModuleNotFoundError:
    from embedder import MODEL_NAME, get_embedding_model

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.json"
DEFAULT_CHROMA_PATH = ROOT / "indexes" / "chroma_db"


def configured_path(environment_name, default_path):
    configured = os.getenv(environment_name)
    path = Path(configured) if configured else default_path
    return path if path.is_absolute() else ROOT / path


class FastEmbedEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma-compatible wrapper around the exact MiniLM model in FastEmbed."""

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = get_embedding_model()

    def __call__(self, input: Documents) -> Embeddings:
        return [vector.tolist() for vector in self.model.embed(input)]

    @staticmethod
    def name() -> str:
        return "fastembed-all-minilm-l6-v2"

    @staticmethod
    def build_from_config(config):
        return FastEmbedEmbeddingFunction(**config)

    def get_config(self):
        return {"model_name": self.model_name}


class ChromaBackend:
    def __init__(self, chroma_path=None, chunks_path=None):
        self.client = None
        self.collection = None
        self.embedding_function = FastEmbedEmbeddingFunction()
        self.chroma_path = Path(chroma_path) if chroma_path else configured_path(
            "CHROMA_DB_PATH", DEFAULT_CHROMA_PATH
        )
        self.chunks_path = Path(chunks_path) if chunks_path else configured_path(
            "CHUNKS_JSON_PATH", DEFAULT_CHUNKS_PATH
        )

    def build_collection(self, collection_name="chatgpt_chunks"):
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # Existing collections persist their original function metadata.
            # We pass explicit vectors below so querying never changes stored data.
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        if self.collection.count() > 0:
            print(f"Loaded existing Chroma collection '{collection_name}' with {self.collection.count()} chunks.")
            return self.collection

        with self.chunks_path.open("r", encoding="utf-8") as file:
            chunks = json.load(file)
        for start in range(0, len(chunks), 100):
            batch = chunks[start:start + 100]
            metadata = [
                {
                    key: ("unknown" if value is None else value)
                    # Chroma accepts only scalar metadata values; null becomes "unknown".
                    for key, value in chunk["metadata"].items()
                }
                for chunk in batch
            ]
            self.collection.add(
                ids=[chunk["id"] for chunk in batch],
                documents=[chunk["text"] for chunk in batch],
                metadatas=metadata,
                embeddings=self.embedding_function([chunk["text"] for chunk in batch]),
            )
        print(f"Built Chroma collection '{collection_name}' with {self.collection.count()} chunks.")
        return self.collection

    def search(self, query_text, top_k=5, where_filter=None):
        kwargs = {
            "query_embeddings": self.embedding_function([query_text]),
            "n_results": top_k,
        }
        if where_filter is not None:
            kwargs["where"] = where_filter
        response = self.collection.query(**kwargs)
        results = []
        for distance, chunk_id, document, metadata in zip(
            response["distances"][0], response["ids"][0], response["documents"][0], response["metadatas"][0]
        ):
            # Cosine distance is lower for closer vectors, so 1 - distance is similarity-like.
            results.append({
                "score": float(1.0 - distance),
                "id": chunk_id,
                "title": metadata.get("title"),
                "text": document[:200],
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

    backend = ChromaBackend()
    collection = backend.build_collection()
    queries = [
        "What have I learned about AWS?",
        "Tips for saving water",
        "cybersecurity team names",
    ]
    for query in queries:
        print(f"\nQUERY: {query}")
        print_results("CHROMA RESULTS", backend.search(query, top_k=5))

    filter_value = {"role_mix": "assistant"}
    assistant_chunks = collection.get(where=filter_value, include=[])
    print(f"\nTotal chunks with role_mix=assistant: {len(assistant_chunks['ids'])}")
    print("FILTERED RESULTS (role_mix=assistant only)")
    print_results("", backend.search("DevOps", top_k=5, where_filter=filter_value))
