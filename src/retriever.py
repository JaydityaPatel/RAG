"""Common retrieval interface over the project's FAISS and Chroma backends."""

import json
import os
from pathlib import Path

try:
    from src.backends.chroma_backend import ChromaBackend
    from src.backends.faiss_backend import FaissBackend
except ModuleNotFoundError:
    from backends.chroma_backend import ChromaBackend
    from backends.faiss_backend import FaissBackend
from src.embedder import get_embedding_model

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.json"


def configured_chunks_path():
    configured = os.getenv("CHUNKS_JSON_PATH")
    path = Path(configured) if configured else DEFAULT_CHUNKS_PATH
    return path if path.is_absolute() else ROOT / path


class Retriever:
    def __init__(self, backend="chroma", **kwargs):
        if backend not in {"chroma", "faiss"}:
            raise ValueError("Unsupported backend. Choose 'chroma' or 'faiss'.")
        with configured_chunks_path().open("r", encoding="utf-8") as file:
            chunks = json.load(file)
        self.id_lookup = {chunk["id"]: chunk for chunk in chunks}
        self.backend_name = backend
        self.backend = ChromaBackend() if backend == "chroma" else FaissBackend()
        if backend == "chroma":
            self.backend.build_collection(kwargs.get("collection_name", "chatgpt_chunks"))
            self.embedding_model = None
            self.index = None
        else:
            self.index = self.backend.load_flat_index()
            self.embedding_model = kwargs.get("embedding_model") or get_embedding_model()

    def retrieve(self, query_text, top_k=5):
        if self.backend_name == "chroma":
            raw_results = self.backend.search(query_text, top_k=top_k)
        else:
            raw_results = self.backend.search(self.index, query_text, self.embedding_model, top_k=top_k)

        results = []
        for result in raw_results:
            chunk = self.id_lookup[result["id"]]
            metadata = chunk.get("metadata", {})
            results.append({
                "score": float(result["score"]),
                "id": result["id"],
                "title": metadata.get("title"),
                "text": chunk.get("text", ""),
                "conversation_id": metadata.get("conversation_id"),
                "create_date": metadata.get("create_date"),
            })
        return results
