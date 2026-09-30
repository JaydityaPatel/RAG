"""Combines retrieval with a Gemini call to answer questions with source citations."""

import os
import sys
import time

from google import genai
from dotenv import load_dotenv

try:
    from src.retriever import Retriever
except ModuleNotFoundError:
    from retriever import Retriever

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_API_KEY = os.getenv("LLM_API_KEY")
if LLM_API_KEY:
    client = genai.Client(api_key=LLM_API_KEY)
else:
    client = None


def build_prompt(question, retrieved_chunks):
    """Build a grounded prompt containing full retrieved chunk context."""
    sections = [
        "You are answering a question using ONLY the provided context from the user's own ChatGPT conversation history.",
        "If the context does not contain enough information to answer, say so clearly. Do not make anything up.",
        "After the answer, list the source chunks used, formatted as [Source: <title>, <create_date>].",
        "\nCONTEXT:\n",
    ]
    for index, chunk in enumerate(retrieved_chunks, 1):
        sections.append(
            f"--- SOURCE CHUNK {index} ---\nTitle: {chunk['title']}\n"
            f"Create date: {chunk['create_date']}\nFull text:\n{chunk['text']}\n"
            f"--- END SOURCE CHUNK {index} ---\n"
        )
    sections.append(f"\nQUESTION:\n{question}")
    return "\n".join(sections)


def answer_question(question, retriever, top_k=5):
    """Retrieve context, ask Gemini, and return the answer with source metadata."""
    retrieval_started = time.perf_counter()
    retrieved = retriever.retrieve(question, top_k=top_k)
    retrieval_seconds = time.perf_counter() - retrieval_started
    sources = [
        {"title": chunk["title"], "create_date": chunk["create_date"],
         "score": chunk["score"], "conversation_id": chunk["conversation_id"]}
        for chunk in retrieved
    ]
    result = {"question": question, "answer": None, "sources": sources}
    try:
        if client is None:
            raise RuntimeError("LLM_API_KEY is not set in .env")
        generation_started = time.perf_counter()
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=build_prompt(question, retrieved),
        )
        result["answer"] = response.text
        print(
            f"RAG timings: retrieval={retrieval_seconds:.3f}s, "
            f"Gemini ({LLM_MODEL})={time.perf_counter() - generation_started:.3f}s"
        )
    except Exception as error:
        print(f"RAG timings: retrieval={retrieval_seconds:.3f}s; Gemini error: {error}")
        result["error"] = str(error)
    return result


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    questions = [
        "What have I learned about DevOps?",
        "What problems did I face while learning OCPP?",
        "What was my RAG learning plan?",
    ]
    retriever = Retriever(backend="faiss")
    for index, question in enumerate(questions):
        result = answer_question(question, retriever)
        print(f"\n{'=' * 80}\nQUESTION: {question}\n{'=' * 80}")
        print(result.get("answer") or f"Error: {result.get('error', 'No answer returned.')}")
        print("\nSOURCES:")
        for source in result["sources"]:
            print(f"- {source['title']} ({source['create_date']})")
        if index < len(questions) - 1:
            time.sleep(2)
