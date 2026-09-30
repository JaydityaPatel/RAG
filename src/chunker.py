"""Cleans conversations and packs messages into metadata-rich chunks."""

import json
import random
import re
import sys
from pathlib import Path

INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "conversations.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks" / "chunks.json"
MAX_TOKENS = 450


def approximate_token_count(text):
    # This is an approximation; exact counts will follow once we choose an embedding model in Phase 3.
    return int(round(len(text.split()) * 1.3))


def clean_text(text):
    """Remove excess whitespace while preserving markdown, including code syntax."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_message(message):
    role = message["role"].capitalize()
    return f"{role}: {message['text']}"


def role_mix(messages):
    return "+".join(role for role in ("user", "assistant") if role in {m["role"] for m in messages})


def make_chunk(conversation, messages, index):
    text = "\n\n".join(format_message(message) for message in messages)
    return {
        "id": f"{conversation['conversation_id']}_chunk{index}",
        "text": text,
        "metadata": {
            "conversation_id": conversation["conversation_id"],
            "title": conversation.get("title"),
            "create_date": (conversation.get("create_time") or "")[:10] or None,
            "role_mix": role_mix(messages),
            "chunk_index": index,
            "token_count": approximate_token_count(text),
            "message_count": len(messages),
        },
    }


def chunk_conversation(conversation):
    """Chunk one conversation and return its chunks plus oversized-message count."""
    messages = [
        {**message, "text": clean_text(message.get("text", ""))}
        for message in conversation.get("messages", [])
    ]
    chunks = []
    current = []
    current_tokens = 0
    oversized = 0

    def emit(items):
        if items:
            chunks.append(make_chunk(conversation, items, len(chunks)))

    for message in messages:
        message_tokens = approximate_token_count(message["text"])
        if message_tokens > MAX_TOKENS:
            if current and len(current) > 1:
                # Keep the same one-message overlap even when the next message
                # is oversized; the oversized message itself remains unsplit.
                emit(current)
                current = [current[-1], message]
                current_tokens = sum(approximate_token_count(item["text"]) for item in current)
            else:
                emit(current)
                current = [message]
                current_tokens = message_tokens
            oversized += 1
            continue

        if current and current_tokens + message_tokens > MAX_TOKENS:
            previous = current[-1:] if len(current) > 1 else []
            emit(current)
            current = previous + [message]
            current_tokens = sum(approximate_token_count(item["text"]) for item in current)
        else:
            current.append(message)
            current_tokens += message_tokens

    emit(current)
    return chunks, oversized


def build_chunks(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    """Read processed conversations, write all chunks, and return summary data."""
    with input_path.open("r", encoding="utf-8") as file:
        conversations = json.load(file)

    all_chunks = []
    oversized_count = 0
    for conversation in conversations:
        chunks, oversized = chunk_conversation(conversation)
        all_chunks.extend(chunks)
        oversized_count += oversized

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(all_chunks, file, ensure_ascii=False, indent=2)
        file.write("\n")

    token_counts = [chunk["metadata"]["token_count"] for chunk in all_chunks]
    return {
        "conversation_count": len(conversations),
        "chunk_count": len(all_chunks),
        "average_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "oversized_count": oversized_count,
        "chunks": all_chunks,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    summary = build_chunks()
    print(f"Total conversations processed: {summary['conversation_count']}")
    print(f"Total chunks produced: {summary['chunk_count']}")
    print(
        "Token count average / min / max: "
        f"{summary['average_tokens']:.2f} / {summary['min_tokens']} / {summary['max_tokens']}"
    )
    print(f"Oversized single-message chunks: {summary['oversized_count']}")
    print("\nSample chunks:")
    for number, chunk in enumerate(random.sample(summary["chunks"], min(2, len(summary["chunks"]))), 1):
        print(f"\n--- Sample {number}: {chunk['id']} ---")
        print(chunk["text"])
