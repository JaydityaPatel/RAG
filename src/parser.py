"""Parses sharded ChatGPT exports into linear conversations with valid messages."""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "conversations.json"


def to_iso(value):
    """Convert a Unix timestamp to an ISO 8601 UTC string, or return None."""
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def load_conversations(raw_dir=RAW_DIR):
    """Load and merge all conversations-* JSON shards from a directory."""
    conversations = []
    for path in sorted(raw_dir.glob("conversations-*.json")):
        with path.open("r", encoding="utf-8") as file:
            shard = json.load(file)
        if isinstance(shard, list):
            conversations.extend(shard)
    return conversations


def linearize(conversation):
    """Return the current-node path in root-to-current order and its branch count."""
    mapping = conversation.get("mapping") or {}
    node_id = conversation.get("current_node")
    path = []
    visited = set()
    while node_id and node_id not in visited:
        visited.add(node_id)
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            break
        path.append(node)
        node_id = node.get("parent")
    path.reverse()
    return path, max(0, len(mapping) - len(path))


def clean_conversation(conversation, skipped):
    """Build one cleaned record and update filtering counters."""
    path, branch_count = linearize(conversation)
    messages = []
    for node in path:
        message = node.get("message")
        if message is None:
            skipped["null_message"] += 1
            continue
        role = (message.get("author") or {}).get("role")
        if role in {"system", "tool"}:
            skipped[f"role_{role}"] += 1
            continue
        if role not in {"user", "assistant"}:
            skipped["unknown_role"] += 1
            continue
        content = message.get("content") or {}
        if content.get("content_type") != "text":
            skipped["non_text_content"] += 1
            continue
        parts = content.get("parts") or []
        text = "".join(part if isinstance(part, str) else str(part) for part in parts)
        messages.append({"role": role, "text": text, "create_time": to_iso(message.get("create_time"))})
    if not messages:
        return None
    return {
        "conversation_id": conversation.get("conversation_id"),
        "title": conversation.get("title"),
        "create_time": to_iso(conversation.get("create_time")),
        "update_time": to_iso(conversation.get("update_time")),
        "branch_count": branch_count,
        "messages": messages,
    }


def parse_export(raw_dir=RAW_DIR, output_path=OUTPUT_PATH):
    """Parse all raw shards, write cleaned conversations, and return summary data."""
    raw_conversations = load_conversations(raw_dir)
    skipped = Counter()
    dropped_reasons = Counter()
    cleaned = []
    for conversation in raw_conversations:
        record = clean_conversation(conversation, skipped)
        if record is None:
            dropped_reasons["no_valid_messages"] += 1
        else:
            cleaned.append(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(cleaned, file, ensure_ascii=False, indent=2)
        file.write("\n")
    dates = [record["create_time"] for record in cleaned if record["create_time"]]
    return {
        "raw_count": len(raw_conversations),
        "kept_count": len(cleaned),
        "dropped_count": len(raw_conversations) - len(cleaned),
        "dropped_reasons": dropped_reasons,
        "messages_kept": sum(len(record["messages"]) for record in cleaned),
        "non_text_skipped": skipped["non_text_content"],
        "earliest": min(dates) if dates else None,
        "latest": max(dates) if dates else None,
    }


if __name__ == "__main__":
    summary = parse_export()
    print(f"Total conversations found (raw): {summary['raw_count']}")
    print(f"Total conversations kept: {summary['kept_count']}")
    print(f"Total conversations dropped: {summary['dropped_count']}")
    print(f"Top drop reasons: {dict(summary['dropped_reasons'].most_common(5))}")
    print(f"Total messages kept: {summary['messages_kept']}")
    print(f"Total non-text content parts skipped: {summary['non_text_skipped']}")
    print(f"Date range of conversations: {summary['earliest']} to {summary['latest']}")
