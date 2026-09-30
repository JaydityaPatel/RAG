"""Minimizes local chunk data before any upload to a managed cloud vector DB."""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "chunks" / "chunks.json"
DEFAULT_OUTPUT = ROOT / "data" / "chunks" / "chunks_minimized.json"
DEFAULT_REPORT = ROOT / "data" / "chunks" / "privacy_report.txt"


def minimize_chunk(chunk):
    metadata = chunk["metadata"]
    # Text/title are readable personal content; role_mix/message_count are low-value and unnecessary for search.
    return {
        "id": chunk["id"],
        "metadata": {
            "conversation_id": metadata["conversation_id"],
            "create_date": metadata["create_date"] if metadata["create_date"] is not None else "unknown",
            "chunk_index": metadata["chunk_index"],
            "token_count": metadata["token_count"],
        },
    }


def minimize_all_chunks(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT):
    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    minimized = [minimize_chunk(chunk) for chunk in chunks]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(minimized, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return minimized


def _flatten_fields(record):
    fields = ["id"]
    if "text" in record:
        fields.append("text")
    fields.extend(f"metadata.{key}" for key in record["metadata"])
    return fields


def generate_privacy_report(input_path=DEFAULT_INPUT, output_minimized_path=DEFAULT_OUTPUT,
                            report_path=DEFAULT_REPORT):
    input_path = Path(input_path)
    output_minimized_path = Path(output_minimized_path)
    report_path = Path(report_path)
    with input_path.open("r", encoding="utf-8") as file:
        original = json.load(file)
    with output_minimized_path.open("r", encoding="utf-8") as file:
        minimized = json.load(file)

    before_fields = _flatten_fields(original[0]) if original else []
    after_fields = _flatten_fields(minimized[0]) if minimized else []
    excluded_count = len(set(before_fields) - set(after_fields))
    excluded_percent = excluded_count / len(before_fields) * 100 if before_fields else 0
    rng = random.Random(42)
    examples = rng.sample(list(zip(original, minimized)), min(3, len(original)))

    lines = [
        "Privacy Minimization Report",
        "============================",
        f"Total chunks processed: {len(original)}",
        f"Fields present before: {before_fields}",
        f"Fields present after: {after_fields}",
        "",
        "Random examples (before -> after):",
    ]
    for number, (before, after) in enumerate(examples, 1):
        lines.extend([
            f"\nExample {number}:",
            json.dumps(before, ensure_ascii=False, indent=2),
            "-- AFTER --",
            json.dumps(after, ensure_ascii=False, indent=2),
        ])
    lines.extend([
        "",
        f"{excluded_percent:.1f}% of original chunk data (by field count) is excluded from any cloud upload.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _contains_key(value, forbidden):
    if isinstance(value, dict):
        if any(key in value for key in forbidden):
            return True
        return any(_contains_key(child, forbidden) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


def verify_no_text_leakage(minimized_path=DEFAULT_OUTPUT):
    with Path(minimized_path).open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    failing_ids = [chunk.get("id", "<missing id>") for chunk in chunks if _contains_key(chunk, {"text", "title"})]
    if failing_ids:
        raise ValueError(f"Privacy leakage detected in chunk ids: {failing_ids}")
    return True


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        minimized = minimize_all_chunks()
        generate_privacy_report()
        verify_no_text_leakage()
        print(Path("data/chunks/chunks_minimized.json"), "exists:", DEFAULT_OUTPUT.exists())
        print(f"Minimized record count: {len(minimized)}")
        print("Sample record:")
        print(json.dumps(minimized[0], ensure_ascii=False, indent=2))
        print("PASSED")
        print("\n" + DEFAULT_REPORT.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"FAILED: {error}")
        raise
