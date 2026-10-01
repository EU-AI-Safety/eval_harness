from __future__ import annotations

import csv
import json
from threading import Lock
from pathlib import Path
from typing import Any, Iterable


_APPEND_LOCK = Lock()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL content in {path}: {line[:120]}") from exc


def load_jsonl_index(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row[key]: row for row in iter_jsonl(path)}


def rewrite_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically rewrite a JSONL file with the given rows (caller chooses order)."""
    ensure_parent_dir(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def compact_jsonl_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
    """Keep the last record per key and rewrite the file. Returns the compacted index."""
    index = load_jsonl_index(path, key)
    if path.exists():
        rewrite_jsonl(path, index.values())
    return index