from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON; repair an interrupted ledger with run_eval.py") from exc
    return rows


def write_text(path, text, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Refusing to change frozen file: {path}. Use a new output directory.")
        return
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def write_json(path, obj, immutable=False):
    write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", immutable)


def write_jsonl(path, rows, immutable=False):
    write_text(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), immutable)


def write_csv(path, rows, fields=None):
    import io
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    write_text(path, buffer.getvalue())


def unique_index(rows, key):
    index = {}
    for row in rows:
        k = key(row)
        if k in index:
            raise ValueError(f"Duplicate key: {k}")
        index[k] = row
    return index


def balanced_sample(rows, count, seed):
    """Sample without outcomes, approximately equally across available categories."""
    rows = sorted(rows, key=lambda r: str(r.get("item_id", r.get("question_id"))))
    if count is None:
        return rows
    if not 0 < count <= len(rows):
        raise ValueError(f"Requested {count} items from {len(rows)} available")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    for group in groups.values():
        rng.shuffle(group)
    names = sorted(groups)
    rng.shuffle(names)
    selected = []
    while len(selected) < count:
        for name in names:
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop())
    return sorted(selected, key=lambda r: str(r.get("item_id", r.get("question_id"))))


def versions():
    import importlib.metadata
    import platform
    result = {"python": platform.python_version()}
    for name in ("torch", "transformers", "datasets", "huggingface-hub", "accelerate",
                 "numpy", "matplotlib", "bitsandbytes", "filelock", "vllm"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def repair_tail(path):
    """Only an unterminated final JSONL fragment is repairable; save its exact bytes."""
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return
    raw = path.read_bytes()
    if raw.endswith(b"\n"):
        return
    offset = raw.rfind(b"\n") + 1
    tail = raw[offset:]
    try:
        json.loads(tail)
    except (json.JSONDecodeError, UnicodeDecodeError):
        backup = path.with_name(path.name + ".interrupted-" + digest(tail.hex())[:12])
        backup.write_bytes(tail)
        with path.open("r+b") as stream:
            stream.truncate(offset)
        print(f"Saved interrupted final line to {backup}")
    else:
        with path.open("ab") as stream:
            stream.write(b"\n")


def append_batch(path, records):
    with Path(path).open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
