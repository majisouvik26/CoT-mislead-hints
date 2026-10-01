from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .common import (balanced_sample, digest, file_hash, read_json, read_jsonl,
                     unique_index, write_json, write_jsonl)
from .protocol import PROVIDED, labels, make_prompt, conditions, EXTENSION, PROTOCOL_VERSION


def content_key(row):
    # Generic stems recur with different options; those are different questions.
    # Sorting options still catches exact items whose answer order was permuted.
    normalize = lambda text: " ".join(text.split()).casefold()
    return digest([normalize(row["question"]), sorted(normalize(x) for x in row["options"])])


def validate_bundle(bundle, source_split):
    if set(bundle) != set(PROVIDED):
        raise ValueError(f"Expected exactly these condition splits: {PROVIDED}")
    indices = {}
    for condition, rows in bundle.items():
        for r in rows:
            required = {"question_id", "category", "question", "options", "gold_answer",
                        "target_answer", "hint", "messages", "condition"}
            if not required <= r.keys():
                raise ValueError(f"Missing fields: {required - r.keys()}")
            if not isinstance(r["question_id"], int) or isinstance(r["question_id"], bool):
                raise ValueError("question_id must be an integer")
            for field in ("category", "question"):
                if not isinstance(r[field], str) or not r[field].strip():
                    raise ValueError(f"Invalid {field}: {r['question_id']}")
            opts = r["options"]
            if not isinstance(opts, list) or not 2 <= len(opts) <= 26 or any(
                    not isinstance(x, str) or not x.strip() for x in opts):
                raise ValueError(f"Invalid options: {r['question_id']}")
            if any(not isinstance(r[key], str) or len(r[key]) != 1 or r[key] not in labels(opts)
                   for key in ("gold_answer", "target_answer")):
                raise ValueError(f"Gold/target out of range: {r['question_id']}")
            if r["gold_answer"] == r["target_answer"] or r["condition"] != condition:
                raise ValueError(f"Invalid target or condition: {r['question_id']}")
            if not isinstance(r["hint"], str) or bool(r["hint"]) != (condition != "no_hint"):
                raise ValueError(f"Invalid hint: {r['question_id']}")
            if "wrong_hint" in condition:
                expected = f"{r['target_answer']}: {opts[labels(opts).index(r['target_answer'])]}"
                if expected not in r["hint"]:
                    raise ValueError(f"Hint does not match fixed wrong target: {r['question_id']}")
            messages = r["messages"]
            if (not isinstance(messages, list) or len(messages) != 2 or
                    [m.get("role") for m in messages] != ["system", "user"] or
                    any(not isinstance(m.get("content"), str) for m in messages)):
                raise ValueError("Expected one system and one user message")
        indices[condition] = unique_index(rows, lambda x: x["question_id"])
    ids = set(indices["no_hint"])
    if not ids or any(set(index) != ids for index in indices.values()):
        raise ValueError("Empty or unmatched question IDs across condition splits")
    items = []
    fields = ("category", "question", "options", "gold_answer", "target_answer")
    for qid in sorted(ids):
        base = indices["no_hint"][qid]
        for condition in PROVIDED:
            row = indices[condition][qid]
            if any(row[k] != base[k] for k in fields):
                raise ValueError(f"Unpaired item {qid}: {condition}")
            if row["messages"][0] != base["messages"][0]:
                raise ValueError(f"System prompts differ across conditions: {qid}")
        item = {k: base[k] for k in fields}
        item.update(question_id=qid, item_id=f"mmlu-pro:{source_split}:{qid}",
                    source_split=source_split,
                    source_conditions={c: {"hint": indices[c][qid]["hint"],
                                           "messages": indices[c][qid]["messages"]} for c in PROVIDED})
        item["content_hash"] = content_key(item)
        items.append(item)
    return items


def load_local(directory, expected_split):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    split = manifest.get("source_split", manifest.get("split"))
    if split != expected_split:
        raise ValueError(f"{directory} source split is {split!r}, expected {expected_split!r}")
    bundle, hashes = {}, {}
    for condition in PROVIDED:
        path = directory / f"{condition}.jsonl"
        hashes[path.name] = file_hash(path)
        advertised = manifest.get("files", {}).get(path.name, {})
        if advertised.get("sha256") and advertised["sha256"] != hashes[path.name]:
            raise ValueError(f"Source checksum mismatch: {path}")
        bundle[condition] = read_jsonl(path)
    return bundle, {"kind": "local", "path": str(directory), "files": hashes, "manifest": manifest}


def load_hub(config):
    from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
    from huggingface_hub import HfApi, hf_hub_download
    repo = config["dataset_id"]
    revision = HfApi().dataset_info(repo, revision=config["dataset_revision"]).sha
    name = config["dataset_config"]
    names = get_dataset_config_names(repo, revision=revision)
    if name not in names:
        raise ValueError(f"Config {name!r} not present; available: {names}")
    splits = get_dataset_split_names(repo, config_name=name, revision=revision)
    if set(splits) != set(PROVIDED):
        raise ValueError(f"Condition splits changed: {splits}; inspect before updating the protocol")
    source_manifest = read_json(hf_hub_download(repo, "manifest.json", repo_type="dataset", revision=revision))
    if source_manifest.get("source_split", source_manifest.get("split")) != "test":
        raise ValueError("Hub manifest does not identify source test questions")
    bundle, fingerprints = {}, {}
    for split in PROVIDED:
        dataset = load_dataset(repo, name=name, split=split, revision=revision)
        bundle[split] = list(dataset)
        fingerprints[split] = dataset._fingerprint
    return bundle, {"kind": "huggingface", "dataset_id": repo, "config": name,
                    "revision": revision, "fingerprints": fingerprints, "manifest": source_manifest}


def audit(items):
    return {"count": len(items), "categories": dict(Counter(r["category"] for r in items)),
            "gold_letters": dict(Counter(r["gold_answer"] for r in items)),
            "target_letters": dict(Counter(r["target_answer"] for r in items)),
            "option_counts": dict(Counter(str(len(r["options"])) for r in items)),
            "source_messages_with_literal_backslash_n": sum(
                any("\\n" in m["content"] for c in r["source_conditions"].values() for m in c["messages"])
                for r in items)}


def prepare_main():
    p = argparse.ArgumentParser(description="Freeze paired HF test questions and disjoint local validation items.")
    p.add_argument("--config", default="configs/study.json")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/data"))
    p.add_argument("--source-dir", type=Path, help="Offline four-JSONL TEST bundle with manifest.json")
    p.add_argument("--dev-dir", type=Path)
    p.add_argument("--test-size", help="Integer or 'all'; default from config")
    p.add_argument("--dev-size", type=int)
    p.add_argument("--sampling-size", type=int)
    p.add_argument("--categories", nargs="+", help="Exact MMLU-Pro categories, e.g. math philosophy")
    a = p.parse_args()
    cfg = read_json(a.config)
    for field in ("dev_size", "sampling_size", "categories"):
        if getattr(a, field) is not None:
            cfg[field] = getattr(a, field)
    if a.test_size is not None:
        cfg["test_size"] = None if a.test_size == "all" else int(a.test_size)
    if a.dev_dir:
        cfg["dev_dir"] = str(a.dev_dir)
    if (a.output_dir / "study.json").exists():
        raise ValueError("Study already frozen. Reuse it or select a new --output-dir.")
    bundle, provenance = load_local(a.source_dir, "test") if a.source_dir else load_hub(cfg)
    dev_bundle, dev_provenance = load_local(cfg["dev_dir"], "validation")
    test = validate_bundle(bundle, "test")
    dev = validate_bundle(dev_bundle, "validation")
    categories = cfg.get("categories", [])
    if categories:
        for population in (test, dev):
            missing = set(categories) - {r["category"] for r in population}
            if missing:
                raise ValueError(f"Categories not found: {sorted(missing)}")
        test = [r for r in test if r["category"] in categories]
        dev = [r for r in dev if r["category"] in categories]
    # Check the entire chosen source populations, before selection.
    overlap = {r["content_hash"] for r in test} & {r["content_hash"] for r in dev}
    if overlap:
        raise ValueError(f"{len(overlap)} overlapping question/option sets between test and validation")
    duplicate_audit = {}
    for name, population in (("test", test), ("dev", dev)):
        seen, kept, dropped = {}, [], []
        for row in population:
            key = row["content_hash"]
            if key in seen:
                prior = seen[key]
                answer = lambda r: " ".join(r["options"][labels(r["options"]).index(r["gold_answer"])].split()).casefold()
                if answer(row) != answer(prior):
                    raise ValueError(f"Duplicate question/option set with conflicting gold labels: {row['item_id']}")
                dropped.append(row["item_id"])
            else:
                seen[key] = row
                kept.append(row)
        population[:] = kept
        duplicate_audit[name] = dropped
    test = balanced_sample(test, cfg["test_size"], cfg["data_seed"])
    dev = balanced_sample(dev, cfg["dev_size"], cfg["data_seed"])
    sample = balanced_sample(test, min(cfg["sampling_size"], len(test)), cfg["sampling_seed"])
    study = {"schema_version": 1, "config": cfg, "provenance": provenance,
             "dev_provenance": dev_provenance, "sampling_item_ids": [r["item_id"] for r in sample],
             "sampling_selection": "Outcome-independent equal-category sample before inference",
             "deduplicated_ids": duplicate_audit, "audits": {"test": audit(test), "dev": audit(dev)},
             "deviations_from_pdf": ["MMLU-Pro, all selected categories, variable option count",
                                     "Existing fixed targets retained; no target resampling",
                                     "Category-balanced subset unless test_size is all"],
             "item_hashes": {"test": digest(test), "dev": digest(dev)}}
    study["study_id"] = digest(study)
    for split, rows in (("test", test), ("dev", dev)):
        write_jsonl(a.output_dir / f"{split}.jsonl", rows, immutable=True)
    write_json(a.output_dir / "study.json", study, immutable=True)
    print(f"Frozen {len(test)} test, {len(dev)} dev, {len(sample)} sampling items in {a.output_dir}")
    print("Gold/target/category counts:", study["audits"])


def prompts_main():
    p = argparse.ArgumentParser(description="Build frozen prompts; default seven-condition bank allows pilot reuse.")
    p.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    p.add_argument("--split", choices=("dev", "test", "sampling"), default="test")
    p.add_argument("--conditions", default="extension", help="pilot, provided, extension, or comma-separated names")
    p.add_argument("--protocol", choices=(PROTOCOL_VERSION, "source-verbatim"), default=PROTOCOL_VERSION)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    study = read_json(a.data_dir / "study.json")
    split = "test" if a.split == "sampling" else a.split
    items = read_jsonl(a.data_dir / f"{split}.jsonl")
    if digest(items) != study["item_hashes"][split]:
        raise ValueError("Frozen manifest changed")
    if a.split == "sampling":
        ids = set(study["sampling_item_ids"])
        items = [r for r in items if r["item_id"] in ids]
    chosen = conditions(a.conditions)
    prompts = [dict(make_prompt(item, c, a.protocol), study_id=study["study_id"],
                    evaluation_split=a.split) for item in items for c in chosen]
    output = a.output or a.data_dir / f"prompts_{a.split}.jsonl"
    write_jsonl(output, prompts, immutable=True)
    print(f"Wrote {len(prompts)} prompts to {output}; protocol={a.protocol}")
