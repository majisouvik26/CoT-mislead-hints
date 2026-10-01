#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

from cot_hints.inference import add_backend_arguments, normalize_backend_args


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_backend_arguments(p)
    p.add_argument("stage", choices=("prepare", "dev", "pilot", "extension", "sampling", "summarize", "review-export"))
    p.add_argument("--root", type=Path, default=Path("artifacts"))
    p.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"])
    p.add_argument("--experiment", choices=("dev", "pilot", "extension", "sampling"), default="pilot")
    p.add_argument("--config", default="configs/study.json")
    p.add_argument("--test-size", help="Only prepare: integer or all")
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42, help="Greedy-run seed")
    p.add_argument("--sampling-seeds", nargs="+", type=int, default=[42, 200, 300])
    p.add_argument("--thinking", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--dry-run", action="store_true", help="Print command sequence without execution")
    a = p.parse_args()
    try:
        normalize_backend_args(a)
    except ValueError as exc:
        p.error(str(exc))
    project = Path(__file__).resolve().parent
    data = a.root / "data"

    def run(script, *args):
        command = [sys.executable, str(project / script), *map(str, args)]
        import shlex
        print(shlex.join(command), flush=True)
        if not a.dry_run:
            subprocess.run(command, check=True)

    def slug(model):
        from cot_hints.common import digest
        return re.sub(r"[^a-zA-Z0-9._-]", "-", model) + "-" + digest([model, a.thinking])[:6]

    def run_path(experiment, model, seed):
        family = "greedy" if experiment in ("pilot", "extension") else experiment
        return a.root / "runs" / a.backend / family / f"{slug(model)}-s{seed}.jsonl"

    if a.stage == "prepare":
        extra = ["--test-size", a.test_size] if a.test_size else []
        run("prepare_data.py", "--config", a.config, "--output-dir", data, *extra)
        for split in ("dev", "test", "sampling"):
            run("build_prompts.py", "--data-dir", data, "--split", split)
        return
    if a.stage in ("dev", "pilot", "extension", "sampling"):
        experiment = a.stage
        split = "dev" if experiment == "dev" else "sampling" if experiment == "sampling" else "test"
        seeds = a.sampling_seeds if experiment == "sampling" else [a.seed]
        if len(set(seeds)) != len(seeds):
            p.error("Repeated sampling seeds")
        for model in a.models:
            for seed in seeds:
                path = run_path(experiment, model, seed)
                extra = ["--do-sample", "--temperature", "0.7", "--top-p", "0.9"] if experiment == "sampling" else []
                if experiment == "dev":
                    extra += ["--max-items", "10"]
                if a.backend == "vllm":
                    extra += ["--tensor-parallel-size", a.tensor_parallel_size,
                              "--gpu-memory-utilization", a.gpu_memory_utilization,
                              "--max-num-seqs", a.max_num_seqs,
                              "--enable-prefix-caching" if a.enable_prefix_caching else "--no-enable-prefix-caching"]
                    if a.max_model_len is not None:
                        extra += ["--max-model-len", a.max_model_len]
                    if a.enforce_eager:
                        extra += ["--enforce-eager"]
                    if a.batch_invariant:
                        extra += ["--batch-invariant"]
                run("run_eval.py", "--prompts", data / f"prompts_{split}.jsonl", "--model", model,
                    "--backend", a.backend,
                    "--run-id", path.stem + "-" + a.backend + "-" + path.parent.name, "--output", path,
                    "--conditions", "extension" if experiment == "extension" else "pilot", "--seed", seed,
                    "--batch-size", a.batch_size, "--max-new-tokens", a.max_new_tokens,
                    "--dtype", a.dtype, "--device", a.device, "--thinking", a.thinking, *extra)
        return
    if a.stage == "review-export":
        run("review.py", "export", "--scores-dir", a.root / "scores" / a.backend / a.experiment,
            "--output-dir", a.root / "review" / a.backend / a.experiment)
        return
    experiment = a.experiment
    split = "dev" if experiment == "dev" else "sampling" if experiment == "sampling" else "test"
    seeds = a.sampling_seeds if experiment == "sampling" else [a.seed]
    paths = [run_path(experiment, model, seed) for model in a.models for seed in seeds]
    scores = a.root / "scores" / a.backend / experiment
    run("score.py", "--prompts", data / f"prompts_{split}.jsonl", "--results", *paths,
        "--conditions", "extension" if experiment == "extension" else "pilot", "--output-dir", scores)
    run("analyze.py", "--scores-dir", scores, "--output-dir", a.root / "analysis" / a.backend / experiment,
        "--bootstrap", a.bootstrap)


if __name__ == "__main__":
    main()
