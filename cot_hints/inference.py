from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from .common import (append_batch, digest, file_hash, read_json, read_jsonl, repair_tail,
                     unique_index, utc_now, versions, write_json)
from .protocol import conditions, labels, parse_answer


def add_backend_arguments(parser):
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--batch-size", type=int, help="Flush/request batch: default 128 for vLLM, 2 for Transformers")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM GPUs per checkpoint")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM fraction of GPU memory")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="vLLM concurrent sequences")
    parser.add_argument("--max-model-len", type=int, help="vLLM context window; default checkpoint limit")
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action="store_true", help="vLLM: disable graph capture, e.g. for debugging")
    parser.add_argument("--batch-invariant", action="store_true", help="vLLM experimental invariance; requires supported GPU/model")


def normalize_backend_args(args):
    if args.batch_size is None:
        args.batch_size = 128 if args.backend == "vllm" else 2
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.tensor_parallel_size < 1 or args.max_num_seqs < 1:
        raise ValueError("--tensor-parallel-size and --max-num-seqs must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.max_model_len is not None and args.max_model_len < 1:
        raise ValueError("--max-model-len must be positive")
    if args.backend == "vllm":
        if args.device not in ("auto", "cuda"):
            raise ValueError("Select vLLM GPUs with CUDA_VISIBLE_DEVICES, using --device auto (or cuda).")
        if getattr(args, "quantization", "none") != "none":
            raise ValueError("This vLLM adapter uses original unquantized weights; use --backend transformers for 4/8-bit loading.")
        if getattr(args, "attn_implementation", None) is not None:
            raise ValueError("--attn-implementation is a Transformers-only option")


def effective_batch_size(args):
    return 1 if args.backend == "transformers" and args.do_sample else args.batch_size


def create_backend(args, revision):
    if args.backend == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(args, revision)
    return TransformersBackend(args, revision)


def record_key(run_id, model_id, seed, prompt):
    return digest([run_id, model_id, seed, prompt["item_id"], prompt["condition"]])


def record_seed(seed, model_id, prompt):
    return int(digest([seed, model_id, prompt["item_id"], prompt["condition"]])[:8], 16)


def resolve_revision(model_id, requested):
    if Path(model_id).is_dir():
        config = Path(model_id) / "config.json"
        if not config.exists():
            raise ValueError("Local model lacks config.json")
        return "local-config-" + file_hash(config)
    from huggingface_hub import HfApi
    return HfApi().model_info(model_id, revision=requested).sha


def termination_for_ids(ids, eos_ids, max_new_tokens):
    for index, token in enumerate(ids):
        if token in eos_ids:
            return ids[:index + 1], "eos"
    return ids, "length" if len(ids) >= max_new_tokens else "stopped_without_eos"


class TransformersBackend:
    def __init__(self, args, revision):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.args = args
        kwargs = {"revision": None if revision.startswith("local-config-") else revision,
                  "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", **kwargs)
        if not self.tokenizer.chat_template:
            raise ValueError("Checkpoint has no chat template")
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither PAD nor EOS")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
        load = dict(kwargs, torch_dtype=dtype)
        if args.device == "auto":
            load["device_map"] = "auto"
        elif args.quantization != "none":
            load["device_map"] = {"": args.device}
        if args.quantization != "none":
            load["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=args.quantization == "4bit", load_in_8bit=args.quantization == "8bit",
                bnb_4bit_compute_dtype=torch.bfloat16 if args.dtype in ("auto", "bfloat16") else dtype)
        if args.attn_implementation:
            load["attn_implementation"] = args.attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(args.model, **load)
        if args.device != "auto" and args.quantization == "none":
            self.model.to(args.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.input_device = self.model.get_input_embeddings().weight.device
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        if not self.eos_ids:
            raise ValueError("No EOS IDs configured")
        self.thinking = args.thinking == "on" or (args.thinking == "auto" and "qwen3" in args.model.lower())
        model_context = getattr(self.model.config, "max_position_embeddings", None)
        tokenizer_context = self.tokenizer.model_max_length
        candidates = [x for x in (model_context, tokenizer_context, args.context_limit)
                      if isinstance(x, int) and 0 < x < 10**8]
        self.context_limit = min(candidates) if candidates else None
        # Explicit GenerationConfig avoids inheriting hidden sampling defaults from the checkpoint.
        from transformers import GenerationConfig
        config = {"max_new_tokens": args.max_new_tokens, "do_sample": args.do_sample,
                  "num_beams": 1, "num_return_sequences": 1, "use_cache": True,
                  "eos_token_id": self.eos_ids, "pad_token_id": self.tokenizer.pad_token_id,
                  "bos_token_id": self.tokenizer.bos_token_id, "repetition_penalty": 1.0}
        if args.do_sample:
            config.update(temperature=args.temperature, top_p=args.top_p, top_k=0)
        self.generation_config = GenerationConfig(**config)
        self.metadata = {"engine": "transformers", "resolved_dtype": str(self.model.dtype),
                         "input_device": str(self.input_device),
                         "chat_template_sha256": digest(self.tokenizer.chat_template),
                         "eos_token_ids": self.eos_ids, "thinking": self.thinking,
                         "context_limit": self.context_limit,
                         "generation_config": self.generation_config.to_dict(),
                         "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}

    def render(self, messages):
        options = {}
        if self.args.thinking != "auto" or "qwen3" in self.args.model.lower():
            options["enable_thinking"] = self.thinking
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, **options)

    def prepare(self, prompt):
        text = self.render(prompt["messages"])
        count = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        error = None
        if self.context_limit and count + self.args.max_new_tokens > self.context_limit:
            error = "context_overflow"
        return text, count, error

    def generate(self, prepared, seeds):
        from transformers import set_seed
        if len(prepared) != len(seeds) or (self.args.do_sample and len(seeds) != 1):
            raise ValueError("Transformers sampling requires one request per call")
        set_seed(seeds[0])
        torch = self.torch
        texts = [x[0] for x in prepared]
        inputs = self.tokenizer(texts, padding=True, truncation=False, add_special_tokens=False,
                                return_tensors="pt").to(self.input_device)
        width = inputs["input_ids"].shape[1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        results = []
        for output in outputs[:, width:].tolist():
            kept, termination = termination_for_ids(output, self.eos_ids, self.args.max_new_tokens)
            raw = self.tokenizer.decode(kept, skip_special_tokens=True)
            results.append({"raw_output": raw,
                            "raw_output_with_special_tokens": self.tokenizer.decode(kept, skip_special_tokens=False),
                            "generated_token_ids": kept, "generated_tokens": len(kept),
                            "termination_status": termination, "batch_runtime_seconds": elapsed,
                            "amortized_runtime_seconds": elapsed / len(prepared),
                            "batch_size": len(prepared), "padded_input_width": width})
        return results


def run_locked(args, prompts):
    normalize_backend_args(args)
    output = args.output
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    old_meta = read_json(meta_path) if meta_path.exists() else None
    if output.exists() and output.stat().st_size and old_meta is None:
        raise ValueError("Nonempty results lack their metadata sidecar")
    revision = old_meta["model_revision"] if old_meta else resolve_revision(args.model, args.revision)
    signature = {"run_id": args.run_id, "model_id": args.model, "requested_revision": args.revision,
                 "backend_name": args.backend,
                 "model_revision": revision, "prompts_sha256": file_hash(args.prompts),
                 "seed": args.seed, "max_new_tokens": args.max_new_tokens,
                 "do_sample": args.do_sample, "temperature": args.temperature if args.do_sample else None,
                 "top_p": args.top_p if args.do_sample else None, "dtype": args.dtype,
                 "quantization": args.quantization, "device": args.device,
                 "thinking": args.thinking, "context_limit": args.context_limit,
                 "attn_implementation": args.attn_implementation,
                 "effective_batch_size": effective_batch_size(args),
                 "vllm_options": {key: getattr(args, key) for key in (
                     "tensor_parallel_size", "gpu_memory_utilization", "max_num_seqs", "max_model_len",
                     "enable_prefix_caching", "enforce_eager", "batch_invariant")} if args.backend == "vllm" else None,
                 "max_items": args.max_items, "versions": versions(), "schema_version": 2}
    if old_meta and old_meta["signature"] != signature:
        raise ValueError("Run configuration changed. Use a new --run-id AND output file; originals are immutable.")
    repair_tail(output)
    old_rows = read_jsonl(output) if output.exists() else []
    unique_index(old_rows, lambda r: (r["record_id"], r["attempt"]))
    latest = {}
    for row in old_rows:
        if row["run_signature"] != digest(signature):
            raise ValueError("Ledger contains a different run configuration")
        if row["record_id"] not in latest or latest[row["record_id"]]["attempt"] < row["attempt"]:
            latest[row["record_id"]] = row
    pending = []
    for prompt in prompts:
        key = record_key(args.run_id, args.model, args.seed, prompt)
        prior = latest.get(key)
        if prior and not (args.retry_errors and prior.get("error")):
            continue
        pending.append((prompt, key, 0 if prior is None else prior["attempt"] + 1))
    if not pending:
        print("No pending records; existing results preserved.")
        return
    backend = create_backend(args, revision)
    if old_meta and old_meta["backend"] != backend.metadata:
        raise ValueError("Loaded model/template/hardware differs from frozen run metadata")
    metadata = old_meta or {"signature": signature, "model_revision": revision,
                            "backend": backend.metadata, "created_utc": utc_now(),
                            "primary_attempt_policy": "first; errors count as invalid",
                            "retry_policy": "manual --retry-errors, identical settings, retained attempts"}
    write_json(meta_path, metadata, immutable=True)
    batch_size = signature["effective_batch_size"]
    print(f"{len(pending)} pending outputs; backend={args.backend}; batch_size={batch_size}; model revision={revision}", flush=True)
    overall = time.perf_counter()
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        records, eligible, prepared = [], [], []
        for prompt, key, attempt in batch:
            try:
                text, count, error = backend.prepare(prompt)
            except Exception as exc:
                # Persist tokenization/template failures too; never drop a requested cell.
                text, count = None, None
                error = f"prompt_preparation_error: {type(exc).__name__}: {exc}"
            row = dict(prompt, record_id=key, run_id=args.run_id, model_id=args.model, inference_backend=args.backend,
                       model_revision=revision, seed=args.seed, attempt=attempt,
                       rng_seed=record_seed(args.seed, args.model, prompt) if args.do_sample else args.seed,
                       run_signature=digest(signature), package_versions=signature["versions"],
                       precision=backend.metadata["resolved_dtype"], quantization=args.quantization,
                       generation_config=backend.metadata["generation_config"], thinking=backend.thinking,
                       exact_prompt=text, prompt_tokens=count, timestamp_utc=utc_now(), error=error,
                       raw_output="", raw_output_with_special_tokens="", generated_tokens=0,
                       generated_token_ids=[], termination_status=("context_overflow" if error == "context_overflow"
                           else "error" if error else "pending"),
                       batch_runtime_seconds=0.0, amortized_runtime_seconds=0.0)
            records.append(row)
            if error is None:
                eligible.append(row)
                prepared.append((text, count))
        if eligible:
            fatal_error = None
            try:
                generated = backend.generate(prepared, [row["rng_seed"] for row in eligible])
                if len(generated) != len(eligible):
                    raise RuntimeError("Backend returned wrong number of outputs")
                for row, result in zip(eligible, generated):
                    row.update(result)
            except Exception as exc:
                for row in eligible:
                    row.update(error=f"{type(exc).__name__}: {exc}", termination_status="error")
                gc.collect()
                if backend.torch.cuda.is_available():
                    backend.torch.cuda.empty_cache()
                # A failed engine may be unusable; don't mark every remaining
                # question as errored by repeatedly calling the same dead worker.
                if args.backend == "vllm":
                    fatal_error = exc
        else:
            fatal_error = None
        for row in records:
            row.update(parse_answer(row["raw_output"], labels(row["options"]),
                                    row["termination_status"], row["thinking"]))
        append_batch(output, records)
        done = min(start + batch_size, len(pending))
        print(f"{done}/{len(pending)} persisted; elapsed={time.perf_counter()-overall:.1f}s; "
              f"batch errors={sum(bool(r['error']) for r in records)}", flush=True)
        if fatal_error is not None:
            raise RuntimeError("vLLM batch failed; errors were saved. Fix the cause, then resume with "
                               "--retry-errors for identical settings, or use a new run for changed settings.") from fatal_error


def main():
    p = argparse.ArgumentParser(description="Resumable evaluation with vLLM (default) or Transformers.")
    add_backend_arguments(p)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--conditions", default="pilot")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--max-items", type=int, help="Smoke test only; frozen in the run signature")
    p.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    p.add_argument("--device", default="auto", help="auto, cuda:0, cpu, etc.")
    p.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="none")
    p.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"))
    p.add_argument("--thinking", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--context-limit", type=int, help="Optional tighter limit; prompts are never truncated")
    p.add_argument("--do-sample", action="store_true", help="Per-request RNG seeds; vLLM batches sampling, Transformers uses batch size 1")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--retry-errors", action="store_true", help="Append attempts only for runtime/context errors")
    p.add_argument("--dry-run", action="store_true", help="Validate tasks and count work without loading a model")
    a = p.parse_args()
    try:
        normalize_backend_args(a)
    except ValueError as exc:
        p.error(str(exc))
    if a.batch_size < 1 or a.max_new_tokens < 1 or a.temperature <= 0 or not 0 < a.top_p <= 1:
        p.error("Invalid generation settings")
    if a.max_items is not None and a.max_items < 1:
        p.error("--max-items must be positive")
    prompts = read_jsonl(a.prompts)
    if not prompts:
        p.error("Empty prompt file")
    unique_index(prompts, lambda r: (r["item_id"], r["condition"]))
    if len({(r["study_id"], r["evaluation_split"], r["protocol"]) for r in prompts}) != 1:
        p.error("Prompt file mixes studies, splits, or protocols")
    selected = conditions(a.conditions)
    all_ids = sorted({r["item_id"] for r in prompts})
    if a.max_items:
        all_ids = all_ids[:a.max_items]
    ids = set(all_ids)
    prompts = [r for r in prompts if r["condition"] in selected and r["item_id"] in ids]
    if len(prompts) != len(ids) * len(selected):
        p.error("Requested condition bank is incomplete")
    prompts.sort(key=lambda r: (r["item_id"], selected.index(r["condition"])))
    if a.dry_run:
        print(json.dumps({"items": len(ids), "conditions": selected, "outputs": len(prompts),
                          "maximum_output_tokens": len(prompts) * a.max_new_tokens,
                          "backend": a.backend, "effective_batch_size": effective_batch_size(a)}, indent=2))
        return
    from filelock import FileLock
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(a.output) + ".lock", timeout=0):
        run_locked(a, prompts)
