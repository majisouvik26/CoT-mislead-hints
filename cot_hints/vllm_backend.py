from __future__ import annotations

import os
import time

from .common import digest
from .inference import text_config, thinking_settings


def configure_environment(batch_invariant=False):
    """Explicit offline scheduling policy, applied before importing vLLM.

Scheduling determinism is not a promise of bitwise equality when the batch
composition changes on resume. Optional batch invariance needs supported GPUs.
"""
    environment = {
        "VLLM_USE_V1": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_BATCH_INVARIANT": "1" if batch_invariant else "0",
    }
    for key, value in environment.items():
        previous = os.environ.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"{key}={previous!r} conflicts with the requested run policy {value!r}. "
                             "Unset it or select matching CLI options before starting a new process.")
        os.environ[key] = value
    return environment


def eos_ids_from_config(generation_config, model_config, tokenizer):
    eos = getattr(generation_config, "eos_token_id", None)
    if eos is None:
        eos = getattr(text_config(model_config), "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    values = [eos] if isinstance(eos, int) else list(eos or [])
    if not values or any(not isinstance(value, int) for value in values):
        raise ValueError("Checkpoint has no usable EOS token IDs")
    return sorted(set(values))


def sampling_config(args, eos_ids):
    # Explicitly override defaults from both vLLM and the model repository.
    return {
        "n": 1,
        "temperature": args.temperature if args.do_sample else 0.0,
        "top_p": args.top_p if args.do_sample else 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "max_tokens": args.max_new_tokens,
        "min_tokens": 0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "stop": [],
        "stop_token_ids": list(eos_ids),
        # EOS is handled explicitly above, including all generation_config IDs.
        "ignore_eos": True,
        "detokenize": True,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": False,
        "truncate_prompt_tokens": None,
    }


def completion_termination(completion, eos_ids):
    reason = completion.finish_reason
    stop = completion.stop_reason
    tokens = list(completion.token_ids)
    if reason == "length":
        return "length"
    if reason == "stop":
        # Only explicit checkpoint EOS IDs were requested as stops.
        if stop in eos_ids or (tokens and tokens[-1] in eos_ids):
            return "eos"
        return "stopped_without_eos"
    return "error" if reason == "abort" else "stopped_without_eos"


class VLLMBackend:
    def __init__(self, args, revision):
        environment = configure_environment(args.batch_invariant)
        try:
            import torch
            from transformers import AutoConfig, AutoTokenizer, GenerationConfig
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("vLLM dependencies are unavailable. Install with "
                               "python -m pip install -r requirements.txt, or use "
                               "requirements-qwen35.txt in a fresh environment for Qwen3.5.") from exc
        self.torch = torch
        self.args = args
        self.SamplingParams = SamplingParams
        if args.batch_invariant:
            n = min(args.tensor_parallel_size, torch.cuda.device_count())
            if n < args.tensor_parallel_size or any(torch.cuda.get_device_capability(i)[0] < 9 for i in range(n)):
                raise ValueError("The pinned vLLM batch-invariant mode requires NVIDIA compute capability >=9.0 "
                                 "(e.g. H100/H200). Omit --batch-invariant on other GPUs.")
        kwargs = {"revision": None if revision.startswith("local-config-") else revision,
                  "trust_remote_code": False}
        config = AutoConfig.from_pretrained(args.model, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, **kwargs)
        if not self.tokenizer.chat_template:
            raise ValueError("Checkpoint has no chat template")
        self.thinking, self.chat_options = thinking_settings(self.tokenizer, args.thinking)
        try:
            generation = GenerationConfig.from_pretrained(args.model, **kwargs)
        except OSError:
            generation = GenerationConfig.from_model_config(config)
        self.eos_ids = eos_ids_from_config(generation, config, self.tokenizer)
        # vLLM auto converts FP32 checkpoint configs to FP16, unlike Transformers.
        # Resolve auto ourselves so moving engines does not silently change dtype.
        dtype = args.dtype
        if dtype == "auto":
            language = text_config(config)
            declared = (getattr(language, "dtype", None) or getattr(language, "torch_dtype", None)
                        or getattr(config, "dtype", None) or getattr(config, "torch_dtype", None))
            dtype = str(declared).removeprefix("torch.") if declared is not None else "float32"
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"Unsupported dtype {dtype!r}; supply --dtype explicitly")
        # Keep the engine seed fixed across sampled runs. Random draws use the
        # stable per-request seeds below; backend metadata can then group seeds.
        engine_args = {
            "model": args.model,
            "tokenizer": args.model,
            "revision": kwargs["revision"],
            "tokenizer_revision": kwargs["revision"],
            "trust_remote_code": False,
            "dtype": dtype,
            "seed": 0,
            "generation_config": "vllm",
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "enable_prefix_caching": args.enable_prefix_caching,
            "enforce_eager": args.enforce_eager,
            "disable_log_stats": True,
        }
        if args.max_model_len is not None:
            engine_args["max_model_len"] = args.max_model_len
        if getattr(config, "model_type", "") in ("qwen3_5", "qwen3_5_moe"):
            # This study supplies text only; do not load/profile the vision encoder.
            engine_args["language_model_only"] = True
        self.llm = LLM(**engine_args)
        engine_config = self.llm.llm_engine.model_config
        engine_limit = engine_config.max_model_len
        candidates = [value for value in (engine_limit, self.tokenizer.model_max_length, args.context_limit)
                      if isinstance(value, int) and 0 < value < 10**8]
        self.context_limit = min(candidates)
        self.sampling = sampling_config(args, self.eos_ids)
        # Validate the complete SamplingParams configuration before writing records.
        SamplingParams(**self.sampling, seed=0)
        self.metadata = {
            "engine": "vllm",
            "resolved_dtype": str(engine_config.dtype),
            "chat_template_sha256": digest(self.tokenizer.chat_template),
            "eos_token_ids": self.eos_ids,
            "thinking": self.thinking,
            "context_limit": self.context_limit,
            "generation_config": self.sampling,
            "engine_config": engine_args,
            "environment": environment,
            "request_seed_policy": "SHA256(master seed, model, item ID, condition); stored per record",
            "reproducibility_scope": (
                "Batch-invariant mode requested; same hardware/software required"
                if args.batch_invariant else
                "Offline deterministic scheduling; batch composition changes may affect numeric outputs"
            ),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }

    def render(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, **self.chat_options)

    def prepare(self, prompt):
        text = self.render(prompt["messages"])
        count = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        error = "context_overflow" if count + self.args.max_new_tokens > self.context_limit else None
        return text, count, error

    def generate(self, prepared, seeds):
        if len(prepared) != len(seeds):
            raise ValueError("Every vLLM request must have one explicit seed")
        # Token-ID inputs avoid reapplying a chat template or introducing a BOS.
        inputs = []
        for text, expected_count in prepared:
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) != expected_count:
                raise ValueError("Prompt tokenization changed between preparation and generation")
            inputs.append({"prompt_token_ids": ids})
        params = [self.SamplingParams(**self.sampling, seed=seed) for seed in seeds]
        start = time.perf_counter()
        outputs = self.llm.generate(inputs, sampling_params=params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        if len(outputs) != len(inputs):
            raise RuntimeError("vLLM returned an unexpected number of requests")
        results = []
        # LLM.generate guarantees input order; validate returned token IDs too.
        for output, expected in zip(outputs, inputs):
            if list(output.prompt_token_ids) != expected["prompt_token_ids"]:
                raise RuntimeError("vLLM output is paired with the wrong prompt")
            if not output.finished or len(output.outputs) != 1:
                raise RuntimeError("Expected exactly one finished completion per prompt")
            completion = output.outputs[0]
            tokens = list(completion.token_ids)
            termination = completion_termination(completion, self.eos_ids)
            result = {
                "raw_output": self.tokenizer.decode(tokens, skip_special_tokens=True),
                "raw_output_with_special_tokens": self.tokenizer.decode(tokens, skip_special_tokens=False),
                "engine_output_text": completion.text,
                "generated_token_ids": tokens,
                "generated_tokens": len(tokens),
                "termination_status": termination,
                "engine_finish_reason": completion.finish_reason,
                "engine_stop_reason": completion.stop_reason,
                "engine_request_id": output.request_id,
                "batch_runtime_seconds": elapsed,
                "amortized_runtime_seconds": elapsed / len(prepared),
                "batch_size": len(prepared),
                "padded_input_width": None,
            }
            if termination == "error":
                result["error"] = "vLLM aborted this request"
            results.append(result)
        return results
