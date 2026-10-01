"""Exercise the real vLLM adapter against controlled external API doubles.

These check our integration logic on CPU; they are not GPU inference benchmarks.
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from cot_hints.common import read_json, read_jsonl, write_jsonl
from cot_hints.inference import (add_backend_arguments, effective_batch_size,
                                 normalize_backend_args, record_seed, run_locked)
from cot_hints.protocol import EXTENSION, parse_answer
from cot_hints.scoring import collect_records, paired_rows
from cot_hints.vllm_backend import VLLMBackend, completion_termination, configure_environment
from test_pipeline import FakeBackend, prompts


def args_for(root, **overrides):
    args = argparse.Namespace(output=root / "vllm.jsonl", model="Qwen/Qwen2.5-3B-Instruct", revision="main",
        run_id="vllm-test", prompts=root / "prompts.jsonl", seed=42, max_new_tokens=128, do_sample=False,
        temperature=0.7, top_p=0.9, dtype="auto", quantization="none", device="auto", thinking="auto",
        context_limit=None, attn_implementation=None, batch_size=8, max_items=None, retry_errors=False,
        backend="vllm", tensor_parallel_size=1, gpu_memory_utilization=0.9, max_num_seqs=64,
        max_model_len=None, enable_prefix_caching=True, enforce_eager=False, batch_invariant=False)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TokenizerDouble:
    chat_template = "chat-template-fixture"
    eos_token_id = 999
    model_max_length = 32768

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False and add_generation_prompt is True
        return "".join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + "<assistant>\n"

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False, "Do not add a second BOS after rendering chat"
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens):
        return "".join(chr(i) if i < 999 else "" if skip_special_tokens else "<eos>" for i in ids)


class SamplingParamsDouble:
    # The fields used here are checked against the pinned SamplingParams source.
    allowed = {"n", "temperature", "top_p", "top_k", "min_p", "max_tokens", "min_tokens",
               "presence_penalty", "frequency_penalty", "repetition_penalty", "stop", "stop_token_ids",
               "ignore_eos", "detokenize", "skip_special_tokens", "spaces_between_special_tokens",
               "truncate_prompt_tokens", "seed"}

    def __init__(self, **kwargs):
        assert not set(kwargs)-self.allowed
        self.__dict__.update(kwargs)


class LLMDouble:
    instances = []
    finish = "stop"
    stop = 1000
    reverse = False
    fail = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.llm_engine = types.SimpleNamespace(model_config=types.SimpleNamespace(
            dtype="torch." + kwargs["dtype"], max_model_len=kwargs.get("max_model_len", 32768)))
        self.instances.append(self)

    def generate(self, inputs, sampling_params, use_tqdm):
        if self.fail:
            raise RuntimeError("Simulated worker failure")
        assert len(inputs) == len(sampling_params)
        assert use_tqdm is False
        self.calls.append((copy.deepcopy(inputs), copy.deepcopy(sampling_params)))
        results = []
        for i, request in enumerate(inputs):
            text = "<think>The hint might be misleading.</think>My complete explanation.\nFINAL: J"
            token_ids = [ord(c) for c in text] + ([1000] if self.finish == "stop" else [])
            results.append(types.SimpleNamespace(prompt_token_ids=request["prompt_token_ids"],
                request_id=str(i), finished=True, outputs=[types.SimpleNamespace(
                    token_ids=token_ids, text=text, finish_reason=self.finish, stop_reason=self.stop)]))
        return list(reversed(results)) if self.reverse else results


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = args_for(self.root)
        self.rows = prompts(3)
        write_jsonl(self.args.prompts, self.rows)
        LLMDouble.instances.clear()
        LLMDouble.finish, LLMDouble.stop, LLMDouble.reverse, LLMDouble.fail = "stop", 1000, False, False
        config = types.SimpleNamespace(dtype="torch.bfloat16", eos_token_id=999)
        generation = types.SimpleNamespace(eos_token_id=[999, 1000])
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(device_count=lambda: 1, get_device_name=lambda i: "H100-test-double",
            is_available=lambda: False, empty_cache=lambda: None, get_device_capability=lambda i: (9, 0))
        transformers = types.ModuleType("transformers")
        transformers.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *a, **kw: config)
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **kw: TokenizerDouble())
        transformers.GenerationConfig = types.SimpleNamespace(from_pretrained=lambda *a, **kw: generation,
                                                               from_model_config=lambda c: generation)
        vllm = types.ModuleType("vllm")
        vllm.LLM, vllm.SamplingParams = LLMDouble, SamplingParamsDouble
        self.modules = patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "vllm": vllm})
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.revision = patch("cot_hints.inference.resolve_revision", return_value="pinned-test-revision")
        self.modules.start()
        self.environment.start()
        self.revision.start()

    def tearDown(self):
        self.revision.stop()
        self.environment.stop()
        self.modules.stop()
        self.temp.cleanup()

    def test_exact_tokens_and_complete_trace(self):
        backend = VLLMBackend(self.args, "revision")
        prepared = [backend.prepare(r) for r in self.rows[:2]]
        outputs = backend.generate([(text, count) for text, count, error in prepared], [101, 202])
        inputs, params = backend.llm.calls[0]
        self.assertEqual(inputs[0]["prompt_token_ids"], [ord(c) for c in prepared[0][0]])
        self.assertEqual([p.seed for p in params], [101, 202])
        self.assertEqual(params[0].stop_token_ids, [999, 1000])
        self.assertEqual(params[0].stop, [])
        self.assertEqual(params[0].temperature, 0)
        self.assertEqual(params[0].top_p, 1)
        self.assertEqual(params[0].top_k, -1)
        self.assertEqual(backend.llm.kwargs["generation_config"], "vllm")
        self.assertEqual(backend.llm.kwargs["dtype"], "bfloat16")
        self.assertEqual(backend.llm.kwargs["tokenizer_revision"], "revision")
        parsed = parse_answer(outputs[0]["raw_output"], "ABCDEFGHIJ", outputs[0]["termination_status"])
        self.assertEqual(parsed["parsed_answer"], "J")
        self.assertEqual(parsed["thinking_trace"], "The hint might be misleading.")
        self.assertIn("My complete explanation.", parsed["final_response"])
        self.assertIn("<eos>", outputs[0]["raw_output_with_special_tokens"])

    def test_sampling_stays_batched_with_per_request_seeds(self):
        self.args.do_sample = True
        run_locked(self.args, self.rows[:8])
        self.assertEqual(len(LLMDouble.instances[0].calls), 1)
        records = read_jsonl(self.args.output)
        params = LLMDouble.instances[0].calls[0][1]
        self.assertEqual(len(params), 8)
        self.assertEqual([p.seed for p in params], [record_seed(42, self.args.model, r) for r in self.rows[:8]])
        self.assertTrue(all(p.temperature == 0.7 and p.top_p == 0.9 for p in params))
        self.assertTrue(all(r["inference_backend"] == "vllm" for r in records))
        self.assertTrue(all(r["batch_size"] == 8 for r in records))

    def test_resume_and_extension_preserve_baselines(self):
        pilot = [r for r in self.rows if r["condition"] in ("no_hint", "neutral_hint", "direct_wrong_hint")]
        run_locked(self.args, pilot)
        initial = self.args.output.read_bytes()
        run_locked(self.args, pilot)
        self.assertEqual(self.args.output.read_bytes(), initial)
        run_locked(self.args, self.rows)
        self.assertTrue(self.args.output.read_bytes().startswith(initial))
        self.assertEqual(len(read_jsonl(self.args.output)), 21)
        rows = collect_records([self.args.output], self.args.prompts, EXTENSION)
        self.assertEqual(sum(r["pair_valid"] for r in paired_rows(rows)), 18)

    def test_three_sampling_seeds_form_one_statistical_group(self):
        self.args.do_sample = True
        paths = []
        for seed in (42, 200, 300):
            self.args.seed = seed
            self.args.output = self.root / f"seed-{seed}.jsonl"
            self.args.run_id = f"seed-{seed}"
            run_locked(self.args, self.rows[:7])
            paths.append(self.args.output)
        records = collect_records(paths, self.args.prompts, EXTENSION)
        self.assertEqual(len({r["group_id"] for r in records}), 1)
        self.assertEqual({r["seed"] for r in records}, {42, 200, 300})
        self.assertEqual({llm.kwargs["seed"] for llm in LLMDouble.instances}, {0})

    def test_backend_switch_and_settings_change_rejected(self):
        run_locked(self.args, self.rows[:1])
        self.args.backend = "transformers"
        with self.assertRaisesRegex(ValueError, "Run configuration changed"):
            run_locked(self.args, self.rows[:1])
        self.args.backend = "vllm"
        self.args.gpu_memory_utilization = 0.8
        with self.assertRaisesRegex(ValueError, "Run configuration changed"):
            run_locked(self.args, self.rows[:1])

    def test_two_backends_remain_separate_in_scoring(self):
        run_locked(self.args, self.rows[:7])
        vllm_path = self.args.output
        self.args.backend = "transformers"
        self.args.output = self.root / "transformers.jsonl"
        self.args.run_id = "transformers-run"
        with patch("cot_hints.inference.TransformersBackend", FakeBackend):
            run_locked(self.args, self.rows[:7])
        records = collect_records([vllm_path, self.args.output], self.args.prompts, EXTENSION)
        self.assertEqual(len({r["group_id"] for r in records}), 2)

    def test_runtime_failure_persisted_without_poisoning_remaining_items(self):
        LLMDouble.fail = True
        with self.assertRaisesRegex(RuntimeError, "errors were saved"):
            run_locked(self.args, self.rows)
        records = read_jsonl(self.args.output)
        self.assertEqual(len(records), 8)
        self.assertTrue(all(r["termination_status"] == "error" for r in records))
        LLMDouble.fail = False
        self.args.retry_errors = True
        run_locked(self.args, self.rows)
        records = collect_records([self.args.output], self.args.prompts, EXTENSION)
        latest = collect_records([self.args.output], self.args.prompts, EXTENSION, "latest")
        self.assertEqual(sum(r["valid"] for r in records), 13)
        self.assertTrue(all(r["valid"] for r in latest))

    def test_context_overflow_is_recorded_not_truncated(self):
        self.args.max_model_len = 200
        run_locked(self.args, self.rows[:2])
        records = read_jsonl(self.args.output)
        self.assertEqual(LLMDouble.instances[0].calls, [])
        self.assertTrue(all(r["termination_status"] == "context_overflow" for r in records))

    def test_length_cap_invalidates_even_a_complete_final_line(self):
        LLMDouble.finish, LLMDouble.stop = "length", None
        run_locked(self.args, self.rows[:1])
        row = read_jsonl(self.args.output)[0]
        self.assertEqual(row["termination_status"], "length")
        self.assertEqual(row["candidate_answer"], "J")
        self.assertFalse(row["valid"])

    def test_misordered_outputs_fail_closed(self):
        LLMDouble.reverse = True
        with self.assertRaisesRegex(RuntimeError, "errors were saved"):
            run_locked(self.args, self.rows[:2])
        self.assertTrue(all(r["termination_status"] == "error" for r in read_jsonl(self.args.output)))

    def test_batch_invariant_hardware_checked(self):
        self.args.batch_invariant = True
        sys.modules["torch"].cuda.get_device_capability = lambda i: (8, 0)
        with self.assertRaisesRegex(ValueError, "compute capability"):
            VLLMBackend(self.args, "revision")


class ConfigurationTests(unittest.TestCase):
    def test_default_and_backend_batch_sizes(self):
        parser = argparse.ArgumentParser()
        add_backend_arguments(parser)
        args = parser.parse_args([])
        args.device, args.do_sample = "auto", True
        normalize_backend_args(args)
        self.assertEqual(args.backend, "vllm")
        self.assertEqual(effective_batch_size(args), 128)
        args = parser.parse_args(["--backend", "transformers"])
        args.device, args.do_sample = "cpu", True
        normalize_backend_args(args)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(effective_batch_size(args), 1)

    def test_unsupported_controls_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for kwargs in ({"device": "cuda:1"}, {"quantization": "4bit"}, {"gpu_memory_utilization": 1.5},
                           {"batch_size": 0}, {"tensor_parallel_size": 0}, {"attn_implementation": "eager"}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    normalize_backend_args(args_for(Path(temp), **kwargs))

    def test_environment_conflicts_are_explicit(self):
        with patch.dict(os.environ, {"VLLM_ENABLE_V1_MULTIPROCESSING": "1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "conflicts"):
                configure_environment()

    def test_unknown_stop_is_invalid(self):
        row = types.SimpleNamespace(finish_reason="stop", stop_reason="FINAL:", token_ids=[7, 8])
        self.assertEqual(completion_termination(row, [999]), "stopped_without_eos")
        row.finish_reason = "abort"
        self.assertEqual(completion_termination(row, [999]), "error")

    def test_pipeline_forwards_engine_arguments(self):
        project = Path(__file__).resolve().parents[1]
        command = [sys.executable, str(project / "run_pipeline.py"), "sampling", "--dry-run",
                   "--models", "Qwen/Qwen2.5-3B-Instruct", "--tensor-parallel-size", "2",
                   "--batch-size", "96", "--max-num-seqs", "32", "--gpu-memory-utilization", "0.8",
                   "--max-model-len", "4096", "--no-enable-prefix-caching", "--enforce-eager"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.count("--backend vllm"), 3)
        for fragment in ("--batch-size 96", "--max-num-seqs 32", "--max-model-len 4096", "--tensor-parallel-size 2",
                         "--no-enable-prefix-caching", "--enforce-eager", "runs/vllm/sampling"):
            self.assertIn(fragment, result.stdout)


if __name__ == "__main__":
    unittest.main()
