from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from cot_hints.common import read_jsonl
from cot_hints.inference import TransformersBackend, run_locked, thinking_settings
from cot_hints.protocol import EXTENSION, parse_answer
from cot_hints.scoring import collect_records, paired_rows
from cot_hints.vllm_backend import VLLMBackend, eos_ids_from_config
import test_vllm_backend as fixtures


class ThinkingTokenizer(fixtures.TokenizerDouble):
    chat_template = "fixture with native enable_thinking switch"
    pad_token_id = 999
    bos_token_id = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        prefix = super().apply_chat_template(messages, tokenize, add_generation_prompt)
        return prefix + ("<think>\n" if kwargs.get("enable_thinking", True)
                         else "<think>\n\n</think>\n\n")


class Qwen35BackendTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AdapterTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root, self.args, self.rows = self.fixture.root, self.fixture.args, self.fixture.rows
        self.args.model = "Qwen/Qwen3.5-4B"
        # The dtype/EOS/context are nested, as in the published Qwen3.5 config.
        self.config = types.SimpleNamespace(model_type="qwen3_5", text_config=types.SimpleNamespace(
            dtype="torch.bfloat16", eos_token_id=1000, max_position_embeddings=40960))
        transformer = sys.modules["transformers"]
        transformer.AutoConfig.from_pretrained = lambda *a, **kw: self.config
        transformer.AutoTokenizer.from_pretrained = lambda *a, **kw: ThinkingTokenizer()

    def test_native_switch_and_nested_dtype(self):
        for mode, suffix, enabled in (("on", "<think>\n", True),
                                      ("off", "<think>\n\n</think>\n\n", False),
                                      ("auto", "<think>\n", True)):
            with self.subTest(mode=mode):
                self.args.thinking = mode
                backend = VLLMBackend(self.args, "revision")
                self.assertEqual(backend.thinking, enabled)
                self.assertTrue(backend.prepare(self.rows[0])[0].endswith(suffix))
                self.assertEqual(backend.llm.kwargs["dtype"], "bfloat16")
                self.assertTrue(backend.llm.kwargs["language_model_only"])
        self.assertEqual(eos_ids_from_config(types.SimpleNamespace(eos_token_id=None),
                                            self.config, ThinkingTokenizer()), [1000])

    def test_trace_and_final_remain_separate_and_modes_do_not_pool(self):
        def generate(llm, inputs, sampling_params, use_tqdm):
            output = []
            for i, request in enumerate(inputs):
                prompt = "".join(chr(t) for t in request["prompt_token_ids"])
                text = ("Consider another option.\nFINAL: A\n</think>\nExplanation.\nFINAL: J"
                        if prompt.endswith("<think>\n") else "Explanation.\nFINAL: J")
                output.append(types.SimpleNamespace(prompt_token_ids=request["prompt_token_ids"],
                    request_id=str(i), finished=True, outputs=[types.SimpleNamespace(
                        token_ids=[ord(c) for c in text]+[1000], text=text,
                        finish_reason="stop", stop_reason=1000)]))
            return output

        paths = []
        with patch.object(fixtures.LLMDouble, "generate", generate):
            for mode in ("on", "off"):
                self.args.thinking = mode
                self.args.run_id = "mode-"+mode
                self.args.output = self.root / f"{mode}.jsonl"
                run_locked(self.args, self.rows[:7])
                paths.append(self.args.output)
                saved = read_jsonl(self.args.output)
                self.assertTrue(all(r["valid"] and r["parsed_answer"] == "J" for r in saved))
                self.assertTrue(all(bool(r["thinking_trace"]) == (mode == "on") for r in saved))
                self.assertTrue(all("FINAL: A" not in r["final_response"] for r in saved))
        records = collect_records(paths, self.args.prompts, EXTENSION)
        self.assertEqual(len({r["group_id"] for r in records}), 2)
        self.assertEqual({r["thinking"] for r in records}, {True, False})
        lookup = {r["record_id"]: r for r in records}
        for pair in paired_rows(records):
            self.assertEqual(pair["thinking"], lookup[pair["baseline_record_id"]]["thinking"])

    def test_transformers_uses_conditional_generation_and_nested_context(self):
        self.args.backend, self.args.device, self.args.thinking = "transformers", "cpu", "on"
        model = types.SimpleNamespace(config=self.config, dtype="torch.bfloat16",
            generation_config=types.SimpleNamespace(eos_token_id=[1000]),
            get_input_embeddings=lambda: types.SimpleNamespace(weight=types.SimpleNamespace(device="cpu")),
            eval=lambda: None, requires_grad_=lambda v: None, to=lambda d: None)
        transformer = sys.modules["transformers"]
        causal = types.SimpleNamespace(from_pretrained=lambda *a, **kw: self.fail("Wrong model loader"))
        image_text = types.SimpleNamespace(from_pretrained=lambda *a, **kw: model)

        class GenerationConfig:
            def __init__(self, **kw):
                self.values = kw

            def to_dict(self):
                return self.values

        with patch.object(transformer, "AutoModelForCausalLM", causal, create=True), \
             patch.object(transformer, "AutoModelForImageTextToText", image_text, create=True), \
             patch.object(transformer, "BitsAndBytesConfig", object, create=True), \
             patch.object(transformer, "GenerationConfig", GenerationConfig):
            backend = TransformersBackend(self.args, "revision")
        self.assertEqual(backend.context_limit, 32768)  # tighter tokenizer limit wins
        self.assertTrue(backend.thinking)
        self.assertTrue(backend.render(self.rows[0]["messages"]).endswith("<think>\n"))


class ModePolicyTests(unittest.TestCase):
    def test_unsupported_thinking_is_not_silently_enabled(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            thinking_settings(fixtures.TokenizerDouble(), "on")
        self.assertEqual(thinking_settings(fixtures.TokenizerDouble(), "off"), (False, {}))
        self.assertEqual(thinking_settings(fixtures.TokenizerDouble(), "auto"), (False, {}))

    def test_thinking_cap_does_not_become_a_valid_final_answer(self):
        parsed = parse_answer("Try this.\nFINAL: J", "ABCDEFGHIJ", "length", thinking=True)
        self.assertFalse(parsed["valid"])
        self.assertEqual(parsed["final_response"], "")

    def commands(self, stage, *extra):
        project = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(project / "run_pipeline.py"), stage,
            "--models", "Qwen/Qwen3.5-4B", "--thinking", "both", "--root", "fixture-root",
            "--dry-run", *extra], capture_output=True, text=True, check=True)
        import shlex
        return [shlex.split(line) for line in result.stdout.splitlines()]

    def test_pipeline_both_uses_distinct_outputs_and_matching_summary_inputs(self):
        generated = self.commands("extension")
        self.assertEqual(len(generated), 2)
        paths = {cmd[cmd.index("--thinking")+1]: cmd[cmd.index("--output")+1] for cmd in generated}
        self.assertNotEqual(paths["on"], paths["off"])
        summaries = self.commands("summarize", "--experiment", "extension")
        self.assertEqual(len(summaries), 4)
        for mode, score, analysis in zip(("on", "off"), summaries[::2], summaries[1::2]):
            self.assertEqual(score[score.index("--results")+1], paths[mode])
            self.assertTrue(score[-1].endswith(f"extension-thinking-{mode}"))
            self.assertEqual(analysis[analysis.index("--scores-dir")+1], score[-1])


if __name__ == "__main__":
    unittest.main()
