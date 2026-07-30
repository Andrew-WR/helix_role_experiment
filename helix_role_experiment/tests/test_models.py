import json
import tempfile
import unittest
from pathlib import Path

from helix_role_experiment.benchmarks import load_math500_integer_problems
from helix_role_experiment.config import config_hash
from helix_role_experiment.models import (
    HuggingFaceTraceCollector,
    resolve_adapter_path,
)


class ModelConfigurationTests(unittest.TestCase):
    def test_math500_loader_selects_exactly_scoreable_levels(self):
        rows = [
            {
                "problem": "Compute 40 + 2.",
                "answer": "42",
                "level": 2,
                "subject": "Prealgebra",
                "unique_id": "accepted",
            },
            {
                "problem": "Give a symbolic result.",
                "answer": r"\frac{1}{2}",
                "level": 2,
                "subject": "Algebra",
                "unique_id": "symbolic",
            },
            {
                "problem": "Too difficult.",
                "answer": "7",
                "level": 5,
                "subject": "Number Theory",
                "unique_id": "wrong-level",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "math500.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            selected = load_math500_integer_problems(
                1,
                {2, 3},
                17,
                path,
            )
        self.assertEqual(selected[0].problem_id, "math500-accepted")
        self.assertEqual(selected[0].answer, "42")
        self.assertEqual(selected[0].metadata["benchmark"], "math500")

    def test_chat_template_receives_explicit_thinking_mode(self):
        calls = []

        class Tokenizer:
            chat_template = "template"

            @staticmethod
            def apply_chat_template(messages, **kwargs):
                calls.append((messages, kwargs))
                return "formatted"

        collector = object.__new__(HuggingFaceTraceCollector)
        collector.tokenizer = Tokenizer()
        collector.chat_template_kwargs = {"enable_thinking": True}
        self.assertEqual(collector.format_prompt("question"), "formatted")
        self.assertTrue(calls[0][1]["enable_thinking"])
        self.assertTrue(calls[0][1]["add_generation_prompt"])

    def test_adapter_path_uses_first_directory_with_peft_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing"
            fallback = root / "fallback"
            fallback.mkdir()
            (fallback / "adapter_config.json").write_text(
                "{}",
                encoding="utf-8",
            )
            resolved = resolve_adapter_path(
                {
                    "adapter_path": str(missing),
                    "adapter_fallback_paths": [str(fallback)],
                }
            )
            self.assertEqual(resolved, str(fallback))

    def test_missing_local_adapter_lists_all_checked_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            with self.assertRaisesRegex(
                FileNotFoundError,
                "No configured PEFT adapter",
            ) as raised:
                resolve_adapter_path(
                    {
                        "adapter_path": str(first),
                        "adapter_fallback_paths": [str(second)],
                    }
                )
            self.assertIn(str(first), str(raised.exception))
            self.assertIn(str(second), str(raised.exception))

    def test_remote_adapter_id_is_preserved(self):
        self.assertEqual(
            resolve_adapter_path({"adapter_path": "owner/adapter"}),
            "owner/adapter",
        )

    def test_fallback_mounts_do_not_change_scientific_config_hash(self):
        base = {
            "model": {"adapter_path": "/kaggle/input/primary"},
            "study": {"seed": 1},
        }
        with_fallback = {
            "model": {
                "adapter_path": "/kaggle/input/primary",
                "adapter_fallback_paths": ["/kaggle/input/fallback"],
            },
            "study": {"seed": 1},
        }
        self.assertEqual(config_hash(base), config_hash(with_fallback))

    def test_explicit_default_thinking_preserves_existing_config_hash(self):
        base = {"model": {"id": "Qwen/Qwen3.6-27B"}}
        explicit = {
            "model": {
                "id": "Qwen/Qwen3.6-27B",
                "chat_template_kwargs": {"enable_thinking": True},
            }
        }
        disabled = {
            "model": {
                "id": "Qwen/Qwen3.6-27B",
                "chat_template_kwargs": {"enable_thinking": False},
            }
        }
        other_base = {"model": {"id": "other/model"}}
        other_explicit = {
            "model": {
                "id": "other/model",
                "chat_template_kwargs": {"enable_thinking": True},
            }
        }
        self.assertEqual(config_hash(base), config_hash(explicit))
        self.assertNotEqual(config_hash(base), config_hash(disabled))
        self.assertNotEqual(
            config_hash(other_base),
            config_hash(other_explicit),
        )


if __name__ == "__main__":
    unittest.main()
