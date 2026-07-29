import tempfile
import unittest
from pathlib import Path

from helix_role_experiment.config import config_hash
from helix_role_experiment.models import resolve_adapter_path


class ModelConfigurationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
