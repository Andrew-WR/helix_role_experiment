import unittest
from types import SimpleNamespace

from helix_role_experiment.hooks import (
    adapter_layers_from_config,
    discover_decoder_layers,
    select_layer_indices,
)


class WrappedModel:
    def __init__(self, base):
        self.base_model = SimpleNamespace(model=base)


class HooksTests(unittest.TestCase):
    def test_decoder_layers_are_found_through_peft_style_wrapper(self):
        layers = [object() for _ in range(12)]
        qwen = SimpleNamespace(model=SimpleNamespace(layers=layers))
        wrapped = WrappedModel(qwen)
        self.assertEqual(discover_decoder_layers(wrapped), layers)

    def test_adapter_neighborhood_includes_sentinels_and_local_window(self):
        selected = select_layer_indices(
            "adapter_neighborhood",
            total_layers=64,
            adapter_target_layers=[37],
        )
        self.assertEqual(
            selected,
            [0, 16, 32, 35, 36, 37, 38, 39, 48, 63],
        )

    def test_explicit_layer_selection_is_validated(self):
        self.assertEqual(select_layer_indices([7, 0, 7], 8), [0, 7])
        with self.assertRaisesRegex(ValueError, "invalid configured"):
            select_layer_indices([8], 8)

    def test_adapter_layer_config_accepts_int_or_list(self):
        self.assertEqual(
            adapter_layers_from_config(SimpleNamespace(layers_to_transform=9)),
            [9],
        )
        self.assertEqual(
            adapter_layers_from_config(
                SimpleNamespace(layers_to_transform=[9, 8, 9])
            ),
            [8, 9],
        )


if __name__ == "__main__":
    unittest.main()
