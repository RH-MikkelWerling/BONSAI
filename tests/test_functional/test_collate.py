import unittest
import torch
from bonsai.functional.collate import dynamic_padding


class TestCollate(unittest.TestCase):
    def setUp(self):
        self.batch = [
            {
                "code": torch.tensor([1, 2]),
                "abspos": torch.tensor([0, 1]),
                "age": torch.tensor([10, 20]),
                "segment": torch.tensor([0, 0]),
                "attention_mask": torch.tensor([1, 1]),
                "target": torch.tensor([0, 1]),
                "subject_id": 1,
            },
            {
                "code": torch.tensor([3]),
                "abspos": torch.tensor([2]),
                "age": torch.tensor([30]),
                "segment": torch.tensor([1]),
                "attention_mask": torch.tensor([1]),
                "target": torch.tensor([1]),
                "subject_id": 2,
            },
        ]

    def test_dynamic_padding_shapes(self):
        output = dynamic_padding(self.batch)
        self.assertEqual(output["code"].shape, (2, 2))
        self.assertEqual(output["abspos"].shape, (2, 2))
        self.assertEqual(output["age"].shape, (2, 2))
        self.assertEqual(output["segment"].shape, (2, 2))
        self.assertEqual(output["attention_mask"].shape, (2, 2))
        self.assertEqual(output["target"].shape, (2, 2))
        self.assertEqual(output["subject_id"].shape, (2,))

    def test_dynamic_padding_values(self):
        output = dynamic_padding(self.batch)
        # Check padding value for target
        self.assertEqual(output["code"][1, 1].item(), 0)
        self.assertEqual(output["abspos"][1, 1].item(), 0)
        self.assertEqual(output["age"][1, 1].item(), 0)
        self.assertEqual(output["segment"][1, 1].item(), 0)
        self.assertEqual(output["attention_mask"][1, 1].item(), 0)
        self.assertEqual(output["target"][1, 1].item(), -100)

    def test_dynamic_padding_value_targets(self):
        batch = [
            {
                "code": torch.tensor([1, 2]),
                "value_bin": torch.tensor([0, 3]),
                "value_normalized": torch.tensor([0.0, 0.7]),
                "value_present": torch.tensor([False, True]),
                "target_value_bin": torch.tensor([-100, 3]),
                "target_value_normalized": torch.tensor([0.0, 0.7]),
                "target_value_mask": torch.tensor([False, True]),
            },
            {
                "code": torch.tensor([4]),
                "value_bin": torch.tensor([2]),
                "value_normalized": torch.tensor([0.2]),
                "value_present": torch.tensor([True]),
                "target_value_bin": torch.tensor([2]),
                "target_value_normalized": torch.tensor([0.2]),
                "target_value_mask": torch.tensor([True]),
            },
        ]

        output = dynamic_padding(batch)

        self.assertEqual(output["value_bin"][1, 1].item(), 0)
        self.assertEqual(output["value_normalized"][1, 1].item(), 0.0)
        self.assertFalse(output["value_present"][1, 1].item())
        self.assertEqual(output["target_value_bin"][1, 1].item(), -100)
        self.assertEqual(output["target_value_normalized"][1, 1].item(), 0.0)
        self.assertFalse(output["target_value_mask"][1, 1].item())
