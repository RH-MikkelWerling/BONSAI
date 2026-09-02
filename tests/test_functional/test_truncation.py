import unittest
import torch
from bonsai.functional.truncation import truncate_subject


class TestTruncation(unittest.TestCase):
    def setUp(self) -> None:
        self.subject = {
            "subject_id": 10,
            "code": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "abspos": torch.tensor([0, 0, 0, 1, 1, 2, 3, 4, 5, 6]),
            "age": torch.tensor([0, 0, 0, 10, 10, 20, 30, 40, 50, 60]),
            "segment": torch.tensor([0, 0, 0, 1, 1, 2, 3, 4, 5, 6]),
        }

    def test_no_truncate(self):
        post_subject = truncate_subject(self.subject, max_len=100, background_length=3)
        self.assertEqual(post_subject["subject_id"], self.subject["subject_id"])
        for key in ("code", "abspos", "age", "segment"):
            torch.testing.assert_close(post_subject[key], self.subject[key])
            self.assertIsNot(post_subject[key], self.subject[key])

    def test_truncate(self):
        post_subject = truncate_subject(self.subject, max_len=5, background_length=3)
        expected_subject = {
            "subject_id": self.subject["subject_id"],
            "code": torch.tensor([1, 2, 3, 9, 10]),
            "abspos": torch.tensor([0, 0, 0, 5, 6]),
            "age": torch.tensor([0, 0, 0, 50, 60]),
            "segment": torch.tensor([0, 0, 0, 5, 6]),
        }
        self.assertEqual(
            post_subject.pop("subject_id"), expected_subject.pop("subject_id")
        )
        for key in post_subject:
            torch.testing.assert_close(post_subject[key], expected_subject[key])

    def test_equal_length(self):
        post_subject = truncate_subject(self.subject, max_len=7, background_length=3)
        torch.testing.assert_close(
            post_subject["code"], torch.tensor([1, 2, 3, 7, 8, 9, 10])
        )
        self.assertEqual(len(post_subject["code"]), 7)

    def test_truncate_preserves_optional_sequence_value_fields(self):
        subject = {
            **self.subject,
            "value_bin": torch.tensor([0, 0, 0, 2, 3, 0, 4, 5, 6, 7]),
            "value_normalized": torch.tensor(
                [0.0, 0.0, 0.0, 0.2, 0.3, 0.0, 0.4, 0.5, 0.6, 0.7]
            ),
            "value_present": torch.tensor(
                [False, False, False, True, True, False, True, True, True, True]
            ),
        }

        post_subject = truncate_subject(subject, max_len=5, background_length=3)

        torch.testing.assert_close(
            post_subject["value_bin"], torch.tensor([0, 0, 0, 6, 7])
        )
        torch.testing.assert_close(
            post_subject["value_normalized"],
            torch.tensor([0.0, 0.0, 0.0, 0.6, 0.7]),
        )
        torch.testing.assert_close(
            post_subject["value_present"],
            torch.tensor([False, False, False, True, True]),
        )

    def test_random_window_does_not_split_event_groups(self):
        subject = {
            "subject_id": 12,
            "code": torch.arange(10),
            "abspos": torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3]),
            "age": torch.arange(10),
            "segment": torch.tensor([1, 1, 2, 2, 2, 3, 3, 4, 4, 4]),
        }
        truncated = truncate_subject(
            subject,
            max_len=7,
            background_length=2,
            strategy="random_window",
            generator=torch.Generator().manual_seed(4),
        )
        # Every retained clinical segment is complete relative to the source.
        for segment in truncated["segment"][2:].unique():
            assert int((truncated["segment"] == segment).sum()) == int(
                (subject["segment"] == segment).sum()
            )

    def test_zero_background_from_legacy_one_based_caller_is_inferred(self):
        subject = {
            "subject_id": 13,
            "code": torch.arange(8),
            "abspos": torch.arange(8),
            "age": torch.arange(8),
            "segment": torch.tensor([1, 1, 2, 3, 4, 5, 6, 7]),
        }
        truncated, metadata = truncate_subject(
            subject,
            max_len=5,
            background_length=0,
            return_metadata=True,
        )
        assert metadata["background_length"] == 2
        assert torch.equal(truncated["code"][:2], subject["code"][:2])
