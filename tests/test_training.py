import unittest

from train import data_history


def splits(train="a", validation="b", test="c"):
    return {name: [{"session": value, "audio_sha256": f"hash-{value}"}]
            for name, value in (("train", train), ("validation", validation), ("test", test))}


class HistoryTests(unittest.TestCase):
    def test_warm_start_preserves_old_training_and_held_out_boundaries(self):
        first = data_history(splits(), {})
        second = data_history(splits(train="d", validation="e", test="c"), first)
        self.assertEqual(second["training_sessions"], ["a", "d"])
        self.assertEqual(second["selection_sessions"], ["a", "b", "d", "e"])
        with self.assertRaises(ValueError):
            data_history(splits(train="d", validation="e", test="a"), first)
        with self.assertRaises(ValueError):
            data_history(splits(train="c", validation="e", test="f"), first)
        with self.assertRaises(ValueError):
            data_history(splits(train="d", validation="a", test="f"), first)


if __name__ == "__main__":
    unittest.main()
