import unittest

from metrics import distance, score


class MetricsTests(unittest.TestCase):
    def test_error_rates_are_weighted_and_preserve_raw_result(self):
        result = score(["지금 마이크 소리 잘 들리시나요?", "Hello OBS!"],
                       ["지금마이크 소리 잘 들리시나요!", "hello obs"])
        self.assertEqual(result["cer"]["percent"], 0)
        self.assertGreater(result["cer_raw"]["percent"], 0)
        self.assertEqual(distance("지금 마이크", "지음 마크"), 2)
        self.assertEqual(score(["a", "abcdefghi"], ["b", "abcdefghi"])["cer"]["percent"], 10)


if __name__ == "__main__":
    unittest.main()
