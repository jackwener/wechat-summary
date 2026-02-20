import unittest

from summarize import _sanitize_filename_component


class TestSanitizeFilenameComponent(unittest.TestCase):
    def test_replace_invalid_characters(self):
        self.assertEqual(
            _sanitize_filename_component('a/b:c*?"<>|'),
            "a_b_c______",
        )

    def test_fallback_for_empty_name(self):
        self.assertEqual(_sanitize_filename_component("..."), "group")

    def test_trim_and_limit_length(self):
        value = " " + ("x" * 100) + " "
        result = _sanitize_filename_component(value)
        self.assertEqual(len(result), 80)
        self.assertTrue(result.startswith("x"))


if __name__ == "__main__":
    unittest.main()
