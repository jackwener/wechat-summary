import tempfile
import unittest
from pathlib import Path

from ocr import _collect_screenshot_files


class TestCollectScreenshotFiles(unittest.TestCase):
    def test_select_latest_run_and_reverse_page_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in [
                "20260219_120000_page_000.png",
                "20260219_120000_page_001.png",
                "20260220_090000_page_000.png",
                "20260220_090000_page_001.png",
                "20260220_090000_page_002.png",
            ]:
                (root / name).write_bytes(b"x")

            files = _collect_screenshot_files(root, None)
            names = [f.name for f in files]

            self.assertEqual(
                names,
                [
                    "20260220_090000_page_002.png",
                    "20260220_090000_page_001.png",
                    "20260220_090000_page_000.png",
                ],
            )

    def test_preserve_explicit_order_for_file_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "a.png"
            second = root / "b.png"
            first.write_bytes(b"x")
            second.write_bytes(b"x")

            files = _collect_screenshot_files(root, [str(second), str(first)])
            names = [f.name for f in files]

            self.assertEqual(names, ["b.png", "a.png"])


if __name__ == "__main__":
    unittest.main()
