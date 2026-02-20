import unittest

from main import chunk_sequence


class TestChunkSequence(unittest.TestCase):
    def test_chunk_size_two(self):
        chunks = chunk_sequence([1, 2, 3, 4, 5], 2)
        self.assertEqual(chunks, [[1, 2], [3, 4], [5]])

    def test_chunk_size_zero_returns_all(self):
        chunks = chunk_sequence([1, 2, 3], 0)
        self.assertEqual(chunks, [[1, 2, 3]])

    def test_empty_items(self):
        chunks = chunk_sequence([], 500)
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
