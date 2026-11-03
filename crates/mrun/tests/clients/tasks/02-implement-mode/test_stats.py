import unittest
from stats import mode


class T(unittest.TestCase):
    def test_single(self):
        self.assertEqual(mode([3, 1, 3, 2]), 3)

    def test_tie_takes_smallest(self):
        self.assertEqual(mode([5, 2, 5, 2, 9]), 2)

    def test_empty(self):
        with self.assertRaises(ValueError):
            mode([])


if __name__ == "__main__":
    unittest.main()
