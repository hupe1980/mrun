import unittest
from units import convert, to_meters


class T(unittest.TestCase):
    def test_to_meters(self):
        self.assertAlmostEqual(to_meters(250, "cm"), 2.5)
        self.assertAlmostEqual(to_meters(2, "km"), 2000)

    def test_convert(self):
        self.assertEqual(convert(1, "ft", "in"), 12.0)
        self.assertEqual(convert(1000, "mm", "m"), 1.0)

    def test_unknown(self):
        with self.assertRaises(ValueError):
            to_meters(1, "yard")


if __name__ == "__main__":
    unittest.main()
