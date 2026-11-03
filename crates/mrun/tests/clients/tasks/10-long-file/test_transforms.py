import unittest
import transforms


class T(unittest.TestCase):
    def test_all(self):
        import re
        for name in dir(transforms):
            if name.startswith('transform_'):
                f = getattr(transforms, name)
                a, b = map(int, re.findall(r'(\d+)', f.__doc__))
                self.assertEqual(f(10), a * 10 + b, name)


if __name__ == "__main__":
    unittest.main()
