import unittest
from shop.cart import Cart
from shop.report import summary
from shop import pricing


class T(unittest.TestCase):
    def test_cart(self):
        c = Cart()
        c.add(10)
        c.add(5)
        self.assertEqual(c.total(0.1), 16.5)

    def test_report(self):
        self.assertEqual(summary([1, 2]), "total: 3.0")

    def test_renamed(self):
        self.assertTrue(hasattr(pricing, "total_with_tax"))
        self.assertFalse(hasattr(pricing, "calc_total"))


if __name__ == "__main__":
    unittest.main()
