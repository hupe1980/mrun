import unittest
from textkit.slugify import slugify
from textkit.count import word_count
from textkit.case import snake_to_camel, camel_to_snake
from textkit.initials import initials
from textkit.wrap import wrap


class T(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("  Hello, World! "), "hello-world")

    def test_count(self):
        self.assertEqual(word_count("a b  c"), 3)

    def test_case(self):
        self.assertEqual(snake_to_camel("max_line_length"), "maxLineLength")
        self.assertEqual(camel_to_snake("maxLineLength"), "max_line_length")

    def test_initials(self):
        self.assertEqual(initials("Ada  King Lovelace"), "AKL")

    def test_wrap(self):
        self.assertEqual(wrap("one two three four", 9), ["one two", "three", "four"])


if __name__ == "__main__":
    unittest.main()
