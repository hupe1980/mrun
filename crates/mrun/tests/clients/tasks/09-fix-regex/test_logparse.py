import unittest
from logparse import parse


class T(unittest.TestCase):
    def test_with_host(self):
        r = parse("2026-10-16T08:33:43Z [WARN] disk: 84% used (host=web-1)")
        self.assertEqual(r["level"], "WARN")
        self.assertEqual(r["source"], "disk")
        self.assertEqual(r["message"], "84% used")
        self.assertEqual(r["host"], "web-1")

    def test_without_host(self):
        r = parse("2026-10-16T08:33:44Z [ERROR] db: connection refused")
        self.assertEqual(r["level"], "ERROR")
        self.assertEqual(r["message"], "connection refused")
        self.assertIsNone(r["host"])

    def test_garbage(self):
        self.assertIsNone(parse("not a log line"))


if __name__ == "__main__":
    unittest.main()
