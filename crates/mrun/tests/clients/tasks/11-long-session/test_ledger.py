import json
import unittest

from ledger import accounts, categories, dates, fmt, money, parse, report

ENTRIES = [
    {"date": "2026-03-14", "account": "checking", "category": "groceries", "amount": -4210},
    {"date": "2026-01-02", "account": "checking", "category": "salary", "amount": 250000},
    {"date": "2026-02-20", "account": "card", "category": "travel", "amount": -18000},
    {"date": "2026-01-15", "account": "card", "category": "groceries", "amount": -5790},
    {"date": "2026-03-01", "account": "savings", "category": "transfer", "amount": 50000},
    {"date": "2026-02-03", "account": "card", "category": "books", "amount": -9000},
    {"date": "2026-02-28", "account": "checking", "category": "rent", "amount": -120000},
    {"date": "2026-03-30", "account": "card", "category": "fees", "amount": -9000},
]


class Item1Money(unittest.TestCase):
    def test_rounds_to_nearest_cent(self):
        self.assertEqual(money.to_cents("19.99"), 1999)
        self.assertEqual(money.to_cents("0.29"), 29)
        self.assertEqual(money.to_cents("1,234.565"), 123457)
        self.assertEqual(money.to_cents("-0.005"), -1)
        self.assertEqual(money.to_cents("+7"), 700)


class Item2Parse(unittest.TestCase):
    def test_skips_comments_and_blank_lines(self):
        text = "# household ledger\n\n2026-03-14 | checking | groceries | -42.10\n   # note\n"
        self.assertEqual(parse.parse_text(text), [ENTRIES[0]])

    def test_still_rejects_bad_lines(self):
        with self.assertRaises(parse.ParseError):
            parse.parse_lines(["2026-03-14 | checking | -42.10"])


class Item3Dates(unittest.TestCase):
    def test_month_key(self):
        self.assertEqual(dates.month_key("2026-03-14"), "2026-03")
        self.assertEqual(dates.month_key(" 2025-12-01 "), "2025-12")
        with self.assertRaises(ValueError):
            dates.month_key("14.03.2026")


class Item4Rename(unittest.TestCase):
    def test_balance(self):
        self.assertFalse(hasattr(accounts, "acct_balance"))
        self.assertEqual(accounts.balance(ENTRIES, "card"), -41790)
        self.assertEqual(accounts.overdrawn(ENTRIES), ["card"])


class Item5Months(unittest.TestCase):
    def test_monthly_totals_ascending(self):
        self.assertEqual(
            report.monthly_totals(ENTRIES),
            [("2026-01", 244210), ("2026-02", -147000), ("2026-03", 36790)],
        )


class Item6Format(unittest.TestCase):
    def test_negative_amounts(self):
        self.assertEqual(fmt.format_cents(-1234), "-12.34")
        self.assertEqual(fmt.format_cents(-5), "-0.05")
        self.assertEqual(fmt.format_cents(1234), "12.34")
        self.assertEqual(fmt.format_cents(0), "0.00")
        self.assertEqual(fmt.signed(-100), "-1.00")


class Item7Config(unittest.TestCase):
    def test_config(self):
        with open("config.json") as f:
            config = json.load(f)
        self.assertEqual(
            config,
            {
                "name": "household",
                "currency": "EUR",
                "accounts": ["checking", "savings", "card"],
                "fiscal_year_start": "01-01",
                "precision": 2,
            },
        )


class Item8Categories(unittest.TestCase):
    def test_top_categories(self):
        self.assertEqual(
            categories.top_categories(ENTRIES, 3),
            [("rent", 120000), ("travel", 18000), ("groceries", 10000)],
        )
        self.assertEqual(
            categories.top_categories(ENTRIES, 5)[3:], [("books", 9000), ("fees", 9000)]
        )
        self.assertEqual(len(categories.top_categories(ENTRIES, 50)), 5)
        with self.assertRaises(ValueError):
            categories.top_categories(ENTRIES, 0)


class Summary(unittest.TestCase):
    def test_summary(self):
        text = report.summary(ENTRIES)
        self.assertIn("card: -417.90", text)
        self.assertIn("2026-02: -1470.00", text)


if __name__ == "__main__":
    unittest.main()
