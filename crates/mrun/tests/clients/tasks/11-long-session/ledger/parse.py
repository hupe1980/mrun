"""Parse ledger lines.

A ledger line has four fields separated by "|":

    date | account | category | amount

for example ``2026-03-14 | checking | groceries | -42.10``. Fields are stripped; the amount is
converted to cents with `money.to_cents`.
"""

from ledger.money import to_cents

FIELDS = ("date", "account", "category", "amount")


class ParseError(ValueError):
    """A line that is not a ledger entry; carries the 1-based line number."""

    def __init__(self, lineno, message):
        super().__init__(f"line {lineno}: {message}")
        self.lineno = lineno


def parse_entry(line, lineno=1):
    """Parse one ledger line into a dict with date, account, category and amount (cents)."""
    parts = [p.strip() for p in line.split("|")]
    if len(parts) != len(FIELDS):
        raise ParseError(lineno, f"expected {len(FIELDS)} fields, got {len(parts)}")
    entry = dict(zip(FIELDS, parts))
    if not entry["account"]:
        raise ParseError(lineno, "empty account")
    try:
        entry["amount"] = to_cents(entry["amount"])
    except ValueError as error:
        raise ParseError(lineno, f"bad amount: {error}") from None
    return entry


def parse_lines(lines):
    """Parse an iterable of ledger lines into a list of entries."""
    entries = []
    for lineno, line in enumerate(lines, start=1):
        entries.append(parse_entry(line, lineno))
    return entries


def parse_text(text):
    """Parse a whole ledger file's text."""
    return parse_lines(text.splitlines())
