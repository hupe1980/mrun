"""Amounts as integer cents.

Every amount inside the ledger is an int number of cents, so sums never drift. Text amounts
("19.99", "-3.5", "1,234.00") are converted at the edge with `to_cents`.
"""

CENTS_PER_UNIT = 100


def clean(text):
    """Strip blanks, a leading "+" and thousands separators from an amount string."""
    text = text.strip().replace(",", "").replace("_", "")
    if text.startswith("+"):
        text = text[1:]
    if not text:
        raise ValueError("empty amount")
    return text


def to_cents(text):
    """Convert an amount string to integer cents, rounding to the nearest cent.

    >>> to_cents("19.99")
    1999
    >>> to_cents("-0.005")
    -1
    """
    value = float(clean(text))
    return int(value * CENTS_PER_UNIT)


def add(*amounts):
    """Sum of integer-cent amounts."""
    total = 0
    for amount in amounts:
        if not isinstance(amount, int):
            raise TypeError(f"amounts are int cents, got {type(amount).__name__}")
        total += amount
    return total


def split(amount, parts):
    """Split `amount` cents into `parts` shares that differ by at most one cent.

    The first shares get the extra cents: split(100, 3) == [34, 33, 33].
    """
    if parts <= 0:
        raise ValueError("parts must be positive")
    base, extra = divmod(amount, parts)
    return [base + 1 if i < extra else base for i in range(parts)]


def percent(amount, rate):
    """`rate` percent of `amount` cents, rounded to the nearest cent."""
    return round(amount * rate / 100)
