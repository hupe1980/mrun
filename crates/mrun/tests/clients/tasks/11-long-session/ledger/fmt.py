"""Formatting integer cents for people."""


def format_cents(cents):
    """Cents as a decimal string with two places: 1234 -> "12.34", -5 -> "-0.05"."""
    units, rest = divmod(cents, 100)
    return f"{units}.{rest:02d}"


def format_table(rows, width=12):
    """Rows of (label, cents) as right-aligned lines."""
    return [f"{label:<{width}}{format_cents(cents):>10}" for label, cents in rows]


def signed(cents):
    """Like format_cents, with an explicit "+" for positive amounts."""
    text = format_cents(cents)
    return text if cents <= 0 else "+" + text
