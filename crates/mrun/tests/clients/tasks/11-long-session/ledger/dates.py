"""Dates in ledger entries are ISO strings, YYYY-MM-DD."""


def split_date(text):
    """(year, month, day) of an ISO date string; raises ValueError when it is not one."""
    parts = text.strip().split("-")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"not an ISO date: {text!r}")
    year, month, day = (int(p) for p in parts)
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        raise ValueError(f"not an ISO date: {text!r}")
    return year, month, day


def month_key(text):
    """The month of an ISO date as "YYYY-MM", e.g. month_key("2026-03-14") == "2026-03".

    Leading and trailing blanks are ignored; anything that is not an ISO date raises
    ValueError (use `split_date`).
    """
    raise NotImplementedError


def year_of(text):
    """The year of an ISO date as an int."""
    return split_date(text)[0]


def in_range(text, start, end):
    """Whether date `text` lies in [start, end] (all ISO strings)."""
    return start <= text.strip() <= end
