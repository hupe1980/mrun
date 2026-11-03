"""Spending by category."""


def spending(entries):
    """{category: spent cents} over negative entries; spent amounts are positive numbers."""
    spent = {}
    for entry in entries:
        if entry["amount"] < 0:
            spent[entry["category"]] = spent.get(entry["category"], 0) - entry["amount"]
    return spent


def top_categories(entries, n):
    """The `n` categories with the most spending, as [(category, spent cents)].

    Uses `spending`. Largest first; ties are broken by category name, alphabetically. When
    fewer than `n` categories have spending, all of them are returned. `n` below 1 raises
    ValueError.
    """
    raise NotImplementedError
