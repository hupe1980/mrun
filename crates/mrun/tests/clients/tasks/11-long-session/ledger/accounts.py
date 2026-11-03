"""Per-account views of a list of entries."""

from ledger.money import add


def accounts_of(entries):
    """Account names in first-seen order."""
    seen = []
    for entry in entries:
        if entry["account"] not in seen:
            seen.append(entry["account"])
    return seen


def acct_balance(entries, account):
    """Balance of `account` in cents: the sum of its entries' amounts."""
    return add(*(e["amount"] for e in entries if e["account"] == account))


def balances(entries):
    """{account: balance} for every account in the entries."""
    return {name: acct_balance(entries, name) for name in accounts_of(entries)}


def overdrawn(entries):
    """Accounts whose balance is below zero, in first-seen order."""
    return [name for name in accounts_of(entries) if acct_balance(entries, name) < 0]
