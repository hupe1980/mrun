"""Reports over entries."""

from ledger.accounts import acct_balance, accounts_of
from ledger.dates import month_key
from ledger.fmt import format_cents


def monthly_totals(entries):
    """[(month "YYYY-MM", total cents)] for every month that has entries, oldest first."""
    totals = {}
    for entry in entries:
        key = month_key(entry["date"])
        totals[key] = totals.get(key, 0) + entry["amount"]
    return list(totals.items())


def account_lines(entries):
    """One "account: amount" line per account, in first-seen order."""
    return [f"{name}: {format_cents(acct_balance(entries, name))}" for name in accounts_of(entries)]


def summary(entries):
    """A small text report: balances, then monthly totals."""
    lines = ["Balances"]
    lines += ["  " + line for line in account_lines(entries)]
    lines.append("Months")
    lines += [f"  {month}: {format_cents(total)}" for month, total in monthly_totals(entries)]
    return "\n".join(lines)
