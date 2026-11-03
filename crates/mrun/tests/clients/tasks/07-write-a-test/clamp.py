def clamp(x, lo, hi):
    """Return x limited to the closed range [lo, hi]. Raise ValueError if lo > hi."""
    if lo > hi:
        raise ValueError("lo > hi")
    return max(lo, min(x, hi))
