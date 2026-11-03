from shop.pricing import calc_total


def summary(prices):
    return f"total: {calc_total(prices, 0.0)}"
