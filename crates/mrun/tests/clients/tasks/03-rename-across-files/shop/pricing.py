def calc_total(prices, tax_rate):
    return round(sum(prices) * (1 + tax_rate), 2)
