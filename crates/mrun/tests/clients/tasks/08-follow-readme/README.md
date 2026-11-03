# units

`units.py` converts lengths. Two functions are missing; implement them:

1. `to_meters(value, unit)`: convert `value` in `unit` to meters. Units: `"mm"`, `"cm"`, `"m"`,
   `"km"`, `"in"` (0.0254 m), `"ft"` (0.3048 m). Raise `ValueError` for any other unit.
2. `convert(value, src, dst)`: convert between any two of those units, rounded to 6 decimals.

Then run `python3 -m unittest`.
