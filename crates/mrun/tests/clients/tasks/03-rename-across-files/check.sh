#!/bin/sh
python3 -I -m unittest -q 2>/dev/null && ! grep -rq calc_total shop && cmp -s test_shop.py "$FIXTURE/test_shop.py"
