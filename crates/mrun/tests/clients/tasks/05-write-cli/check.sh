#!/bin/sh
[ "$(python3 -I csvsum.py sample.csv price)" = "4.50" ] || exit 1
[ "$(python3 -I csvsum.py sample.csv qty)" = "16.00" ] || exit 1
python3 -I csvsum.py sample.csv weight >/dev/null 2>err.txt; [ $? -eq 2 ] || exit 1
grep -q "unknown column: weight" err.txt
