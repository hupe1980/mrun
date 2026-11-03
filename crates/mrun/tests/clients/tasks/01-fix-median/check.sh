#!/bin/sh
python3 -I -m unittest -q 2>/dev/null && cmp -s test_stats.py "$FIXTURE/test_stats.py"
