#!/bin/sh
python3 -I -m unittest -q 2>/dev/null && cmp -s test_units.py "$FIXTURE/test_units.py"
