#!/bin/sh
python3 -I -m unittest -q 2>/dev/null && cmp -s test_textkit.py "$FIXTURE/test_textkit.py"
