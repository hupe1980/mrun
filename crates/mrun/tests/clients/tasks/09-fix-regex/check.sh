#!/bin/sh
python3 -I -m unittest -q 2>/dev/null && cmp -s test_logparse.py "$FIXTURE/test_logparse.py"
