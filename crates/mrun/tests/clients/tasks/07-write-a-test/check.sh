#!/bin/sh
# Tests must pass on the real clamp and fail on each planted bug.
[ -f test_clamp.py ] || exit 1
cmp -s clamp.py "$FIXTURE/clamp.py" || exit 1
python3 -I -m unittest discover -q -p test_clamp.py 2>/dev/null || exit 1
cp clamp.py clamp.orig
for bug in 'return max(lo, min(x, hi - 1))' 'return min(x, hi)' 'return max(lo, x)'; do
  python3 -I - "$bug" <<'PY'
import sys
src = open("clamp.orig").read().replace("return max(lo, min(x, hi))", sys.argv[1])
open("clamp.py", "w").write(src)
PY
  if python3 -I -m unittest discover -q -p test_clamp.py 2>/dev/null; then cp clamp.orig clamp.py; exit 1; fi
done
sed 's/if lo > hi:/if False:/' clamp.orig > clamp.py
if python3 -I -m unittest discover -q -p test_clamp.py 2>/dev/null; then cp clamp.orig clamp.py; exit 1; fi
cp clamp.orig clamp.py; rm clamp.orig
