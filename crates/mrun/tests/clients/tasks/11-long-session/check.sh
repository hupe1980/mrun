#!/bin/sh
python3 -I -m unittest -q 2>/dev/null || exit 1
cmp -s test_ledger.py "$FIXTURE/test_ledger.py" || exit 1
! grep -rq acct_balance ledger || exit 1
[ "$(grep -c '^- \[x\]' TODO.md)" = 8 ] || exit 1
! grep -q '^- \[ \]' TODO.md
