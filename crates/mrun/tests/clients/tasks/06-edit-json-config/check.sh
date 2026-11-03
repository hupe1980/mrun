#!/bin/sh
python3 -I - <<'PY'
import json, sys
c = json.load(open("config.json"))
want = {"server": {"host": "127.0.0.1", "port": 9090, "workers": 4},
        "logging": {"level": "debug", "file": None},
        "features": ["search", "export", "import"]}
sys.exit(0 if c == want else 1)
PY
