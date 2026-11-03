#!/bin/sh
python3 -I -m unittest -q 2>/dev/null && cmp -s test_transforms.py "$FIXTURE/test_transforms.py" && [ "$(grep -c "^def transform_" transforms.py)" = 120 ]
