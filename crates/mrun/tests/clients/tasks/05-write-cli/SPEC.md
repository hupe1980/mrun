# csvsum

Write `csvsum.py`, a command-line tool:

    python3 csvsum.py FILE COLUMN

It reads the CSV file FILE (with a header row) and prints the sum of the numeric column named
COLUMN, formatted with exactly two decimals (for example `12.50`). If the column does not exist,
it prints `unknown column: COLUMN` to stderr and exits with status 2. Use only the standard
library.
