import re

# Lines look like: 2026-10-16T08:33:43Z [WARN] disk: 84% used (host=web-1)
LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) "
    r"\[(?P<level>[A-Z])\] "
    r"(?P<source>\w+): (?P<message>.*?)"
    r"(?: \(host=(?P<host>[\w-]+)\))?$"
)


def parse(line):
    m = LINE.match(line)
    return m.groupdict() if m else None
