def snake_to_camel(name):
    head, *rest = name.split("_")
    return head + "".join(part.capitalize() for part in rest)


def camel_to_snake(name):
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)
