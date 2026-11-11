#!/usr/bin/env python3
"""Reorder the bibitems of sections/references.tex in order of first citation
(IEEE style), following \\input files in document order. Reports uncited items."""
import io
import os
import re
import sys

root = sys.argv[1]
seen = []


def walk(path):
    if not path.endswith(".tex"):
        path += ".tex"
    full = os.path.join(root, path)
    if not os.path.exists(full) or path.endswith("references.tex"):
        return
    text = io.open(full, encoding="utf-8").read()
    text = re.sub(r"(?<!\\)%.*", "", text)
    for m in re.finditer(r"\\input\{([^}]+)\}|\\cite\{([^}]+)\}", text):
        if m.group(1):
            walk(m.group(1))
        else:
            for k in m.group(2).split(","):
                k = k.strip()
                if k not in seen:
                    seen.append(k)


walk("main.tex")
ref_path = os.path.join(root, "sections", "references.tex")
ref = io.open(ref_path, encoding="utf-8").read()
head, rest = ref.split("\\bibitem", 1)
body, tail = rest.rsplit("\\end{thebibliography}", 1)
items = {}
for chunk in ("\\bibitem" + body).split("\\bibitem")[1:]:
    key = re.match(r"\{([^}]+)\}", chunk).group(1)
    items[key] = "\\bibitem" + chunk.rstrip() + "\n\n"
missing = [k for k in seen if k not in items]
uncited = [k for k in items if k not in seen]
order = [k for k in seen if k in items]
io.open(ref_path, "w", encoding="utf-8", newline="").write(
    head + "".join(items[k] for k in order) + "\\end{thebibliography}" + tail)
print(f"{len(order)} cited items reordered; missing bibitems: {missing}; uncited (removed): {uncited}")
