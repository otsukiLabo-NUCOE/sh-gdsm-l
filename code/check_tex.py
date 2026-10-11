#!/usr/bin/env python3
"""Static checks for the manuscript when no TeX installation is available:
brace balance, \\begin/\\end pairing, \\ref/\\eqref vs \\label, \\cite vs \\bibitem,
\\input targets, unknown \\N... macros; also reorders the bibliography by
first citation (IEEE style) when --fix-bib is given."""
import argparse
import os
import re
import sys


def expand(root, rel, seen=None):
    path = os.path.join(root, rel if rel.endswith(".tex") else rel + ".tex")
    if not os.path.exists(path):
        return None, [f"missing \\input target: {rel}"]
    txt = open(path, encoding="utf-8").read()
    txt = re.sub(r"(?<!\\)%.*", "", txt)
    errs = []
    def sub(m):
        inner, e = expand(root, m.group(1))
        errs.extend(e)
        return inner or ""
    txt = re.sub(r"\\input\{([^}]+)\}", sub, txt)
    return txt, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../IEEE_Access/manuscript")
    ap.add_argument("--fix-bib", action="store_true")
    a = ap.parse_args()
    doc, errs = expand(a.root, "main")
    # braces
    depth = 0
    for i, ch in enumerate(doc):
        if ch == "{" and doc[i - 1] != "\\":
            depth += 1
        elif ch == "}" and doc[i - 1] != "\\":
            depth -= 1
            if depth < 0:
                errs.append(f"unbalanced '}}' near: {doc[max(0,i-60):i+10]!r}")
                depth = 0
    if depth:
        errs.append(f"{depth} unclosed '{{'")
    # environments
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", doc):
        if m.group(1) == "begin":
            stack.append(m.group(2))
        else:
            if not stack or stack[-1] != m.group(2):
                errs.append(f"\\end{{{m.group(2)}}} does not match {stack[-1] if stack else None}")
            else:
                stack.pop()
    if stack:
        errs.append(f"unclosed environments: {stack}")
    labels = re.findall(r"\\label\{([^}]+)\}", doc)
    dup = {l for l in labels if labels.count(l) > 1}
    if dup:
        errs.append(f"duplicate labels: {sorted(dup)}")
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", doc))
    for r in sorted(refs - set(labels)):
        errs.append(f"undefined reference: {r}")
    cites = []
    for m in re.finditer(r"\\cite\{([^}]+)\}", doc):
        for k in m.group(1).split(","):
            k = k.strip()
            if k not in cites:
                cites.append(k)
    bib_path = os.path.join(a.root, "sections", "references.tex")
    bib = open(bib_path, encoding="utf-8").read()
    items = re.findall(r"\\bibitem\{([^}]+)\}", bib)
    for c in cites:
        if c not in items:
            errs.append(f"undefined citation: {c}")
    unused = [i for i in items if i not in cites]
    if unused:
        errs.append(f"bibitems never cited: {unused}")
    nums = open(os.path.join(a.root, "numbers.tex"), encoding="utf-8").read()
    defined = set(re.findall(r"\\newcommand\{\\(N[A-Za-z]+)\}", nums))
    for u in sorted(set(re.findall(r"\\(N[A-Z][A-Za-z]*)", doc)) - defined):
        errs.append(f"undefined number macro: {u}")
    if "[pending]" in nums or any("[pending]" in open(os.path.join(a.root, "tables", f), encoding="utf-8").read()
                                  for f in os.listdir(os.path.join(a.root, "tables"))):
        print("note: some results are still [pending]")
    if a.fix_bib:
        blocks = re.split(r"(?=\\bibitem\{)", bib)
        head, entries = blocks[0], blocks[1:]
        tail = ""
        if entries and "\\end{thebibliography}" in entries[-1]:
            last, tail = entries[-1].split("\\end{thebibliography}")
            entries[-1] = last
            tail = "\\end{thebibliography}" + tail
        key = lambda e: re.match(r"\\bibitem\{([^}]+)\}", e).group(1)
        order = {k: i for i, k in enumerate(cites)}
        entries.sort(key=lambda e: order.get(key(e), 10_000))
        entries = [e.rstrip() + "\n\n" for e in entries]
        open(bib_path, "w", encoding="utf-8").write(head + "".join(entries) + tail)
        print("bibliography reordered by first citation")
    words = len(re.findall(r"[A-Za-z]+", open(os.path.join(a.root, "sections", "abstract.tex"), encoding="utf-8").read())) \
        if os.path.exists(os.path.join(a.root, "sections", "abstract.tex")) else 0
    print(f"abstract words (approx.): {words}")
    print("\n".join(errs) if errs else "no problems found")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
