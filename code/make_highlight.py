#!/usr/bin/env python3
"""Highlighted PDF for the resubmission: words of the revised PDF that are not
present, in sequence, in the submitted PDF are highlighted in yellow.

    python make_highlight.py submitted.pdf revised.pdf highlighted.pdf
"""
import difflib
import re
import sys

import pymupdf

old_pdf, new_pdf, out_pdf = sys.argv[1:4]


def words(doc):
    out = []
    for pno, page in enumerate(doc):
        for w in page.get_text("words"):
            x0, y0, x1, y1, t = w[:5]
            n = re.sub(r"[^\w%.+\-]", "", t.lower())
            if n:
                out.append((n, pno, pymupdf.Rect(x0, y0, x1, y1)))
    return out


old = pymupdf.open(old_pdf)
new = pymupdf.open(new_pdf)
ow, nw = words(old), words(new)
sm = difflib.SequenceMatcher(None, [w[0] for w in ow], [w[0] for w in nw], autojunk=False)
marked = 0
ops = sm.get_opcodes()
# short runs of matching words (<= 3, e.g. "the", "of a") inside changed text
# are highlighted too, so that a rewritten sentence reads as one block
ops = [("insert" if t == "equal" and (j2 - j1) <= 3 and 0 < k < len(ops) - 1 else t, i1, i2, j1, j2)
       for k, (t, i1, i2, j1, j2) in enumerate(ops)]
for tag, i1, i2, j1, j2 in ops:
    if tag in ("replace", "insert"):
        by_page = {}
        for _, pno, r in nw[j1:j2]:
            by_page.setdefault(pno, []).append(r)
        for pno, rects in by_page.items():
            page = new[pno]
            annot = page.add_highlight_annot(rects)
            annot.set_colors(stroke=(1, 1, 0))
            annot.update()
            marked += len(rects)
new.save(out_pdf)
print(f"highlighted {marked} of {len(nw)} words ({100 * marked / len(nw):.0f}%) -> {out_pdf}")
