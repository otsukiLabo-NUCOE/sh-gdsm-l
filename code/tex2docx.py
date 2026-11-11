#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the IEEE Access LaTeX manuscript to Word (.docx) without pandoc.

  python tex2docx.py --root ../IEEE_Access/manuscript --out ../IEEE_Access/SH-GDSM-L_manuscript.docx

Handles the subset of LaTeX used by this manuscript: sections (IEEE numbering),
cross-references (sections, tables, figures, equations, algorithm), citations
(numbered by order of the bibliography), inline and display math (converted to
native Word equations, OMML), booktabs tables, figures (PDF rendered to PNG),
itemize, the algorithmic environment, the bibliography and the biography.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from xml.sax.saxutils import escape

MNS = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" ' \
      'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'

# ---------------------------------------------------------------------------
# 1. Source assembly and macro expansion
# ---------------------------------------------------------------------------


def read_tex(root: str, rel: str) -> str:
    path = os.path.join(root, rel if rel.endswith(".tex") else rel + ".tex")
    txt = open(path, encoding="utf-8").read()
    txt = re.sub(r"(?<!\\)%[^\n]*", "", txt)
    return re.sub(r"\\input\{([^}]+)\}", lambda m: read_tex(root, m.group(1)), txt)


def brace_arg(s: str, i: int) -> Tuple[str, int]:
    """s[i] == '{' -> (content, index after closing brace)."""
    assert s[i] == "{", s[i:i + 30]
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces near: " + s[i:i + 60])


def collect_macros(src: str) -> Dict[str, str]:
    macros = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}", src):
        body, _ = brace_arg(src, m.end())
        macros[m.group(1)] = body
    return macros


def expand_macros(s: str, macros: Dict[str, str]) -> str:
    names = sorted(macros, key=len, reverse=True)
    for _ in range(6):
        before = s
        for n in names:
            body = macros[n].replace("\\xspace", "")
            s = re.sub(r"\\" + n + r"(?![A-Za-z])(\{\})?", lambda m, b=body: b, s)
        if s == before:
            break
    return s


# ---------------------------------------------------------------------------
# 2. LaTeX math -> OMML
# ---------------------------------------------------------------------------

SYM = {
    "rightarrow": "→", "to": "→", "leftrightarrow": "↔", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
    "neq": "≠", "in": "∈", "notin": "∉", "cup": "∪", "cap": "∩", "cdot": "·", "times": "×",
    "oplus": "⊕", "sim": "∼", "mid": "|", "ldots": "…", "dots": "…", "infty": "∞", "pm": "±",
    "approx": "≈", "lfloor": "⌊", "rfloor": "⌋", "square": "□", "checkmark": "✓", "epsilon": "ϵ",
    "emptyset": "∅", "prime": "′", "ast": "∗", "star": "⋆", "subset": "⊂", "subseteq": "⊆",
    "forall": "∀", "exists": "∃", "partial": "∂", "lambda": "λ", "alpha": "α", "beta": "β",
    "gamma": "γ", "delta": "δ", "theta": "θ", "rho": "ρ", "sigma": "σ", "tau": "τ", "pi": "π",
    "mu": "μ", "nu": "ν", "phi": "φ", "psi": "ψ", "omega": "ω", "eta": "η", "kappa": "κ",
    "Sigma": "Σ", "Delta": "Δ", "Gamma": "Γ", "Omega": "Ω", "Pi": "Π", "Theta": "Θ", "Lambda": "Λ",
    "{": "{", "}": "}", "%": "%", "_": "_", "&": "&", "#": "#", "lvert": "|", "rvert": "|",
    "vert": "|", "|": "‖", "langle": "⟨", "rangle": "⟩", "circ": "∘", "bullet": "•", "neg": "¬",
    "wedge": "∧", "vee": "∨", "Rightarrow": "⇒", "leftarrow": "←", "uparrow": "↑", "downarrow": "↓",
}
SPACES = {",": "\u2009", ";": "\u2005", ":": "\u2005", " ": " ", "quad": "\u2003", "qquad": "\u2003\u2003",
          "!": ""}
DELIMS = {"bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr", "big", "Big", "left", "right", "bigm", "Bigm"}
NARY = {"prod": "∏", "sum": "∑", "bigcup": "⋃", "bigcap": "⋂", "int": "∫"}
FUNCS = {"max", "min", "arg", "log", "exp", "lim", "sup", "inf", "argmax", "argmin", "Pr", "det"}
ACC = {"hat": "\u0302", "widehat": "\u0302", "tilde": "\u0303", "widetilde": "\u0303", "bar": "\u0305",
       "dot": "\u0307", "vec": "\u20D7"}


def mrun(t: str, upright: bool = False) -> str:
    if not t:
        return ""
    pr = '<m:rPr><m:sty m:val="p"/></m:rPr>' if upright else ""
    return f'<m:r>{pr}<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>' \
           f'<m:t xml:space="preserve">{escape(t)}</m:t></m:r>'


class MathParser:
    def __init__(self, s: str):
        self.toks = self.tokenize(s)
        self.i = 0

    @staticmethod
    def tokenize(s: str) -> List[str]:
        out, i = [], 0
        while i < len(s):
            c = s[i]
            if c == "\\":
                m = re.match(r"\\([A-Za-z]+|.)", s[i:])
                out.append("\\" + m.group(1))
                i += len(m.group(0))
            elif c.isspace():
                i += 1
            else:
                out.append(c)
                i += 1
        return out

    def peek(self) -> Optional[str]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> Optional[str]:
        t = self.peek()
        self.i += 1
        return t

    def group_tokens_text(self) -> str:
        """Raw text of a {...} group (for \\mathrm, \\text)."""
        if self.peek() != "{":
            t = self.take()
            return t.lstrip("\\") if t else ""
        self.take()
        depth, parts = 1, []
        while self.peek() is not None:
            t = self.take()
            if t == "{":
                depth += 1
            elif t == "}":
                depth -= 1
                if depth == 0:
                    break
            if t.startswith("\\") and len(t) > 1:
                name = t[1:]
                parts.append(SYM.get(name, SPACES.get(name, "" if name in DELIMS else name)))
            elif t not in "{}":
                parts.append(t)
        return "".join(parts)

    def parse_seq(self, stop=("}",)) -> str:
        out = []
        while self.peek() is not None and self.peek() not in stop:
            t = self.peek()
            if t in ("_", "^"):          # script without base
                out.append(self.scripts(""))
                continue
            atom = self.atom()
            if atom is None:
                continue
            if isinstance(atom, tuple):  # n-ary / limit operator consumes the rest
                kind, payload = atom
                rest = self.parse_seq(stop)
                if kind == "nary" and not rest:          # inline operator with nothing after it
                    chr_, sub_, sup_ = payload
                    out.append(f"<m:sSub><m:e>{mrun(chr_, True)}</m:e><m:sub>{sub_}</m:sub></m:sSub>"
                               if sub_ else mrun(chr_, True))
                elif kind == "nary":
                    chr_, sub_, sup_ = payload
                    out.append(f'<m:nary><m:naryPr><m:chr m:val="{chr_}"/><m:limLoc m:val="undOvr"/>'
                               f'{"" if sup_ else "<m:supHide m:val=\"1\"/>"}'
                               f'{"" if sub_ else "<m:subHide m:val=\"1\"/>"}</m:naryPr>'
                               f'<m:sub>{sub_}</m:sub><m:sup>{sup_}</m:sup><m:e>{rest}</m:e></m:nary>')
                else:
                    name, sub_ = payload
                    fname = (f'<m:limLow><m:e>{mrun(name, True)}</m:e><m:lim>{sub_}</m:lim></m:limLow>'
                             if sub_ else mrun(name, True))
                    out.append(f'<m:func><m:fName>{fname}</m:fName><m:e>{rest}</m:e></m:func>')
                break
            if self.peek() in ("_", "^"):
                atom = self.scripts(atom)
            out.append(atom)
        return "".join(out)

    def scripts(self, base: str) -> str:
        sub_ = sup_ = None
        while self.peek() in ("_", "^"):
            t = self.take()
            arg = self.arg()
            if t == "_":
                sub_ = arg
            else:
                sup_ = arg
        b = f"<m:e>{base}</m:e>"
        if sub_ is not None and sup_ is not None:
            return f"<m:sSubSup>{b}<m:sub>{sub_}</m:sub><m:sup>{sup_}</m:sup></m:sSubSup>"
        if sub_ is not None:
            return f"<m:sSub>{b}<m:sub>{sub_}</m:sub></m:sSub>"
        return f"<m:sSup>{b}<m:sup>{sup_}</m:sup></m:sSup>"

    def arg(self) -> str:
        if self.peek() == "{":
            self.take()
            s = self.parse_seq(("}",))
            self.take()
            return s
        a = self.atom()
        return a if isinstance(a, str) else ""

    def limits(self) -> Tuple[str, str]:
        sub_ = sup_ = ""
        if self.peek() == "\\limits":
            self.take()
        while self.peek() in ("_", "^"):
            t = self.take()
            a = self.arg()
            if t == "_":
                sub_ = a
            else:
                sup_ = a
        return sub_, sup_

    def atom(self):
        t = self.take()
        if t == "{":
            s = self.parse_seq(("}",))
            self.take()
            return s
        if t == "}":
            return None
        if not t.startswith("\\") or len(t) == 1:
            if t == "&":
                return ""
            if t == "'":
                return mrun("′")
            return mrun(t, upright=not t.isalpha())
        name = t[1:]
        if name in ("frac", "dfrac", "tfrac"):
            a, b = self.arg(), self.arg()
            return f"<m:f><m:num>{a}</m:num><m:den>{b}</m:den></m:f>"
        if name in ACC:
            a = self.arg()
            return f'<m:acc><m:accPr><m:chr m:val="{ACC[name]}"/></m:accPr><m:e>{a}</m:e></m:acc>'
        if name == "overline":
            a = self.arg()
            return f'<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>{a}</m:e></m:bar>'
        if name in ("mathrm", "text", "textrm", "operatorname", "mathsf", "textnormal", "mbox"):
            return mrun(self.group_tokens_text(), upright=True)
        if name == "mathbb":
            txt = self.group_tokens_text()
            return mrun("".join({"E": "𝔼", "R": "ℝ", "N": "ℕ", "P": "ℙ",
                                 "Z": "ℤ"}.get(c, c) for c in txt), True)
        if name in ("mathit", "textit"):
            return mrun(self.group_tokens_text())
        if name in ("mathbf", "boldsymbol", "bm", "textbf"):
            return mrun(self.group_tokens_text(), upright=True)
        if name in ("label", "nonumber", "notag"):
            if name == "label":
                self.group_tokens_text()
            return ""
        if name in NARY:
            sub_, sup_ = self.limits()
            return ("nary", (NARY[name], sub_, sup_))
        if name in FUNCS:
            sub_, _ = self.limits()
            return ("func", (name, sub_))
        if name in DELIMS:
            d = self.take() or ""
            if d.startswith("\\"):
                d = SYM.get(d[1:], d[1:])
            return mrun(d, upright=True)
        if name in SPACES:
            return mrun(SPACES[name], upright=True)
        if name in SYM:
            return mrun(SYM[name], upright=True)
        if name == "hfill":
            return ""
        return mrun(name, upright=True)


def omml(latex: str) -> str:
    body = MathParser(latex).parse_seq(stop=())
    return f"<m:oMath {MNS}>{body}</m:oMath>"


# ---------------------------------------------------------------------------
# 3. Document model: labels, citations
# ---------------------------------------------------------------------------

def _roman(n: int) -> str:
    out = ""
    for v, s in ((40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out += s
            n -= v
    return out


ROMAN = [_roman(n) for n in range(1, 61)]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DISPLAY_ENVS = ("equation", "align", "multline", "equation*", "align*", "multline*", "gather")


def split_math_lines(body: str) -> List[str]:
    return [l for l in re.split(r"\\\\", body) if l.strip()]


def number_labels(doc: str) -> Dict[str, str]:
    """Walk the document in order and assign the printed number of every label."""
    lab: Dict[str, str] = {}
    sec = sub = subsub = 0
    tab = fig = eq = alg = 0
    in_app = False
    current = ""
    pat = re.compile(r"\\(appendices|section\*?|subsection|subsubsection|label)\{?|"
                     r"\\begin\{(table\*?|figure\*?|algorithm|" + "|".join(re.escape(e) for e in DISPLAY_ENVS) + r")\}")
    i = 0
    stack_env: List[str] = []
    while True:
        m = pat.search(doc, i)
        if not m:
            break
        if m.group(1) == "appendices":
            in_app, sec = True, 0
            i = m.end()
            continue
        if m.group(1) in ("section", "subsection", "subsubsection"):
            kind = m.group(1)
            _, j = brace_arg(doc, m.end() - 1)
            if kind == "section":
                sec += 1
                sub = subsub = 0
                current = LETTERS[sec - 1] if in_app else ROMAN[sec - 1]
            elif kind == "subsection":
                sub += 1
                subsub = 0
                current = f"{LETTERS[sec - 1] if in_app else ROMAN[sec - 1]}-{LETTERS[sub - 1]}"
            else:
                subsub += 1
                current = f"{LETTERS[sec - 1] if in_app else ROMAN[sec - 1]}-{LETTERS[sub - 1]}{subsub}"
            i = j
            continue
        if m.group(1) == "section*":
            _, j = brace_arg(doc, m.end() - 1)
            i = j
            continue
        if m.group(2):
            env = m.group(2)
            end = doc.find("\\end{" + env + "}", m.end())
            body = doc[m.end():end]
            if env.startswith("table"):
                tab += 1
                for l in re.findall(r"\\label\{([^}]+)\}", body):
                    lab[l] = ROMAN[tab - 1]
            elif env.startswith("figure"):
                fig += 1
                for l in re.findall(r"\\label\{([^}]+)\}", body):
                    lab[l] = str(fig)
            elif env == "algorithm":
                alg += 1
                for l in re.findall(r"\\label\{([^}]+)\}", body):
                    lab[l] = str(alg)
            else:
                lines = split_math_lines(body) if env.startswith(("align", "gather")) else [body]
                for line in lines:
                    if env.endswith("*") or "\\nonumber" in line or "\\notag" in line:
                        continue
                    eq += 1
                    for l in re.findall(r"\\label\{([^}]+)\}", line):
                        lab[l] = str(eq)
            i = end + len(env) + 6
            continue
        if m.group(1) == "label":
            name, j = brace_arg(doc, m.end() - 1)
            lab.setdefault(name, current)
            i = j
            continue
        i = m.end()
    return lab


# ---------------------------------------------------------------------------
# 4. Inline text conversion
# ---------------------------------------------------------------------------

TEXT_SYM = {"%": "%", "_": "_", "&": "&", "#": "#", "$": "$", "{": "{", "}": "}", ",": "\u2009",
            " ": " ", "ldots": "…", "dots": "…", "textbackslash": "\\", "checkmark": "✓",
            "S": "§", "copyright": "©", "o": "ø", "i": "ı"}
DROP_CMDS = {"centering", "noindent", "small", "footnotesize", "normalsize", "large", "Large", "hfill",
             "par", "break", "relax", "protect", "titlepgskip", "maketitle", "selectfont"}


class Seg:
    def __init__(self, kind: str, text: str = "", b=False, it=False, tt=False, sup=False, xml: str = ""):
        self.kind, self.text, self.b, self.it, self.tt, self.sup, self.xml = kind, text, b, it, tt, sup, xml


def typography(t: str) -> str:
    t = t.replace("---", "—").replace("--", "–").replace("``", "“").replace("''", "”")
    t = t.replace("`", "‘").replace("'", "’").replace("~", "\u00a0")
    return re.sub(r"[ \t\n]+", " ", t)


class Inline:
    def __init__(self, labels: Dict[str, str], cites: Dict[str, int]):
        self.labels, self.cites = labels, cites

    def cite(self, keys: str) -> str:
        nums = sorted(self.cites.get(k.strip(), 0) for k in keys.split(","))
        parts, run = [], []
        for n in nums:
            if run and n == run[-1] + 1:
                run.append(n)
            else:
                if run:
                    parts.append(run)
                run = [n]
        if run:
            parts.append(run)
        out = []
        for r in parts:
            out.append(f"[{r[0]}]–[{r[-1]}]" if len(r) > 2 else ", ".join(f"[{x}]" for x in r))
        return ", ".join(out)

    ACCENTS = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303",
               "v": "\u030C", "c": "\u0327", "=": "\u0304", ".": "\u0307", "u": "\u0306"}

    def accents(self, s: str) -> str:
        import unicodedata

        def sub(m):
            base = m.group(2) or m.group(3)
            base = "i" if base in ("\\i", "i") else base
            return unicodedata.normalize("NFC", base + self.ACCENTS[m.group(1)])
        return re.sub(r"\\([\'`^\"~v=.uc])(?:\{(\\i|[A-Za-z])\}|([A-Za-z]))", sub, s)

    def convert(self, s: str, b=False, it=False, tt=False) -> List[Seg]:
        s = self.accents(s)
        segs: List[Seg] = []
        buf = []

        def flush():
            if buf:
                segs.append(Seg("text", typography("".join(buf)), b, it, tt))
                buf.clear()

        i = 0
        while i < len(s):
            c = s[i]
            if c == "$" or s.startswith("\\(", i):
                flush()
                if s.startswith("\\(", i):
                    j = s.index("\\)", i)
                    math, i = s[i + 2:j], j + 2
                else:
                    j = s.index("$", i + 1)
                    math, i = s[i + 1:j], j + 1
                m = re.fullmatch(r"\s*\^\{?([^{}]*)\}?\s*", math)
                if m:                                  # $^{a}$ footnote mark
                    segs.append(Seg("text", m.group(1), b, it, tt, sup=True))
                else:
                    segs.append(Seg("math", xml=omml(math)))
                continue
            if c == "\\":
                m = re.match(r"\\([A-Za-z]+|.)\*?", s[i:])
                name = m.group(1)
                j = i + len(m.group(0))
                if name in ("emph", "textit"):
                    flush()
                    a, j = brace_arg(s, j)
                    segs += self.convert(a, b, not it, tt)
                    i = j
                    continue
                if name == "textbf":
                    flush()
                    a, j = brace_arg(s, j)
                    segs += self.convert(a, True, it, tt)
                    i = j
                    continue
                if name in ("texttt", "url"):
                    flush()
                    a, j = brace_arg(s, j)
                    segs += self.convert(a, b, it, True)
                    i = j
                    continue
                if name == "textsuperscript":
                    flush()
                    a, j = brace_arg(s, j)
                    segs.append(Seg("text", a, b, it, tt, sup=True))
                    i = j
                    continue
                if name == "cite":
                    a, j = brace_arg(s, j)
                    buf.append(self.cite(a))
                    i = j
                    continue
                if name in ("ref", "eqref"):
                    a, j = brace_arg(s, j)
                    v = self.labels.get(a, "??")
                    buf.append(f"({v})" if name == "eqref" else v)
                    i = j
                    continue
                if name == "label":
                    _, j = brace_arg(s, j)
                    i = j
                    continue
                if name == "PARstart":
                    a, j = brace_arg(s, j)
                    b2, j = brace_arg(s, j)
                    buf.append(a + b2)
                    i = j
                    continue
                if name in ("setlength",):
                    _, j = brace_arg(s, j)
                    _, j = brace_arg(s, j)
                    i = j
                    continue
                if name in DROP_CMDS:
                    i = j
                    continue
                if name == "\\":
                    buf.append(" ")
                    i = j
                    continue
                if name in TEXT_SYM:
                    buf.append(TEXT_SYM[name])
                    i = j
                    continue
                if name in SYM:
                    buf.append(SYM[name])
                    i = j
                    continue
                buf.append(name)
                i = j
                continue
            if c in "{}":
                i += 1
                continue
            buf.append(c)
            i += 1
        flush()
        return segs


# ---------------------------------------------------------------------------
# 5. Writer
# ---------------------------------------------------------------------------

class Writer:
    def __init__(self, root: str, inline: Inline, labels: Dict[str, str], figdir: str):
        self.root, self.il, self.labels, self.figdir = root, inline, labels, figdir
        self.doc = Document()
        st = self.doc.styles["Normal"]
        st.font.name = "Times New Roman"
        st.font.size = Pt(10)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        pf = st.paragraph_format
        pf.space_after = Pt(3)
        pf.line_spacing = 1.08
        for s in self.doc.sections:
            s.left_margin = s.right_margin = Inches(0.8)
            s.top_margin = s.bottom_margin = Inches(0.8)
        for lvl, size in ((1, 11), (2, 10.5), (3, 10)):
            h = self.doc.styles[f"Heading {lvl}"]
            h.font.name = "Times New Roman"
            h.font.size = Pt(size)
            h.font.color.rgb = RGBColor(0, 0, 0)
            h.font.bold = lvl == 1
            h.font.italic = lvl > 1
            rf = h.element.rPr.rFonts
            for att in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                rf.attrib.pop(qn(att), None)
            rf.set(qn("w:eastAsia"), "Times New Roman")
            h.paragraph_format.space_before = Pt(10 if lvl == 1 else 6)
            h.paragraph_format.space_after = Pt(3)
        self.sec = self.sub = self.subsub = 0
        self.in_app = False
        self.eq = 0

    # -- primitives ---------------------------------------------------------
    def add_segs(self, p, segs: List[Seg], size: Optional[float] = None):
        for g in segs:
            if g.kind == "math":
                p._p.append(parse_xml(g.xml))
                continue
            r = p.add_run(g.text)
            r.bold, r.italic = g.b or None, g.it or None
            if g.tt:
                r.font.name = "Courier New"
            if g.sup:
                r.font.superscript = True
            if size:
                r.font.size = Pt(size)

    def para(self, latex: str, style: Optional[str] = None, align=None, indent=True, size=None):
        txt = latex.strip()
        if not txt:
            return None
        p = self.doc.add_paragraph(style=style)
        if align is not None:
            p.alignment = align
        else:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if indent and style is None:
            p.paragraph_format.first_line_indent = Inches(0.17)
        self.add_segs(p, self.il.convert(txt), size)
        return p

    def heading(self, kind: str, title: str):
        if kind == "section":
            self.sec += 1
            self.sub = self.subsub = 0
            num = (f"APPENDIX {LETTERS[self.sec - 1]}" if self.in_app else f"{ROMAN[self.sec - 1]}.")
            p = self.doc.add_heading(level=1)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            text = f"{num} " + "".join(g.text for g in self.il.convert(title) if g.kind == "text").upper()
            p.add_run(text)
        elif kind == "section*":
            p = self.doc.add_heading(level=1)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run("".join(g.text for g in self.il.convert(title) if g.kind == "text").upper())
        elif kind == "subsection":
            self.sub += 1
            self.subsub = 0
            p = self.doc.add_heading(level=2)
            p.add_run(f"{LETTERS[self.sub - 1]}. ")
            self.add_segs(p, self.il.convert(title))
        else:
            self.subsub += 1
            p = self.doc.add_heading(level=3)
            p.add_run(f"{self.subsub}) ")
            self.add_segs(p, self.il.convert(title))

    def display_math(self, body: str, numbered: bool):
        lines = split_math_lines(body) if numbered is not None else [body]
        for line in lines:
            num = None
            if numbered and "\\nonumber" not in line and "\\notag" not in line:
                self.eq += 1
                num = self.eq
            p = self.doc.add_paragraph()
            pf = p.paragraph_format
            pf.tab_stops.add_tab_stop(Inches(3.45), WD_TAB_ALIGNMENT.CENTER)
            pf.tab_stops.add_tab_stop(Inches(6.9), WD_TAB_ALIGNMENT.RIGHT)
            p.add_run("\t")
            try:
                p._p.append(parse_xml(omml(line.replace("&", ""))))
            except Exception:
                print("EQUATION FAILED:", repr(line))
                raise
            if num:
                p.add_run(f"\t({num})")

    def caption(self, kind: str, cap: str, number: str):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if kind == "table" else WD_ALIGN_PARAGRAPH.JUSTIFY
        lead = f"TABLE {number}. " if kind == "table" else f"Fig. {number}. "
        r = p.add_run(lead)
        r.bold = True
        r.font.size = Pt(8.5)
        self.add_segs(p, self.il.convert(cap), size=8.5)
        p.paragraph_format.keep_with_next = kind == "table"

    def table(self, tabular: str, wide: bool):
        body = re.sub(r"\\(toprule|bottomrule)", "", tabular)
        rows_src = re.split(r"\\\\", body)
        rows, header_done, hdr_idx = [], False, None
        for r in rows_src:
            if "\\midrule" in r:
                if hdr_idx is None and rows:
                    hdr_idx = len(rows) - 1
                r = r.replace("\\midrule", "")
            if not r.strip():
                continue
            cells = re.split(r"(?<!\\)&", r)
            rows.append([c.strip() for c in cells])
        if not rows:
            return
        ncol = max(len(r) for r in rows)
        t = self.doc.add_table(rows=len(rows), cols=ncol)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, r in enumerate(rows):
            for j in range(ncol):
                cell = t.cell(i, j)
                cell.text = ""
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                txt = r[j] if j < len(r) else ""
                segs = self.il.convert(txt, b=(hdr_idx is not None and i <= hdr_idx))
                self.add_segs(p, segs, size=8)
        self.doc.add_paragraph()

    def figure(self, body: str, wide: bool):
        m = re.search(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}", body)
        cap_m = re.search(r"\\caption\{", body)
        cap, _ = brace_arg(body, cap_m.end() - 1) if cap_m else ("", 0)
        labels = re.findall(r"\\label\{([^}]+)\}", body)
        num = self.labels.get(labels[0], "?") if labels else "?"
        if m:
            src = os.path.join(self.root, m.group(2))
            png = self.pdf_to_png(src)
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(png, width=Inches(6.8 if wide else 3.4))
        self.caption("figure", cap, num)

    def pdf_to_png(self, path: str) -> str:
        import fitz
        if not path.lower().endswith(".pdf"):
            return path
        out = os.path.join(self.figdir, os.path.splitext(os.path.basename(path))[0] + ".png")
        d = fitz.open(path)
        d[0].get_pixmap(dpi=300).save(out)
        return out

    def algorithm(self, body: str):
        cap_m = re.search(r"\\caption\{", body)
        cap, _ = brace_arg(body, cap_m.end() - 1)
        num = self.labels.get(re.findall(r"\\label\{([^}]+)\}", body)[0], "?")
        p = self.doc.add_paragraph()
        r = p.add_run(f"Algorithm {num}: ")
        r.bold = True
        self.add_segs(p, self.il.convert(cap))
        p.paragraph_format.border_bottom = None
        inner = body[body.find("\\begin{algorithmic}"):body.find("\\end{algorithmic}")]
        inner = re.sub(r"\\begin\{algorithmic\}(\[[^\]]*\])?", "", inner)
        toks = re.split(r"(\\(?:REQUIRE|ENSURE|STATE|IF|ELSIF|ELSE|ENDIF|WHILE|ENDWHILE|FOR|ENDFOR|RETURN)\b)", inner)
        n, depth = 0, 0
        k = 1
        while k < len(toks):
            cmd, rest = toks[k][1:], toks[k + 1]
            k += 2
            arg = ""
            rest = rest.strip()
            if cmd in ("IF", "ELSIF", "WHILE", "FOR") and rest.startswith("{"):
                arg, j = brace_arg(rest, 0)
                rest = rest[j:].strip()
            if cmd in ("ENDIF", "ENDWHILE", "ENDFOR", "ELSE", "ELSIF"):
                depth = max(0, depth - 1)
            line = {"REQUIRE": "Require: ", "ENSURE": "Ensure: ", "STATE": "", "RETURN": "return ",
                    "IF": "if ", "ELSIF": "else if ", "ELSE": "else", "ENDIF": "end if",
                    "WHILE": "while ", "ENDWHILE": "end while", "FOR": "for ", "ENDFOR": "end for"}[cmd]
            text = arg + (" then" if cmd in ("IF", "ELSIF") else " do" if cmd in ("WHILE", "FOR") else "")
            n += 1
            q = self.doc.add_paragraph()
            q.paragraph_format.left_indent = Inches(0.25 + 0.25 * depth)
            q.paragraph_format.space_after = Pt(0)
            q.add_run(f"{n}: " if cmd not in ("REQUIRE", "ENSURE") else "").font.size = Pt(9)
            rr = q.add_run(line)
            rr.bold = cmd not in ("STATE",)
            rr.font.size = Pt(9)
            self.add_segs(q, self.il.convert(text.replace("\\AND", " and ") + " " + rest), size=9)
            if cmd in ("IF", "ELSIF", "ELSE", "WHILE", "FOR"):
                depth += 1
        self.doc.add_paragraph()

    def itemize(self, body: str):
        for item in re.split(r"\\item\b", body)[1:]:
            p = self.doc.add_paragraph(style="List Bullet")
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            self.add_segs(p, self.il.convert(item.strip()))

    # -- body walker ---------------------------------------------------------
    def body(self, s: str):
        pat = re.compile(r"\\(section\*?|subsection|subsubsection)\{|\\appendices|"
                         r"\\begin\{([A-Za-z*]+)\}(\[(?:[^\]\[]|\[[^\]]*\])*\])?")
        i = 0
        while True:
            m = pat.search(s, i)
            chunk = s[i:m.start()] if m else s[i:]
            for par in re.split(r"\n\s*\n", chunk):
                if par.strip():
                    self.para(par)
            if not m:
                break
            if m.group(0) == "\\appendices":
                self.in_app, self.sec = True, 0
                i = m.end()
                continue
            if m.group(1):
                title, j = brace_arg(s, m.end() - 1)
                self.heading(m.group(1), title)
                i = j
                continue
            env = m.group(2)
            end = s.find("\\end{" + env + "}", m.end())
            inner = s[m.end():end]
            i = end + len(env) + 6
            if env.startswith("multline"):
                inner = inner.replace("\\\\", " ")
            if env in ("equation", "align", "multline", "gather"):
                self.display_math(inner, True)
            elif env.endswith("*") and env.rstrip("*") in ("equation", "align", "multline"):
                self.display_math(inner, False)
            elif env.startswith("table"):
                cap_m = re.search(r"\\caption\{", inner)
                cap, _ = brace_arg(inner, cap_m.end() - 1) if cap_m else ("", 0)
                labels = re.findall(r"\\label\{([^}]+)\}", inner)
                self.caption("table", cap, self.labels.get(labels[0], "?") if labels else "?")
                tm = re.search(r"\\begin\{tabular\}", inner)
                if tm:
                    _, j = brace_arg(inner, tm.end())
                    te = inner.find("\\end{tabular}", j)
                    self.table(inner[j:te], env.endswith("*"))
            elif env.startswith("figure"):
                self.figure(inner, env.endswith("*"))
            elif env == "algorithm":
                self.algorithm(inner)
            elif env in ("itemize", "enumerate"):
                self.itemize(inner)
            elif env == "thebibliography":
                self.bibliography(inner)
            elif env == "IEEEbiography":
                full = s[m.start():end]
                bm = re.match(r"\\begin\{IEEEbiography\}\[\{\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}\}\]\{([^}]*)\}(.*)",
                              full, re.S)
                photo, name, text = bm.group(1), bm.group(2), bm.group(3)
                t = self.doc.add_table(rows=1, cols=2)
                t.autofit = False
                c0, c1 = t.cell(0, 0), t.cell(0, 1)
                c0.width, c1.width = Inches(1.15), Inches(5.75)
                c0.paragraphs[0].add_run().add_picture(os.path.join(self.root, photo), width=Inches(1.0))
                p = c1.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                r = p.add_run(name.upper() + " ")
                r.bold = True
                self.add_segs(p, self.il.convert(text.strip()))
            elif env == "IEEEbiographynophoto":
                name, j = brace_arg(inner, inner.find("{"))
                p = self.doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                r = p.add_run(name + " ")
                r.bold = True
                self.add_segs(p, self.il.convert(inner[j:].strip()))
            elif env == "abstract":
                pass
            else:
                for par in re.split(r"\n\s*\n", inner):
                    self.para(par)

    def bibliography(self, inner: str):
        p = self.doc.add_heading(level=1)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run("REFERENCES")
        items = re.split(r"\\bibitem\{[^}]+\}", inner)[1:]
        for n, it in enumerate(items, 1):
            q = self.doc.add_paragraph()
            q.paragraph_format.left_indent = Inches(0.3)
            q.paragraph_format.first_line_indent = Inches(-0.3)
            q.paragraph_format.space_after = Pt(1)
            q.alignment = WD_ALIGN_PARAGRAPH.LEFT
            q.add_run(f"[{n}]\t").font.size = Pt(8.5)
            self.add_segs(q, self.il.convert(it.strip()), size=8.5)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../IEEE_Access/manuscript")
    ap.add_argument("--out", default="../IEEE_Access/SH-GDSM-L_manuscript.docx")
    a = ap.parse_args()
    src = read_tex(a.root, "main")
    macros = collect_macros(src)
    pre, doc = src.split("\\begin{document}", 1)
    doc = doc.split("\\end{document}", 1)[0]
    doc = expand_macros(doc, macros)
    labels = number_labels(doc)
    bib = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", doc, re.S).group(0)
    cites = {k: n for n, k in enumerate(re.findall(r"\\bibitem\{([^}]+)\}", bib), 1)}
    figdir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "docx_figures")
    os.makedirs(figdir, exist_ok=True)
    W = Writer(a.root, Inline(labels, cites), labels, figdir)

    # --- front matter
    title, _ = brace_arg(doc, doc.index("\\title{") + 6)
    p = W.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run("".join(g.text for g in W.il.convert(title)))
    r.bold, r.font.size = True, Pt(18)
    author, _ = brace_arg(doc, doc.index("\\author{") + 7)
    author = re.sub(r"\\uppercase\{([^}]*)\}", lambda m: m.group(1).upper(), author)
    author = re.sub(r"\\authorrefmark\{(\d)\}", r"\\textsuperscript{\1}", author)
    W.para(author, align=WD_ALIGN_PARAGRAPH.LEFT, indent=False)
    for m in re.finditer(r"\\address\[(\d)\]\{", doc):
        addr, _ = brace_arg(doc, m.end() - 1)
        W.para(f"\\textsuperscript{{{m.group(1)}}}{addr}", align=WD_ALIGN_PARAGRAPH.LEFT, indent=False, size=8.5)
    corr, _ = brace_arg(doc, doc.index("\\corresp{") + 8)
    W.para(corr, align=WD_ALIGN_PARAGRAPH.LEFT, indent=False, size=8.5)
    tfn, _ = brace_arg(doc, doc.index("\\tfootnote{") + 10)
    W.para(tfn, align=WD_ALIGN_PARAGRAPH.LEFT, indent=False, size=8.5)
    abs_ = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", doc, re.S).group(1)
    p = W.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run("ABSTRACT ")
    r.bold = True
    W.add_segs(p, W.il.convert(abs_.strip()))
    kw = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", doc, re.S).group(1)
    p = W.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run("INDEX TERMS ")
    r.bold = True
    W.add_segs(p, W.il.convert(kw.strip()))

    body = doc[doc.index("\\maketitle") + len("\\maketitle"):]
    body = body.replace("\\EOD", "")
    W.body(body)
    W.doc.save(a.out)
    missing = sorted({l for l in re.findall(r"\\(?:eq)?ref\{([^}]+)\}", doc) if l not in labels})
    print("wrote", a.out, "| equations:", W.eq, "| labels:", len(labels), "| unresolved refs:", missing)


if __name__ == "__main__":
    main()
