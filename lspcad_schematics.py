#!/usr/bin/env python3
"""Draw schematics of the LspCAD filter files in ``Test av olika nya
filter`` into one PDF (``filter_schematics.pdf`` in the same folder).

How it works
------------
LspCAD 6 stores schematics as readable text.  Every ``Component N``
block has a grid anchor ``pt[0]`` plus relative points; the electrical
pins are ``pt0 + rot90(pt_k)`` (one 90 deg counter-clockwise step per
``rot`` unit).  Component types:  1 component group, 3 wire,
20 bent wire, 21 resistor, 22 inductor, 23 capacitor, 26 driver,
40 generator.

Two terminals belong to the same net when they

* coincide exactly,
* sit on another symbol's lead (the side of the body, i.e. of the
  anchor pt[0], decides which pin they join),
* sit on / very near a wire (tap tol 2.6, stub tol 4.2 grid units).

LspCAD draws a 4 unit long ground stub below every shunt leg; those
stubs tie the shunt legs and the driver's low pin to the common rail.

From the netlist every branch (Bass / Mid / Treble) is re-laid out as a
clean ladder: idea generator left, series elements on the top line,
parallel combinations below it, shunt legs dropping to the rail, and
the driver at the right with its polarity annotation.

Usage:
    python3 lspcad_schematics.py            # writes the PDF
    python3 lspcad_schematics.py --debug    # also print the nets
"""

import math
import os
import re
import sys

from simple_pdf import PDF

HERE = os.path.dirname(os.path.abspath(__file__))
FILTER_DIR = os.path.join(HERE, "Test av olika nya filter")
OUTPUT_NAME = "filter_schematics.pdf"

TYPE = {3: "wire", 20: "wire", 21: "R", 22: "L", 23: "C",
        26: "driver", 40: "gen"}
TAP_TOL = 2.6
STUB_TOL = 4.2
BRANCH_ORDER = ["Bass", "Mid", "Treble"]


# ------------------------------------------------------------------ parse

def rot90(p, n):
    x, y = p
    for _ in range(n % 4):
        x, y = -y, x
    return x, y


def parse_lsp(path):
    text = open(path, encoding="iso-8859-1").read()
    comps = []
    starts = [(int(m.group(1)), m.start(), m.end() - 1)
              for m in re.finditer(r"Component (\d+)\r?\n\s*\{", text)]
    for ctype, mstart, brace in starts:
        depth, i = 0, brace
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        block = text[mstart:i + 1]
        num = re.search(r"\bnum=(\d+)", block)
        gname = re.search(r"gname=\r?\n(.*?)\r?\n\s*<##", block, re.S)
        nick = re.search(r"nickName=\r?\n(.*?)\r?\n\s*<##", block, re.S)
        val = re.search(r"\n\s*val=([0-9.eE+-]+)", block)
        rm = re.search(r"\n\s+rot=(\d+)", block)
        inv = re.search(r"Driver\r?\n\s*\{\r?\n\s*invert=(\d)", block) \
            if ctype == 26 else None
        comps.append(dict(
            type=ctype,
            pts={k: (int(a), int(b)) for k, a, b in
                 re.findall(r"pt\[(\w+)\]=(-?\d+) (-?\d+)", block)},
            num=int(num.group(1)) if num else -1,
            name=(gname.group(1).strip() if gname else "") or
                 (nick.group(1).strip() if nick else ""),
            val=float(val.group(1)) if val else None,
            rot=int(rm.group(1)) if rm else 0,
            invert=int(inv.group(1)) if inv else 0))
    return comps


def pins(c):
    """Electrical terminals: [(tag, x, y), ...] in page coordinates."""
    base = c["pts"].get("0")
    if base is None:
        return []
    if c["type"] in (3, 20):                     # wires: relative as well
        return [(k, base[0] + (0 if k == "0" else x),
                 base[1] + (0 if k == "0" else y))
                for k, (x, y) in sorted(c["pts"].items()) if k[0] != "T"]
    out = [("0", base[0], base[1])]
    for k, (x, y) in sorted(c["pts"].items()):
        if k[0] == "T" or k == "0":
            continue
        rx, ry = rot90((x, y), c["rot"])
        out.append((k, base[0] + rx, base[1] + ry))
    return out


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class DSU:
    def __init__(self):
        self.p = {}

    def add(self, k):
        self.p.setdefault(k, k)
        return k

    def find(self, k):
        self.p.setdefault(k, k)
        while self.p[k] != k:
            self.p[k] = self.p[self.p[k]]
            k = self.p[k]
        return k

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def qk(x, y):
    return (round(x, 1), round(y, 1))


def build_nets(comps):
    """Return (dsu, circ, wires, notes):  notes log evidence for review."""
    dsu = DSU()
    circ = [c for c in comps if c["type"] in (21, 22, 23, 26, 40)]
    wires = [c for c in comps if c["type"] in (3, 20)]
    notes, points = [], []

    for c in circ:
        c["_sites"] = {}
        for tag, x, y in pins(c):
            k = dsu.add(qk(x, y))
            c["_sites"][tag] = k
            if tag != "0":                 # anchors are not electrical
                points.append((k, x, y, c))
    wire_keys = []
    for w in wires:
        wk = []
        for tag, x, y in pins(w):
            k = dsu.add(qk(x, y))
            wk.append(k)
            points.append((k, x, y, w))
        wire_keys.append(wk)
    for w, wk in zip(wires, wire_keys):
        w["_verts"] = [(x, y) for _, x, y in pins(w)]
        for a, b in zip(wk, wk[1:]):       # polyline -> one conductor
            dsu.union(a, b)

    common = dsu.add("COMMON")
    for w, wk in zip(wires, wire_keys):
        xs = [p[0] for p in w["_verts"]]
        ys = [p[1] for p in w["_verts"]]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        if span <= 5.0:            # 4-unit ground stub (or stub cluster)
            dsu.union(common, wk[0])

    # near coincidence (4 unit ground stubs, generator lead spans, ...)
    for i in range(len(points)):
        ki, xi, yi, ci = points[i]
        for j in range(i + 1, len(points)):
            kj, xj, yj, cj = points[j]
            if ki == kj or ci is cj:
                continue
            d = math.hypot(xi - xj, yi - yj)
            if 0 < d <= STUB_TOL:
                dsu.union(ki, kj)
                notes.append(f"near: {ci['name'] or TYPE[ci['type']]}@{xi},{yi} "
                             f"- {cj['name'] or TYPE[cj['type']]}@{xj},{yj} "
                             f"(d={d:.1f})")

    # terminals lying ON a wire segment
    for c in circ:
        c["_pinxy"] = {tag: (x, y) for tag, x, y in pins(c)}
        for tag in ("1", "2"):
            if tag not in c["_sites"]:
                continue
            x, y = c["_pinxy"][tag]
            for w, wk in zip(wires, wire_keys):
                for a, b in zip(w["_verts"], w["_verts"][1:]):
                    if seg_dist(x, y, a[0], a[1], b[0], b[1]) <= TAP_TOL:
                        dsu.union(c["_sites"][tag], wk[0])
                        notes.append(f"tap: {c['name'] or TYPE[c['type']]}"
                                     f"{tag}@{x},{y} on wire {w['num']}")

    # terminals lying ON another symbol's lead; the side relative to the
    # anchor pt[0] decides which pin of that lead they join
    for c in circ:
        if c["type"] not in (21, 22, 23):
            continue
        for tag in ("1", "2"):
            if tag not in c["_sites"]:
                continue
            x, y = c["_pinxy"][tag]
            for cj in circ:
                if cj is c or cj["type"] not in (21, 22, 23):
                    continue
                pj = [(t, px, py) for t, px, py in pins(cj) if t != "0"]
                if len(pj) < 2:
                    continue
                (t1, x1, y1), (t2, x2, y2) = pj[0], pj[1]
                if seg_dist(x, y, x1, y1, x2, y2) > TAP_TOL:
                    continue
                ax, ay = cj["pts"]["0"]
                side = (x - ax) * (x2 - x1) + (y - ay) * (y2 - y1)
                ct = t2 if side > 0 else t1
                if ct in cj["_sites"]:
                    dsu.union(c["_sites"][tag], cj["_sites"][ct])
                    notes.append(f"tap: {c['name'] or TYPE[c['type']]}{tag}"
                                 f"@{x},{y} on {cj['name'] or TYPE[cj['type']]}"
                                 f" lead side {ct} (pin {cj['num']})")

    for c in circ:
        c["sites"] = {t: dsu.find(k) for t, k in c["_sites"].items()}
        del c["_sites"], c["_pinxy"]
    return dsu, circ, wires, notes


# ------------------------------------------------------------- branching

def fmt_val(c):
    t, v = TYPE.get(c["type"]), c["val"]
    if v is None:
        return ""
    if t == "R":
        return f"{v:g} \u03a9" if v < 1000 else f"{v/1000:g} k\u03a9"
    if t == "L":
        return f"{v*1e3:.3g} mH"
    if t == "C":
        return f"{v*1e6:.3g} \u00b5F"
    return ""


class El:
    """Passive two-terminal element between two nets."""

    def __init__(self, c):
        self.comp = c
        self.kind = TYPE[c["type"]]
        self.name = c["name"]
        self.value = fmt_val(c)
        self.na = c["sites"]["1"]
        self.nb = c["sites"]["2"]
        self.assumed = False        # end whose attachment is uncertain


def build_branch(driver, els, dsu):
    """Walk the netlist from the driver back to the input.

    Returns dict:  common, hot, items (reversed: input side first), where
    items alternate (node_id, shunt-legs) and ("bundle", [El]).
    """
    common = driver["sites"]["2"]
    node = driver["sites"]["1"]
    hot = node
    unused = set(els)
    items = []

    def shunt_paths(start):
        found = []

        def dfs(n, path, visited):
            for el in unused:
                if el in path:
                    continue
                if el.na == n and el.nb not in visited:
                    nxt = el.nb
                elif el.nb == n and el.na not in visited:
                    nxt = el.na
                else:
                    continue
                if nxt == common:
                    found.append(path + [el])
                else:
                    dfs(nxt, path + [el], visited | {nxt})
        dfs(start, [], {start})

        def is_true(chain):
            """Interior nodes of a shunt chain must be fully covered by
            the chain itself (no ladder element passing through)."""
            touched = {chain[0].na, chain[0].nb}
            for el in chain:
                touched |= {el.nb, el.na}
            interior = set(touched)
            interior.discard(start)
            interior.discard(common)
            for r in interior:
                for el in els:
                    if el in chain:
                        continue
                    if el.na == r or el.nb == r:
                        return False
            return True

        return [p for p in found if is_true(p)]

    while True:
        legs = sorted(shunt_paths(node),
                      key=lambda p: (len(p), min(el.kind for el in p)))
        for p in legs:
            for el in p:
                unused.discard(el)
        passages = {}
        for el in list(unused):
            if el.na == node and el.nb != node:
                passages.setdefault(el.nb, []).append(el)
            elif el.nb == node and el.na != node:
                passages.setdefault(el.na, []).append(el)
        if not passages:
            items.append((node, legs))
            break
        if len(passes := passages) > 1:
            tgt = max(passes, key=lambda k: len(passes[k]))
        else:
            tgt = next(iter(passes))
        bundle = passages[tgt]
        for el in bundle:
            unused.discard(el)
        items.append((node, legs))
        items.append(("bundle", bundle))
        node = tgt
    items.reverse()
    return {"common": common, "hot": hot, "items": items,
            "orphans": list(unused)}


def attach_orphans(branch, band_lo, band_hi):
    """Chains of leftover elements that reach common at one end and float
    at the other are assumed to hang across the driver (typical Zobel).
    Only orphan components whose anchor positions sit within the branch
    band are considered."""
    orph = [e for e in branch["orphans"]
            if band_lo <= e.comp["pts"]["0"][1] <= band_hi]
    if not orph:
        return
    by_root = {}
    for el in orph:
        by_root.setdefault(el.na, []).append(el)
        by_root.setdefault(el.nb, []).append(el)
    # connected components over shared nets
    seen, groups = set(), []
    for el in orph:
        if el in seen:
            continue
        stack, group = [el], []
        seen.add(el)
        while stack:
            cur = stack.pop()
            group.append(cur)
            for r in (cur.na, cur.nb):
                for f in by_root.get(r, []):
                    if f not in seen:
                        seen.add(f)
                        stack.append(f)
        groups.append(group)
    for group in groups:
        adj = {}
        for el in group:
            adj.setdefault(el.na, []).append(el)
            adj.setdefault(el.nb, []).append(el)
        ends = [r for r, elsx in adj.items() if len(elsx) == 1]
        if ends.count(branch["common"]) != 1 or len(ends) != 2:
            continue
        free = [r for r in ends if r != branch["common"]]
        chain, cur = [], free[0]
        ok = True
        while cur != branch["common"]:
            nxt = None
            for el in adj.get(cur, []):
                if el not in chain:
                    nxt = el
                    break
            if nxt is None:
                ok = False
                break
            chain.append(nxt)
            cur = nxt.nb if nxt.na == cur else nxt.na
        if ok and cur == branch["common"]:
            for el in chain:
                el.assumed = True
                branch["orphans"].remove(el)
            for it in reversed(branch["items"]):
                if it[0] != "bundle":
                    it[1].append(chain)
                    break
            break


def branch_chain_text(branch):
    """One line description:  series chain with shunts in {braces}."""
    out = []
    for it in branch["items"]:
        if it[0] == "bundle":
            els = it[1]
            if len(els) == 1:
                out.append(_elname(els[0]))
            else:
                out.append("(" + " \u2225 ".join(_elname(e) for e in els) + ")")
        else:
            legs = it[1]
            for leg in legs:
                out.append("{" + "+".join(_elname(e) for e in leg) + "\u2192com}")
    return "\u2014".join(out) if out else "(direct)"


def _elname(e):
    base = f"{e.value}" if e.value else TYPE[e.kind]
    return f"{e.name} {base}".strip() if e.name else base


def read_filter_file(path):
    comps = parse_lsp(path)
    dsu, circ, wires, notes = build_nets(comps)
    els = [El(c) for c in circ if c["type"] in (21, 22, 23)]
    gens = [c for c in circ if c["type"] == 40]
    drivers = [c for c in circ if c["type"] == 26]
    branches = []
    for d in drivers:
        b = build_branch(d, els, dsu)
        # leftover chains within this driver's vertical band are assumed
        # to hang across it (this also picks up the bass Zobel)
        y = d["pts"]["0"][1]
        lo, hi = (12, 50) if y < 50 else (50, 85) if y < 85 else (85, 130)
        attach_orphans(b, lo, hi)
        gen = None
        # generator with the largest name overlap / nearest input node
        cand = [g for g in gens if g["name"].lower().endswith(d["name"].lower())]
        if cand:
            gen = cand[0]
        b["driver"], b["gen"] = d, gen
        branches.append(b)
    return {"comps": comps, "notes": notes, "branches": branches, "path": path}


# ------------------------------------------------------------- rendering

SYM_W = {"R": 46, "L": 56, "C": 36}
LEG_H = {"R": 40, "L": 40, "C": 20}     # keeps 3-element legs within 114
LEG_SPACE = 40          # horizontal pitch between fan legs
ROW_DY = 24             # vertical pitch of parallel rows under main line
MAIN_DY = 120           # main line height above branch origin
RAIL_DY = 8             # rail height above branch origin


def elw(e):
    return SYM_W[e.kind]


def draw_sym_h(pg, kind, x0, x1, y):
    """Horizontal element with leads between x0 and x1, wire at y."""
    if kind == "R":
        a, b, n = x0 + 7, x1 - 7, 6
        pts = [(a, y)]
        for i in range(1, n):
            pts.append((a + (b - a) * i / n, y + (6 if i % 2 else -6)))
        pts.append((b, y))
        pg.poly(pts + [(x0, y), (a, y)], 0.9)
        pg.poly(pts, 0.9)
        pg.line(b, y, x1, y, 0.9)
    elif kind == "L":
        span = x1 - x0
        r = (span - 20) / 8
        cx = x0 + 10
        for i in range(4):
            pg.arc(cx + 2 * r * i + r, y, r, 180, 0, 0.9)
        pg.line(x0, y, cx, y, 0.9)
        pg.line(cx + 8 * r, y, x1, y, 0.9)
    elif kind == "C":
        xc = (x0 + x1) / 2
        pg.line(x0, y, xc - 4, y, 0.9)
        pg.line(xc + 4, y, x1, y, 0.9)
        pg.line(xc - 4, y + 9, xc - 4, y - 9, 0.9)
        pg.line(xc + 4, y + 9, xc + 4, y - 9, 0.9)


def draw_sym_v(pg, kind, y0, y1, x):
    """Vertical element between y0 (top) and y1 (bottom), wire at x."""
    if kind == "R":
        a, b, n = y0 - 9, y1 + 9, 6
        pts = [(x, a)]
        for i in range(1, n):
            pts.append((x + (5 if i % 2 else -5), a + (b - a) * i / n))
        pts.append((x, b))
        pg.poly(pts, 0.9)
        pg.line(x, y0, x, a, 0.9)
        pg.line(x, b, x, y1, 0.9)
    elif kind == "L":
        r = (y0 - y1) / 8
        cy = y1
        for i in range(4):
            pg.arc(x, cy + 2 * r * i + r, r, 270, 90, 0.9)
        pg.line(x, y0, x, cy + 8 * r, 0.9)
        pg.line(x, cy, x, y1, 0.9)
    elif kind == "C":
        yc = (y0 + y1) / 2
        pg.line(x, y0, x, yc + 4, 0.9)
        pg.line(x, yc - 4, x, y1, 0.9)
        pg.line(x - 9, yc + 4, x + 9, yc + 4, 0.9)
        pg.line(x - 9, yc - 4, x + 9, yc - 4, 0.9)


def draw_leg(pg, chain, x, y_from, y_rail):
    """Vertical shunt leg, chain of symbols top-down; labels to the right."""
    ycur = y_from
    for e in chain:
        h = LEG_H[e.kind]
        y_mid = ycur - h / 2
        draw_sym_v(pg, e.kind, y_mid + h / 2, y_mid - h / 2, x)
        lbl = f"{e.name} {e.value}".strip()
        if e.assumed:
            lbl += " *"
        pg.text(x + 10, y_mid + 2, lbl, 7)
        ycur = ycur - h - 6
    pg.line(x, y_from, x, y_rail, 0.9)


def draw_bundle(pg, els, x, y):
    """Elements in parallel between two points of the main line."""
    k = len(els)
    innr = max(elw(e) for e in els) + 46
    if k == 1:
        e = els[0]
        x1 = x + 22 + elw(e) + 18
        pg.line(x, y, x + 22, y, 0.9)
        draw_sym_h(pg, e.kind, x + 22, x + 22 + elw(e), y)
        lbl = f"{e.name} {e.value}".strip()
        if e.assumed:
            lbl += " *"
        pg.text(x + 22 + elw(e) / 2, y + 13, lbl, 7.5, align="c")
        pg.line(x + 22 + elw(e), y, x1, y, 0.9)
        return x1
    ep = x + innr
    for j, e in enumerate(els):
        ry = y if j == 0 else y - ROW_DY * j
        if j == 0:
            x0, x1 = x + 22, x + 22 + elw(e)
            pg.line(x, y, x0, y, 0.9)
            draw_sym_h(pg, e.kind, x0, x1, y)
            pg.text((x0 + x1) / 2, y + 13,
                    f"{e.name} {e.value}".strip(), 7.5, align="c")
            pg.line(x1, y, ep, y, 0.9)
        else:
            x0, x1 = x + 26, x + 26 + elw(e)
            pg.line(x, ry, x0, ry, 0.9)
            draw_sym_h(pg, e.kind, x0, x1, ry)
            pg.text((x0 + x1) / 2, ry - 10,
                    f"{e.name} {e.value}".strip(), 7.5, align="c")
            pg.line(x1, ry, ep, ry, 0.9)
        pg.line(x, y, x, ry, 0.9)
        pg.line(ep, ry, ep, y, 0.9)
    pg.dot(x, y)
    pg.dot(ep, y)
    return ep


def draw_source(pg, x, y, gen):
    name = (gen["name"] if gen and gen["name"] else "generator")
    pg.circle(x, y, 11, 0.9)
    pts = [(x - 6 + 12 * i / 12, y + 3.2 * math.sin(2 * math.pi * i / 12))
           for i in range(13)]
    pg.poly(pts, 0.7)
    pg.text(x, y + 17, name, 7.5, align="c")
    pg.text(x, y - 17, "1 V", 7, align="c")
    pg.line(x + 11, y, x + 26, y, 0.9)     # hot lead out to the right
    pg.line(x, y - 11, x, y - 114, 0.9)    # return down to the rail


def draw_driver(pg, x, y, drv):
    name = drv["name"] or "driver"
    pg.circle(x, y, 16, 0.9)
    pg.poly([(x - 9, y + 6), (x + 4, y + 11), (x + 4, y - 11), (x - 9, y - 6)],
            0.9)
    pol = "reversed polarity" if drv["invert"] else "normal polarity"
    pg.text(x + 22, y + 9, f"{name} driver", 8, bold=True)
    pg.text(x + 22, y - 2, pol, 7)
    s_top = "\u2212" if drv["invert"] else "+"
    s_bot = "+" if drv["invert"] else "\u2212"
    pg.text(x - 16, y + 13, s_top, 9, bold=True)
    pg.text(x - 16, y - 13, s_bot, 9, bold=True)


def draw_branch(pg, ox, ymain, branch):
    """Draw one branch: main line at ymain, common rail at ymain-114."""
    rail = ymain - 114
    x = ox + 70
    draw_source(pg, x, ymain, branch.get("gen"))
    x += 26

    xend = x
    for it in branch["items"]:
        if it[0] == "bundle":
            xend = draw_bundle(pg, it[1], x, ymain)
            x = xend
        else:
            node, legs = it
            for i, leg in enumerate(legs):
                lx = x + 8 + i * LEG_SPACE
                pg.dot(lx, ymain)
                draw_leg(pg, leg, lx, ymain, rail)
            if any(e.assumed for leg in legs for e in leg):
                pg.text(x - 16, rail - 6, "* assumed (floating lead in file)",
                        6.5)
            x = x + 8 + len(legs) * LEG_SPACE
    # driver below the main line, then down to the rail
    drv = branch["driver"]
    dx = x + 28
    pg.line(xend, ymain, dx, ymain, 0.9)
    draw_driver(pg, dx, ymain - 24, drv)
    pg.line(dx, ymain - 40, dx, rail, 0.9)
    pg.line(ox + 70, rail, dx, rail, 1.1)
    pg.text(ox + 70, rail - 7, "common return", 6.5)
    return dx + 40


# ---------------------------------------------------------------- report

def make_cover(pg, files):
    pg.text(60, 540, "Crossover filter schematics", 22, bold=True)
    pg.text(60, 518, "Redrawn from LspCAD 6 GenFilter files in "
                     "\u201cTest av olika nya filter\u201d", 11)
    y = 480
    for i, f in enumerate(files):
        b = f["branches"]
        pg.text(60, y, f"Page {i + 2}:  {os.path.basename(f['path'])}",
                10.5, bold=True)
        y -= 15
        for br in sorted(b, key=lambda z: BRANCH_ORDER.index(
                z["driver"]["name"])):
            txt = branch_chain_text(br)
            lines = _wrap(txt, 96)
            pg.text(75, y, f"{br['driver']['name']}:", 9, bold=True)
            for ln in lines:
                pg.text(125, y, ln, 8)
                y -= 11
            y -= 3
        y -= 10
    y = min(y, 150)
    pg.text(60, y, "Notes", 11, bold=True)
    y -= 14
    for note in [
            "Topology is reconstructed from the coordinates stored in the "
            ".lsp files and redrawn; it reproduces LspCAD's own netlist, "
            "not its pixel layout.",
            "Component values are exactly as stored (3 decimal digits "
            "trimmed). Drivers named Bass / Mid / Treble.",
            "Mid and Treble are saved with inverted polarity in all three "
            "files; the schematics mark this on the driver symbol.",
            "Each branch is driven from its own ideal 1 V generator "
            "(one per driver in the files), drawn here at each branch "
            "input.",
            "\u201cKanske n\u00e5got.flt\u201d is a binary LspCAD curve "
            "export without schematic information and is therefore not "
            "included.",
            "In all three files the bass Zobel (6.8 \u03a9 + 33 \u00b5F) "
            "has one floating lead: its top terminal sits ~2.5 units from "
            "but not on any conductor. It is drawn attached to the bass "
            "hot node and marked \u201cassumed\u201d."]:
        for ln in _wrap(note, 108):
            pg.text(60, y, ln, 8)
            y -= 10.5
        y -= 2


def _wrap(s, n):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def filter_page(pg, fdata, title):
    pg.text(40, 566, title, 13, bold=True)
    pg.text(40, 553, os.path.basename(fdata["path"]) +
            "  \u2014  redrawn netlist (LspCAD 6 GenFilter)", 8.5)
    blocks = sorted(fdata["branches"],
                    key=lambda z: BRANCH_ORDER.index(z["driver"]["name"]))
    ymains = [520, 346, 174]
    for i, b in enumerate(blocks[:3]):
        ymain = ymains[i]
        rail = ymain - 114
        name = b["driver"]["name"] or "?"
        gen = b.get("gen")
        gen_txt = f" \u2014 {gen['name']}" if gen and gen["name"] else ""
        pg.text(40, rail - 22, f"{name} branch{gen_txt}", 10.5, bold=True)
        pg.text(40, rail - 34, branch_chain_text(b), 7.5)
        draw_branch(pg, 40, ymain, b)
        # separator between this block and the next one
        if i + 1 < len(blocks):
            pg.line(30, rail - 38, 812, rail - 38, 0.4)


def main():
    debug = "--debug" in sys.argv
    files = []
    for fname in sorted(os.listdir(FILTER_DIR)):
        if fname.lower().endswith(".lsp"):
            files.append(read_filter_file(os.path.join(FILTER_DIR, fname)))
    if debug:
        for f in files:
            print("=" * 100)
            print(os.path.basename(f["path"]))
            for n in f["notes"]:
                print("   ", n)
            for b in sorted(f["branches"],
                            key=lambda z: BRANCH_ORDER.index(
                                z["driver"]["name"])):
                drv = b["driver"]
                print(f"  {drv['name']:7s} invert={drv['invert']}: "
                      f"{branch_chain_text(b)}")
                if b["orphans"]:
                    print(f"      orphans: "
                          f"{[ _elname(e) for e in b['orphans']]}")

    pdf = PDF("Filter schematics")
    pg = pdf.page()
    make_cover(pg, files)
    for i, f in enumerate(files):
        filter_page(pdf.page(), f, f"Optimized filter {i + 1}")
    out = os.path.join(FILTER_DIR, OUTPUT_NAME)
    pdf.write(out)
    print("written:", out)


if __name__ == "__main__":
    main()
