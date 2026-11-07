#!/usr/bin/env python3
"""Minimal dependency-free PDF writer.

Vector primitives (lines, polylines, arcs, circles, rectangles) and text
in Helvetica / Helvetica-Bold with WinAnsiEncoding (so µ, Ω, ä, ö work).
Streams are Flate compressed using the stdlib zlib module.
"""

import math
import zlib


def _esc(s):
    out = []
    for ch in str(s):
        if ch in "()\\":
            out.append("\\" + ch)
        else:
            o = ord(ch)
            if o > 255:
                cp = {"œ": "\x9c", "–": "\x96", "—": "\x97",
                      "‘": "\x91", "’": "\x92", "“": "\x93",
                      "”": "\x94", "…": "\x85", "€": "\x80"}.get(ch)
                out.append(cp or "?")
            else:
                out.append(ch)
    return "".join(out).encode("cp1252", "replace").decode("cp1252")


class Page:
    def __init__(self, w=842, h=595):
        self.w, self.h = w, h
        self.ops = []

    # ------------------------------------------------------------- text
    def text(self, x, y, s, size=10, bold=False, align="l"):
        if align != "l":
            w = self.text_width(s, size, bold)
            x -= w / 2 if align == "c" else w
        self.ops.append("BT /%s %.2f Tf %.2f %.2f Td (%s) Tj ET" % (
            "F2" if bold else "F1", size, x, y, _esc(s)))

    def text_width(self, s, size, bold=False):
        # rough Helvetica metrics (enough for centring labels)
        narrow = "iljtf.,:;'|!()[]{}1 "
        wide = "mwMW"
        w = sum(size * (0.62 if c in wide else
                        0.34 if c in narrow else 0.50) for c in str(s))
        return w * (1.04 if bold else 1.0)

    # ------------------------------------------------------------- vectors
    def line(self, x0, y0, x1, y1, w=1.0):
        self.ops.append("%.2f w %.2f %.2f m %.2f %.2f l S"
                        % (w, x0, y0, x1, y1))

    def poly(self, pts, w=1.0, close=False, fill=False):
        if len(pts) < 2:
            return
        s = "%.2f w %.2f %.2f m" % (w, pts[0][0], pts[0][1])
        for p in pts[1:]:
            s += " %.2f %.2f l" % (p[0], p[1])
        s += " h" if close else ""
        s += " f" if fill else " S"
        self.ops.append(s)

    def dot(self, x, y, r=2.4):
        self.ops.append("%.2f %.2f %.2f %.2f re f" % (x - r, y - r, 2 * r, 2 * r))

    def arc(self, cx, cy, r, a0, a1, w=1.0, n=20):
        pts = [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
                cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
               for i in range(n + 1)]
        self.poly(pts, w)

    def circle(self, cx, cy, r, w=1.0):
        self.arc(cx, cy, r, 0, 360, w)

    def rect(self, x, y, w, h, lw=1.0):
        self.ops.append("%.2f w %.2f %.2f %.2f %.2f re S" % (lw, x, y, w, h))


class PDF:
    def __init__(self, title=""):
        self.pages = []
        self.title = title

    def page(self, w=842, h=595):
        p = Page(w, h)
        self.pages.append(p)
        return p

    def write(self, path):
        objs = {}

        def add(body):
            n = len(objs) + 1
            objs[n] = body
            return n

        fh = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
                 "/Encoding /WinAnsiEncoding >>")
        fb = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
                 "/Encoding /WinAnsiEncoding >>")
        pages_id = add("PLACEHOLDER-PAGES")
        page_ids = []
        for pg in self.pages:
            stream = " ".join(pg.ops)
            comp = zlib.compress(stream.encode("latin-1", "replace"))
            cid = add(("STREAM", comp))
            pid = add(f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox "
                      f"[0 0 {pg.w:.0f} {pg.h:.0f}] /Contents {cid} 0 R "
                      f"/Resources << /Font << /F1 {fh} 0 R /F2 {fb} 0 R >> "
                      ">> >>")
            page_ids.append(pid)
        objs[pages_id] = (f"<< /Type /Pages /Count {len(page_ids)} /Kids ["
                          + " ".join(f"{p} 0 R" for p in page_ids) + "] >>")
        info_id = add(f"<< /Title ({_esc(self.title)}) /Producer "
                      "(lspcad_schematics.py) >>")
        catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

        out = bytearray(b"%PDF-1.4\n%\xc7\xec\x8f\xa2\n")
        offsets = {}
        nobjs = len(objs)
        for n in range(1, nobjs + 1):
            offsets[n] = len(out)
            body = objs[n]
            if isinstance(body, tuple) and body[0] == "STREAM":
                data = body[1]
                out += (f"{n} 0 obj\n<< /Length {len(data)} /Filter "
                        "/FlateDecode >>\nstream\n").encode("latin-1")
                out += data + b"\nendstream\nendobj\n"
            else:
                out += (f"{n} 0 obj\n{body}\nendobj\n").encode("latin-1")
        xref = len(out)
        out += (f"xref\n0 {nobjs + 1}\n0000000000 65535 f \n").encode()
        for n in range(1, nobjs + 1):
            out += ("%010d 00000 n \n" % offsets[n]).encode()
        out += (f"trailer\n<< /Size {nobjs + 1} /Root {catalog_id} 0 R "
                f"/Info {info_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n").encode()
        with open(path, "wb") as f:
            f.write(out)
        return path
