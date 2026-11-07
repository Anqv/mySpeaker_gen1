#!/usr/bin/env python3
"""Debug rasterizer: same API as simple_pdf.Page but writes PNG/PPM so the
layout can be checked without a PDF viewer. Pure stdlib."""

import math
import struct
import zlib

BLACK = (0, 0, 0)


class Canvas:
    def __init__(self, w=842, h=595, scale=2):
        self.w, self.h, self.s = w, h, scale
        self.buf = bytearray([255]) * (w * scale * h * scale * 3)

    # helpers -------------------------------------------------------------
    def _px(self, x, y):
        s = self.s
        if 0 <= x < self.w * s and 0 <= y < self.h * s:
            row = self.h * s - 1 - int(y)     # PDF y is up, PNG is down
            i = (row * self.w * s + int(x)) * 3
            self.buf[i:i + 3] = bytes(BLACK)

    def _line(self, x0, y0, x1, y1, w):
        n = int(max(abs(x1 - x0), abs(y1 - y0), 1) * 2)
        for i in range(n + 1):
            x = x0 + (x1 - x0) * i / n
            y = y0 + (y1 - y0) * i / n
            s = self.s
            for dy in range(int(w * s) + 1):
                for dx in range(int(w * s) + 1):
                    self._px(x * s + dx - (w * s) / 2, y * s + dy - (w * s) / 2)

    # API mirroring simple_pdf.Page --------------------------------------
    def text(self, x, y, s, size=10, bold=False, align="l"):
        if align != "l":
            w = self.text_width(s, size, bold)
            x -= w / 2 if align == "c" else w
        # schematic text: small line under the baseline ink
        scale = size * self.s * 0.12
        for i in range(int(len(str(s)) * size * 0.5)):
            self._px(x * self.s + i * scale, y * self.s, )

    def line(self, x0, y0, x1, y1, w=1.0):
        self._line(x0, y0, x1, y1, w)

    def poly(self, pts, w=1.0, close=False, fill=False):
        for a, b in zip(pts, pts[1:]):
            self._line(a[0], a[1], b[0], b[1], w)
        if close and len(pts) > 2:
            self._line(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1], w)

    def dot(self, x, y, r=2.4):
        self._line(x, y, x, y, r)

    def arc(self, cx, cy, r, a0, a1, w=1.0, n=28):
        pts = [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
                cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
               for i in range(n + 1)]
        self.poly(pts, w)

    def circle(self, cx, cy, r, w=1.0):
        self.arc(cx, cy, r, 0, 360, w)

    def rect(self, x, y, w, h, lw=1.0):
        self.poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], lw, True)

    def text_width(self, s, size, bold=False):
        return len(str(s)) * size * (0.55 if bold else 0.5) * 0.96

    # output --------------------------------------------------------------
    def write_png(self, path):
        W, H = self.w * self.s, self.h * self.s
        raw = bytearray()
        for row in range(H):
            raw.append(0)
            raw += self.buf[row * W * 3:(row + 1) * W * 3]
        idat = zlib.compress(bytes(raw), 6)

        def chunk(tag, data):
            c = struct.pack(">I", len(data)) + tag + data
            return c + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
        out = b"\x89PNG\r\n\x1a\n"
        out += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
        out += chunk(b"IDAT", idat)
        out += chunk(b"IEND", b"")
        open(path, "wb").write(out)
