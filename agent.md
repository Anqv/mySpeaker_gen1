# agent.md — notes for AI coding agents

## Project

DIY 3-way speaker project. Measured driver data lives in `Mätdata/`
(keep the Swedish spelling). Three generations of crossover documentation,
plus tooling to analyze and render it.

| Path | What it is |
| --- | --- |
| `Mätdata/` | LspCAD text exports: `*_imp.txt` = impedance (Hz, ohm, deg); other `.txt` = SPL (Hz, dB, deg) |
| `Test av olika nya filter/` | Three LspCAD 6 `.lsp` filter files + `filter_schematics.pdf` (generated) |
| `analyze_speakers.py` | Reads `Mätdata/`, plots, writes report to `analysis/` |
| `plot_speaker_gui.py` | PySide6 viewer for the same measurements |
| `crossover_design.py` | 3-way LR4 crossover proposal + illustrated PDF |
| `lspcad_schematics.py` | Parses the `.lsp` filter files, rebuilds netlists, draws one schematic page per file into `filter_schematics.pdf` |
| `simple_pdf.py` | Dependency-free PDF writer (objects, xref, FlateDecode, base-14 fonts) |
| `debug_canvas.py` | Dev-only PNG rasterizer with the same drawing API as `simple_pdf.Page`; use it to eyeball layouts without a PDF viewer |

## Environment constraints (verified Nov 2026)

- The project `.venv` was **removed on purpose** (Nov 2026) and this
  environment has **no pip/ensurepip/uv** on system `python3`. So
  `analyze_speakers.py`, `crossover_design.py` and `plot_speaker_gui.py`
  (numpy/pandas/matplotlib/PySide6) are **unrunnable here** until someone
  rebuilds the venv on a machine that has pip. Don't trust a stale
  `.venv` to exist.
- `analyze_speakers.py` and `crossover_design.py` import numpy/pandas/
  matplotlib and may be **unrunnable here**. The newer schematic tooling
  is strictly stdlib-only (`zlib`, `struct`, `math`, `re`, `os`) and runs
  with plain `python3`.
- No PDF rasterizer/viewer (`pdftoppm`, `gs`, ImageMagick) is installed,
  and the model cannot read PDFs. To visually verify PDF output, mirror
  the drawing code to `debug_canvas.Canvas(...)` and inspect the PNGs
  with the read tool, and/or log `line`/`text` calls and check
  coordinates numerically.
- `.pdf`, `.png`, images and binary data cannot be read directly by the
  model; anything visual needs rasterization + a numeric sanity check.

## Run

```bash
python3 lspcad_schematics.py           # writes Test av olika nya filter/filter_schematics.pdf
python3 lspcad_schematics.py --debug   # prints reconstructed branch chains (acceptance test)
```

`--debug` output is the spec to beat: 9 branches over 3 files, e.g.
`Treble: C1 10 µF—{L2 0.18 mH→com}—22 µF—2.2 Ω—{5.6 Ω→com}` (file 3 uses
4.7 Ω). Never "fix" the parser if debug chains would diverge from this —
fix the geometry instead.

## LspCAD `.lsp` format notes (hard-won, don't re-derive)

- Text format, components carry a numeric `type` map:
  1=group, 3=wire, 20=bent wire, 21=resistor, 22=inductor, 23=capacitor,
  26=driver, 40=generator. Group ("Component 1") children live in the same
  coordinate system; a flat regex parse captures everything.
- **All `pt` values are relative to `pt[0]`**, wires included. Treating wire
  points as absolute creates phantom coordinate tables that poison nets.
- Pins for symbols = `pt0 + rot90(ptk)` (rot value 1 = 90° CCW:
  `(x,y) → (−y,x)`). Tag "0" anchors are layout anchors, **not**
  electrical terminals — exclude them from proximity unions.
- The files have no ground symbol. The short (4-unit, type 3) stubs under
  every shunt leg and driver low pin are the return path; they union into
  a synthetic COMMON node. Type-20 wires are real conductors.
- Net merge order: exact coincidence (quantize to 0.1) → tap-on-lead
  (TAP_TOL 2.6) → near-coincidence (STUB_TOL 4.2). Shunt chains are found
  by DFS to COMMON with a minimality filter (interior nodes must have no
  other elements) or the DFS retraces the ladder.
- The bass Zobel (6.8 Ω + 33 µF) has one floating lead in **all three**
  files (~2.5 units off the net). It is attached to the bass hot node and
  marked "assumed" (`*` in the schematic and cover notes). This is
  intentional; the tolerance could be raised instead, but the current
  choice keeps false unions out.
- Driver polarity: Bass normal, Mid and Treble `invert=1` (reversed) in
  every file. Schematics and the cover must keep saying so.
- `Kanske något.flt` is a binary curve export with no schematic data —
  documented on the cover page, never parsed.

## Rendering/layout conventions (`lspcad_schematics.py`)

- Page geometry (A4 landscape, 842×595): branch main lines at y = 520 /
  346 / 174, rail = ymain − 114, separator hairlines at rail − 38,
  branch title at rail − 22, chain summary at rail − 34, header at 566/553.
- Symbol sizes: `SYM_W = {R:46, L:56, C:36}`, `LEG_H = {R:40, L:40, C:20}`.
  These are balanced so a 3-element leg (L+C+R) exactly fits the 114-unit
  rail space and vertical zigzags/arcs stay legible. Check leg-height
  budgets before changing either.
- Fonts: Helvetica base-14 with WinAnsiEncoding; non-ASCII (µ Ω ä ö −)
  must be cp1252-encoded. The debug canvas renders text as dashed rules —
  it proves *placement*, not glyph shape.

## Conventions

- Docs/comments in English (README style); keep existing Swedish folder
  and file names (`Mätdata`, `Test av olika nya filter`, `Kanske något.flt`).
- Generated artifacts that belong in the repo: the PDFs and `analysis/`
  outputs. Anything prefixed `_` or named `debug_*` is a scratch debug aid,
  safe to keep or delete.
- When touching layout, re-render the debug PNGs first, then run
  `--debug`, then a plain run — in that order.
