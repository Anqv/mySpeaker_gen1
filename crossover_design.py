#!/usr/bin/env python3
"""Design a 3-way passive crossover and write an illustrated PDF report.

The design is based on the measured SPL and impedance files in ``Mätdata/``.
It uses second-order Linkwitz-Riley (LR2) filters at about 306 Hz and 2 kHz,
Zobel impedance compensation on each driver, and a small L-pad on the tweeter.

Run with::

    source .venv/bin/activate
    python crossover_design.py

The generated files are written to ``analysis/`` by default.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from analyze_speakers import (
    Measurement,
    analyze_impedance,
    analyze_spl,
    band_for_driver,
    discover_measurements,
    driver_color,
    driver_label,
    fractional_octave_smooth,
    ordered_drivers,
    resolve_data_dir,
)


# ---------------------------------------------------------------------------
# Design values
# ---------------------------------------------------------------------------

SMOOTH_FRACTION = 12
GRID = np.logspace(math.log10(20.0), math.log10(20000.0), 900)

# Second-order Linkwitz-Riley values, rounded to standard E12/E24 values.
L1 = 8.2e-3      # woofer low-pass, series
C1 = 33e-6       # woofer low-pass, shunt
L2 = 8.2e-3      # mid high-pass, shunt
C2 = 33e-6       # mid high-pass, series
L3 = 1.2e-3      # mid low-pass, series
C3 = 5.1e-6      # mid low-pass, shunt
L4 = 1.0e-3      # tweeter high-pass, shunt
C4 = 6.8e-6      # tweeter high-pass, series

R1 = 1.5         # tweeter L-pad, series
R2 = 18.0        # tweeter L-pad, shunt

RZ_W, CZ_W = 9.1, 3.3e-6   # woofer Zobel
RZ_M, CZ_M = 8.8, 3.3e-6   # midrange Zobel
RZ_T, CZ_T = 6.4, 2.2e-6   # tweeter Zobel

FC_LOW = 1.0 / (2.0 * math.pi * math.sqrt(L1 * C1))
FC_HIGH_MID = 1.0 / (2.0 * math.pi * math.sqrt(L3 * C3))
FC_HIGH_TWEETER = 1.0 / (2.0 * math.pi * math.sqrt(L4 * C4))

MID_POLARITY = -1.0
SOURCE_RESISTANCE = 0.1


# ---------------------------------------------------------------------------
# Data loading and interpolation
# ---------------------------------------------------------------------------


@dataclass
class DriverData:
    key: str
    label: str
    color: str
    spl: Measurement
    impedance: Measurement
    spl_analysis: dict[str, Any]
    impedance_analysis: dict[str, Any]
    spl_db: np.ndarray
    spl_phase: np.ndarray
    z: np.ndarray


def interpolate_on_grid(
    freq: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    log_freq = np.log10(freq)
    log_grid = np.log10(grid)
    return np.interp(log_grid, log_freq, values)


def load_driver_data(data_dir: Path) -> tuple[list[DriverData], list[tuple[Path, Exception]]]:
    drivers, errors = discover_measurements(data_dir)
    result: list[DriverData] = []

    for key in ordered_drivers(drivers):
        bucket = drivers[key]
        if "spl" not in bucket or "impedance" not in bucket:
            errors.append(
                (
                    data_dir,
                    ValueError(
                        f"Driver '{key}' needs both SPL and impedance files; skipping."
                    ),
                )
            )
            continue

        spl = bucket["spl"]
        impedance = bucket["impedance"]
        band = band_for_driver(key, None)
        spl_analysis = analyze_spl(spl, band, SMOOTH_FRACTION)
        impedance_analysis = analyze_impedance(impedance, fs_max=2000.0)

        phase_unwrapped = np.unwrap(np.deg2rad(spl.phase))
        spl_db = interpolate_on_grid(spl.freq, spl.mag, GRID)
        spl_phase = interpolate_on_grid(spl.freq, phase_unwrapped, GRID)

        z_complex = impedance.mag * np.exp(1j * np.deg2rad(impedance.phase))
        z = interpolate_on_grid(impedance.freq, z_complex.real, GRID) + 1j * interpolate_on_grid(
            impedance.freq, z_complex.imag, GRID
        )

        result.append(
            DriverData(
                key=key,
                label=driver_label(key),
                color=driver_color(key),
                spl=spl,
                impedance=impedance,
                spl_analysis=spl_analysis,
                impedance_analysis=impedance_analysis,
                spl_db=spl_db,
                spl_phase=spl_phase,
                z=z,
            )
        )

    return result, errors


# ---------------------------------------------------------------------------
# Circuit simulation
# ---------------------------------------------------------------------------


def solve_ladder(
    series: Iterable[complex],
    shunts: Iterable[complex],
    z_load: complex,
    source_resistance: float = SOURCE_RESISTANCE,
) -> tuple[complex, complex]:
    """Solve a ladder network.

    Returns ``(driver_voltage, branch_input_impedance)`` for a 1 V source.
    ``series`` are complex impedances, ``shunts`` are complex admittances.
    """

    series = list(series)
    shunts = list(shunts)
    n_nodes = len(series) + 1

    y_matrix = np.zeros((n_nodes, n_nodes), dtype=complex)
    current = np.zeros(n_nodes, dtype=complex)

    # Norton equivalent of the 1 V source behind source_resistance.
    y_matrix[0, 0] += 1.0 / source_resistance
    current[0] += 1.0 / source_resistance

    for index, impedance in enumerate(series):
        left = index
        right = index + 1
        admittance = 1.0 / impedance
        y_matrix[left, left] += admittance
        y_matrix[right, right] += admittance
        y_matrix[left, right] -= admittance
        y_matrix[right, left] -= admittance

    for index, admittance in enumerate(shunts):
        y_matrix[index + 1, index + 1] += admittance

    y_matrix[-1, -1] += 1.0 / z_load
    voltage = np.linalg.solve(y_matrix, current)

    input_current = (1.0 - voltage[0]) / source_resistance
    input_impedance = voltage[0] / input_current
    return voltage[-1], input_impedance


def zobel_load(z_driver: complex, r_zobel: float, c_zobel: float, omega: float) -> complex:
    z_zobel = r_zobel + 1.0 / (1j * omega * c_zobel)
    return 1.0 / (1.0 / z_driver + 1.0 / z_zobel)


def simulate_crossover(data: list[DriverData]) -> dict[str, Any]:
    by_key = {driver.key: driver for driver in data}
    woofer = by_key["bas"]
    mid = by_key["mid"]
    tweeter = by_key["high"]

    omega = 2.0 * np.pi * GRID
    h_woofer = np.empty_like(GRID, dtype=complex)
    h_mid = np.empty_like(GRID, dtype=complex)
    h_tweeter = np.empty_like(GRID, dtype=complex)
    z_woofer_in = np.empty_like(GRID, dtype=complex)
    z_mid_in = np.empty_like(GRID, dtype=complex)
    z_tweeter_in = np.empty_like(GRID, dtype=complex)

    for index, w in enumerate(omega):
        zw = zobel_load(woofer.z[index], RZ_W, CZ_W, w)
        zm = zobel_load(mid.z[index], RZ_M, CZ_M, w)
        zt = zobel_load(tweeter.z[index], RZ_T, CZ_T, w)

        h_woofer[index], z_woofer_in[index] = solve_ladder(
            [1j * w * L1],
            [1j * w * C1],
            zw,
        )
        h_mid[index], z_mid_in[index] = solve_ladder(
            [1.0 / (1j * w * C2), 1j * w * L3],
            [1.0 / (1j * w * L2), 1j * w * C3],
            zm,
        )
        h_tweeter[index], z_tweeter_in[index] = solve_ladder(
            [1.0 / (1j * w * C4), R1],
            [1.0 / (1j * w * L4), 1.0 / R2],
            zt,
        )

    z_total = 1.0 / (
        1.0 / z_woofer_in + 1.0 / z_mid_in + 1.0 / z_tweeter_in
    )

    sw = 10.0 ** (woofer.spl_db / 20.0) * np.exp(1j * woofer.spl_phase) * h_woofer
    sm = 10.0 ** (mid.spl_db / 20.0) * np.exp(1j * mid.spl_phase) * h_mid
    st = 10.0 ** (tweeter.spl_db / 20.0) * np.exp(1j * tweeter.spl_phase) * h_tweeter
    summed = sw + MID_POLARITY * sm + st

    summed_db = 20.0 * np.log10(np.abs(summed))
    band_mask = (GRID >= 300.0) & (GRID <= 3000.0)
    offset = float(np.median(summed_db[band_mask]))
    summed_db -= offset

    filtered_woofer_db = 20.0 * np.log10(np.abs(h_woofer)) + woofer.spl_db - offset
    filtered_mid_db = 20.0 * np.log10(np.abs(h_mid)) + mid.spl_db - offset
    filtered_tweeter_db = 20.0 * np.log10(np.abs(h_tweeter)) + tweeter.spl_db - offset

    response_mask = (GRID >= 300.0) & (GRID <= 10000.0)
    response = summed_db[response_mask]
    minimum_index = int(np.argmin(np.abs(z_total)))
    response_min = float(np.min(response))
    response_max = float(np.max(response))

    return {
        "h_woofer": h_woofer,
        "h_mid": h_mid,
        "h_tweeter": h_tweeter,
        "filtered_woofer_db": filtered_woofer_db,
        "filtered_mid_db": filtered_mid_db,
        "filtered_tweeter_db": filtered_tweeter_db,
        "summed_db": summed_db,
        "summed_offset": offset,
        "z_total": z_total,
        "z_min": float(np.abs(z_total[minimum_index])),
        "z_min_freq": float(GRID[minimum_index]),
        "response_min": response_min,
        "response_max": response_max,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def format_frequency(value: float | None, digits: int = 0) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    if value >= 1000.0:
        return f"{value / 1000.0:.{max(digits, 1)}f}k"
    return f"{value:.{digits}f}"


def format_value(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def freq_formatter(value: float, _position: int | None = None) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:g}k"
    return f"{value:g}"


def setup_frequency_axis(ax: plt.Axes, f_min: float = 20.0, f_max: float = 20000.0) -> None:
    ax.set_xscale("log")
    ax.set_xlim(f_min, f_max)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_major_formatter(FuncFormatter(freq_formatter))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(labelsize=8)


def add_crossover_lines(ax: plt.Axes, labels: bool = False) -> None:
    for frequency, label in ((FC_LOW, "306 Hz"), (2000.0, "2.0 kHz")):
        ax.axvline(frequency, color="0.45", linestyle="--", linewidth=0.8, zorder=0)
        if labels:
            ax.text(
                frequency,
                0.98,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7,
                color="0.35",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0},
            )


def add_header(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.045, 0.958, title, fontsize=19, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.045, 0.917, subtitle, fontsize=9.5, color="0.35", va="top")
    fig.add_artist(
        Line2D([0.045, 0.955], [0.895, 0.895], color="0.75", linewidth=0.8)
    )


def add_footer(fig: plt.Figure, page: int, total: int) -> None:
    fig.text(0.955, 0.026, f"{page} / {total}", fontsize=8, color="0.45", ha="right")
    fig.text(
        0.045,
        0.026,
        "3-way passive crossover · simulation based on Mätdata",
        fontsize=7.5,
        color="0.45",
    )


def add_note(fig: plt.Figure, text: str, y: float = 0.045) -> None:
    fig.text(0.045, y, text, fontsize=7.4, color="0.35", va="bottom")


# ---------------------------------------------------------------------------
# Page 1: summary
# ---------------------------------------------------------------------------


def page_summary(
    pdf: PdfPages,
    data: list[DriverData],
    simulation: dict[str, Any],
    page: int,
    total: int,
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    add_header(
        fig,
        "3-Way Passive Crossover — Design Proposal",
        "Based on the measured woofer / midrange / tweeter files in Mätdata/  ·  6 November 2026",
    )

    # Driver table
    ax_table = fig.add_axes([0.045, 0.47, 0.52, 0.33])
    ax_table.axis("off")
    ax_table.set_title("Driver summary", loc="left", fontsize=12, fontweight="bold", pad=8)

    headers = ["Driver", "Re (Ω)", "Fs (Hz)", "Qts", "Le (mH)", "Rel. dB", "−3 dB range (Hz)"]
    rows = []
    for driver in data:
        imp = driver.impedance_analysis
        spl = driver.spl_analysis
        low = spl.get("f3_low_hz")
        high = spl.get("f3_high_hz")
        if high is None or high >= 19900.0:
            range_text = f"{format_frequency(low)} – >20k"
        else:
            range_text = f"{format_frequency(low)} – {format_frequency(high)}"
        rows.append(
            [
                driver.label,
                format_value(imp.get("re_ohm"), 2),
                format_value(imp.get("fs_hz"), 1),
                format_value(imp.get("qts"), 2),
                format_value(imp.get("le_mh"), 2),
                format_value(spl.get("spl_band_median_db"), 1),
                range_text,
            ]
        )

    table = ax_table.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        colWidths=[0.24, 0.10, 0.10, 0.08, 0.10, 0.11, 0.27],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.82")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#e8eef5")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#fafbfc")

    fig.text(
        0.045,
        0.425,
        "T/S values are rough estimates from the impedance curves; the drivers may be box-loaded. "
        "SPL is relative / uncalibrated.",
        fontsize=7.2,
        color="0.38",
        va="top",
    )

    # Design summary box
    ax_box = fig.add_axes([0.60, 0.40, 0.36, 0.44])
    ax_box.axis("off")
    ax_box.add_patch(
        FancyBboxPatch(
            (0.01, 0.01),
            0.98,
            0.98,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            transform=ax_box.transAxes,
            facecolor="#f4f7fb",
            edgecolor="#9fb3c8",
            linewidth=1.0,
        )
    )
    ax_box.text(
        0.06,
        0.93,
        "CROSSOVER AT A GLANCE",
        transform=ax_box.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    box_body = (
        f"Woofer → mid:  {FC_LOW:.0f} Hz, LR2 (12 dB/oct)\n"
        "Mid → tweeter:  ~2.0 kHz, LR2\n"
        "Mid polarity:  reversed\n"
        "Tweeter pad:  2.5 dB (1.5 Ω / 18 Ω)\n"
        "Zobels:  9.1 Ω + 3.3 µF (W)\n"
        "         8.8 Ω + 3.3 µF (M)\n"
        "         6.4 Ω + 2.2 µF (T)\n"
        f"Predicted sum: {simulation['response_min']:+.1f} … {simulation['response_max']:+.1f} dB\n"
        "               300 Hz – 10 kHz\n"
        f"System Zmin:  {simulation['z_min']:.1f} Ω at {simulation['z_min_freq']:.0f} Hz"
    )
    ax_box.text(
        0.06,
        0.84,
        box_body,
        transform=ax_box.transAxes,
        fontsize=8.7,
        va="top",
        linespacing=1.55,
    )

    # Rationale
    fig.text(0.045, 0.355, "Why these crossover points?", fontsize=11.5, fontweight="bold")
    rationale = (
        "• 306 Hz: the woofer and midrange have almost the same measured level around\n"
        "  300 Hz, and both are more than an octave from their −3 dB limits.\n"
        "• ~2.0 kHz: the tweeter is flat above 1.5 kHz and 4× its Fs; the midrange's\n"
        "  5 kHz breakup is attenuated by the low-pass filter.\n"
        "• LR2 keeps the component count low.  The midrange is connected with reversed\n"
        "  polarity, as required by LR2.\n"
        "• Zobel networks make the driver loads more resistive near the crossover, so\n"
        "  the calculated values behave closer to the simulation."
    )
    fig.text(0.045, 0.325, rationale, fontsize=8.2, color="0.22", va="top", linespacing=1.45)

    # Notes
    fig.text(0.60, 0.345, "Important notes", fontsize=11.5, fontweight="bold")
    notes = (
        "• The SPL measurements are relative, not calibrated dB SPL.  The L-pad is a\n"
        "  starting point and may need adjustment after the final measurement.\n"
        "• The impedance files may have been measured in the box.  T/S values are\n"
        "  therefore rough estimates, not free-air data.\n"
        "• The simulation uses measured complex impedance and measured SPL phase.\n"
        "  It assumes a common time reference; the final sum must be verified.\n"
        "• Use low-DCR inductors, good-quality capacitors and 10 W resistors.  Measure the\n"
        "  finished system before final voicing."
    )
    fig.text(0.60, 0.315, notes, fontsize=8.0, color="0.22", va="top", linespacing=1.45)

    add_footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 2: measurements
# ---------------------------------------------------------------------------


def page_measurements(
    pdf: PdfPages,
    data: list[DriverData],
    page: int,
    total: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.15, hspace=0.35, wspace=0.22)
    add_header(
        fig,
        "Measured driver data",
        "SPL magnitude and phase, impedance magnitude and phase  ·  phase unwrapped for readability",
    )

    ax_spl, ax_z, ax_spl_phase, ax_z_phase = axes.flat

    for driver in data:
        smoothed = fractional_octave_smooth(driver.spl.freq, driver.spl.mag, SMOOTH_FRACTION)
        ax_spl.plot(driver.spl.freq, driver.spl.mag, color=driver.color, alpha=0.22, linewidth=0.7)
        ax_spl.plot(
            driver.spl.freq,
            smoothed,
            color=driver.color,
            linewidth=1.8,
            label=driver.label,
        )
        ax_z.plot(
            driver.impedance.freq,
            driver.impedance.mag,
            color=driver.color,
            linewidth=1.7,
            label=driver.label,
        )
        ax_spl_phase.plot(
            driver.spl.freq,
            np.rad2deg(np.unwrap(np.deg2rad(driver.spl.phase))),
            color=driver.color,
            linewidth=1.2,
            label=driver.label,
        )
        ax_z_phase.plot(
            driver.impedance.freq,
            np.rad2deg(np.unwrap(np.deg2rad(driver.impedance.phase))),
            color=driver.color,
            linewidth=1.4,
            label=driver.label,
        )

    ax_spl.set_title("SPL magnitude (1/12-oct smoothed)", loc="left", fontsize=11, fontweight="bold")
    ax_spl.set_ylabel("Relative SPL (dB)")
    ax_spl.legend(fontsize=7.5, loc="lower right")

    ax_z.set_title("Impedance magnitude", loc="left", fontsize=11, fontweight="bold")
    ax_z.set_ylabel("|Z| (Ω)")
    ax_z.set_ylim(0.0, max(float(driver.impedance.mag.max()) for driver in data) * 1.12)
    ax_z.legend(fontsize=7.5, loc="upper right")

    ax_spl_phase.set_title("SPL phase (unwrapped)", loc="left", fontsize=11, fontweight="bold")
    ax_spl_phase.set_ylabel("Phase (deg)")
    ax_spl_phase.axhline(0.0, color="0.65", linewidth=0.7)
    ax_spl_phase.legend(fontsize=7.5, loc="lower left")

    ax_z_phase.set_title("Impedance phase (unwrapped)", loc="left", fontsize=11, fontweight="bold")
    ax_z_phase.set_ylabel("Phase (deg)")
    ax_z_phase.axhline(0.0, color="0.65", linewidth=0.7)
    ax_z_phase.legend(fontsize=7.5, loc="upper right")

    for ax in axes.flat:
        setup_frequency_axis(ax)
        add_crossover_lines(ax, labels=ax in (ax_spl, ax_z))

    for ax in (ax_spl_phase, ax_z_phase):
        ax.set_xlabel("Frequency (Hz)")

    add_note(
        fig,
        "SPL is relative / uncalibrated.  Impedance was measured with the same setup.  "
        "Dashed lines mark the proposed 306 Hz and 2.0 kHz crossover points.",
        y=0.065,
    )
    add_footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 3: schematic
# ---------------------------------------------------------------------------


def _wire(ax: plt.Axes, points: list[tuple[float, float]], color: str = "k", lw: float = 1.2) -> None:
    ax.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        color=color,
        linewidth=lw,
        solid_capstyle="round",
        zorder=1,
    )


def _node(ax: plt.Axes, x: float, y: float, color: str = "k") -> None:
    ax.plot([x], [y], "o", color=color, markersize=3.5, zorder=3)


def _ground(ax: plt.Axes, x: float, y: float) -> None:
    _wire(ax, [(x, y), (x, y - 1.2)])
    for index, width in enumerate((3.0, 2.0, 1.0)):
        ax.plot(
            [x - width / 2.0, x + width / 2.0],
            [y - 1.2 - index * 0.55, y - 1.2 - index * 0.55],
            color="k",
            linewidth=1.1,
        )


def _inductor_h(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    label: str,
    value: str,
    color: str = "k",
) -> None:
    _wire(ax, [(x1, y), (x2, y)], color=color)
    count = 4
    width = (x2 - x1) / count
    for index in range(count):
        ax.add_patch(
            Arc(
                (x1 + (index + 0.5) * width, y),
                width * 0.95,
                width * 0.95,
                theta1=0,
                theta2=180,
                color=color,
                linewidth=1.2,
                zorder=2,
            )
        )
    ax.text(
        (x1 + x2) / 2.0,
        y + 2.7,
        f"{label}\n{value}",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )


def _inductor_v(
    ax: plt.Axes,
    x: float,
    y1: float,
    y2: float,
    label: str,
    value: str,
    color: str = "k",
) -> None:
    _wire(ax, [(x, y1), (x, y2)], color=color)
    count = 4
    height = (y2 - y1) / count
    for index in range(count):
        center_y = y1 + (index + 0.5) * height
        ax.add_patch(
            Arc(
                (x, center_y),
                height * 0.95,
                height * 0.95,
                theta1=90,
                theta2=270,
                color=color,
                linewidth=1.2,
                zorder=2,
            )
        )
    ax.text(
        x + 2.2,
        (y1 + y2) / 2.0,
        f"{label}\n{value}",
        ha="left",
        va="center",
        fontsize=7.5,
    )


def _capacitor_h(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    label: str,
    value: str,
    color: str = "k",
) -> None:
    middle = (x1 + x2) / 2.0
    gap = 0.8
    _wire(ax, [(x1, y), (middle - gap, y)], color=color)
    _wire(ax, [(middle + gap, y), (x2, y)], color=color)
    ax.plot([middle - gap, middle - gap], [y - 1.8, y + 1.8], color=color, linewidth=1.8)
    ax.plot([middle + gap, middle + gap], [y - 1.8, y + 1.8], color=color, linewidth=1.8)
    ax.text(middle, y + 2.7, f"{label}\n{value}", ha="center", va="bottom", fontsize=7.5)


def _capacitor_v(
    ax: plt.Axes,
    x: float,
    y1: float,
    y2: float,
    label: str,
    value: str,
    color: str = "k",
) -> None:
    middle = (y1 + y2) / 2.0
    gap = 0.8
    _wire(ax, [(x, y1), (x, middle + gap)], color=color)
    _wire(ax, [(x, middle - gap), (x, y2)], color=color)
    ax.plot([x - 1.8, x + 1.8], [middle + gap, middle + gap], color=color, linewidth=1.8)
    ax.plot([x - 1.8, x + 1.8], [middle - gap, middle - gap], color=color, linewidth=1.8)
    ax.text(x + 2.2, middle, f"{label}\n{value}", ha="left", va="center", fontsize=7.5)


def _resistor_v(
    ax: plt.Axes,
    x: float,
    y1: float,
    y2: float,
    label: str,
    value: str,
    color: str = "k",
) -> None:
    middle = (y1 + y2) / 2.0
    height = 4.4
    width = 2.2
    _wire(ax, [(x, y1), (x, middle + height / 2.0)], color=color)
    _wire(ax, [(x, middle - height / 2.0), (x, y2)], color=color)
    ax.add_patch(
        Rectangle(
            (x - width / 2.0, middle - height / 2.0),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=1.3,
            zorder=2,
        )
    )
    ax.text(x + 2.2, middle, f"{label}\n{value}", ha="left", va="center", fontsize=7.5)


def _resistor_h(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    label: str,
    value: str,
    color: str = "k",
) -> None:
    middle = (x1 + x2) / 2.0
    width = 4.4
    height = 2.2
    _wire(ax, [(x1, y), (middle - width / 2.0, y)], color=color)
    _wire(ax, [(middle + width / 2.0, y), (x2, y)], color=color)
    ax.add_patch(
        Rectangle(
            (middle - width / 2.0, y - height / 2.0),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=1.3,
            zorder=2,
        )
    )
    ax.text(middle, y + 2.7, f"{label}\n{value}", ha="center", va="bottom", fontsize=7.5)


def _driver(
    ax: plt.Axes,
    x: float,
    y: float,
    letter: str,
    name: str,
    note: str,
    color: str,
) -> None:
    radius = 3.4
    ax.add_patch(Circle((x, y), radius, fill=False, edgecolor=color, linewidth=1.5, zorder=2))
    ax.add_patch(
        Rectangle(
            (x - 2.0, y + radius),
            4.0,
            1.8,
            fill=False,
            edgecolor=color,
            linewidth=1.2,
            zorder=2,
        )
    )
    ax.text(x, y, letter, ha="center", va="center", fontsize=10, fontweight="bold", color=color)
    ax.text(x + radius + 2.0, y + 1.2, name, ha="left", va="center", fontsize=8, fontweight="bold", color=color)
    ax.text(x + radius + 2.0, y - 1.2, note, ha="left", va="center", fontsize=7, color="0.35")


def _zobel(
    ax: plt.Axes,
    x: float,
    y: float,
    ref_r: str,
    value_r: str,
    ref_c: str,
    value_c: str,
    color: str,
) -> None:
    width, height = 9.0, 5.2
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2.0, y - height / 2.0),
            width,
            height,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            fill=False,
            edgecolor="0.45",
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            zorder=2,
        )
    )
    ax.text(
        x,
        y,
        f"Zobel\n{ref_r} {value_r}\n{ref_c} {value_c}",
        ha="center",
        va="center",
        fontsize=6.8,
        color="0.25",
    )
    _wire(ax, [(x, y + height / 2.0), (x, y + height / 2.0 + 1.5)], color=color)
    _wire(ax, [(x, y - height / 2.0), (x, y - height / 2.0 - 1.5)], color=color)


def draw_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    woofer_color = driver_color("bas")
    mid_color = driver_color("mid")
    tweeter_color = driver_color("high")

    y_woofer, y_mid, y_tweeter = 50.0, 34.0, 18.0

    # Input
    ax.text(1.5, 39.5, "FROM\nAMPLIFIER", ha="left", va="center", fontsize=8.5, fontweight="bold")
    ax.text(7.7, 37.2, "+", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(7.7, 31.2, "−", ha="center", va="center", fontsize=10, fontweight="bold")
    _wire(ax, [(8.2, 36.0), (10.0, 36.0)])
    _wire(ax, [(8.2, 30.0), (7.5, 30.0)])
    _ground(ax, 7.5, 30.0)
    _wire(ax, [(10.0, 36.0), (10.0, y_woofer)])
    _wire(ax, [(10.0, 36.0), (10.0, y_tweeter)])

    for y, label, color in (
        (y_woofer, "WOOFER", woofer_color),
        (y_mid, "MID", mid_color),
        (y_tweeter, "TWEETER", tweeter_color),
    ):
        _wire(ax, [(10.0, y), (14.0, y)], color=color)
        ax.text(1.5, y, label, ha="left", va="center", fontsize=8.5, fontweight="bold", color=color)

    # Woofer branch
    _inductor_h(ax, 14.0, 24.0, y_woofer, "L1", "8.2 mH", woofer_color)
    _node(ax, 24.0, y_woofer, woofer_color)
    _capacitor_v(ax, 24.0, y_woofer, 38.0, "C1", "33 µF", woofer_color)
    _ground(ax, 24.0, 38.0)
    _wire(ax, [(24.0, y_woofer), (32.0, y_woofer)], color=woofer_color)
    _zobel(ax, 36.0, 45.0, "R3", "9.1 Ω", "C5", "3.3 µF", woofer_color)
    _wire(ax, [(32.0, y_woofer), (36.0, y_woofer), (36.0, 47.6)], color=woofer_color)
    _wire(ax, [(36.0, 42.4), (36.0, 38.0)], color=woofer_color)
    _ground(ax, 36.0, 38.0)
    _wire(ax, [(32.0, y_woofer), (58.0, y_woofer)], color=woofer_color)
    _driver(ax, 58.0, 44.0, "W", "Woofer", "8 Ω", woofer_color)
    ax.text(56.2, 48.2, "+", ha="right", va="center", fontsize=7.5, fontweight="bold")
    ax.text(56.2, 39.8, "−", ha="right", va="center", fontsize=7.5, fontweight="bold")
    _wire(ax, [(58.0, y_woofer), (58.0, 47.4)], color=woofer_color)
    _wire(ax, [(58.0, 40.6), (58.0, 38.0)], color=woofer_color)
    _ground(ax, 58.0, 38.0)

    # Midrange branch
    _capacitor_h(ax, 14.0, 22.0, y_mid, "C2", "33 µF", mid_color)
    _node(ax, 22.0, y_mid, mid_color)
    _inductor_v(ax, 22.0, y_mid, 22.0, "L2", "8.2 mH", mid_color)
    _ground(ax, 22.0, 22.0)
    _inductor_h(ax, 22.0, 32.0, y_mid, "L3", "1.2 mH", mid_color)
    _node(ax, 32.0, y_mid, mid_color)
    _capacitor_v(ax, 32.0, y_mid, 22.0, "C3", "5.1 µF", mid_color)
    _ground(ax, 32.0, 22.0)
    _wire(ax, [(32.0, y_mid), (36.0, y_mid)], color=mid_color)
    _zobel(ax, 44.0, 29.0, "R4", "8.8 Ω", "C6", "3.3 µF", mid_color)
    _wire(ax, [(36.0, y_mid), (44.0, y_mid), (44.0, 31.6)], color=mid_color)
    _wire(ax, [(44.0, 26.4), (44.0, 22.0)], color=mid_color)
    _ground(ax, 44.0, 22.0)
    _wire(ax, [(36.0, y_mid), (58.0, y_mid)], color=mid_color)
    _driver(ax, 58.0, 28.0, "M", "Midrange", "8 Ω · reverse polarity", mid_color)
    ax.text(56.2, 32.2, "−", ha="right", va="center", fontsize=7.5, fontweight="bold")
    ax.text(56.2, 23.8, "+", ha="right", va="center", fontsize=7.5, fontweight="bold")
    _wire(ax, [(58.0, y_mid), (58.0, 31.4)], color=mid_color)
    _wire(ax, [(58.0, 24.6), (58.0, 22.0)], color=mid_color)
    _ground(ax, 58.0, 22.0)

    # Tweeter branch
    _capacitor_h(ax, 14.0, 22.0, y_tweeter, "C4", "6.8 µF", tweeter_color)
    _node(ax, 22.0, y_tweeter, tweeter_color)
    _inductor_v(ax, 22.0, y_tweeter, 6.0, "L4", "1.0 mH", tweeter_color)
    _ground(ax, 22.0, 6.0)
    _resistor_h(ax, 22.0, 30.0, y_tweeter, "R1", "1.5 Ω", tweeter_color)
    _node(ax, 30.0, y_tweeter, tweeter_color)
    _resistor_v(ax, 30.0, y_tweeter, 6.0, "R2", "18 Ω", tweeter_color)
    _ground(ax, 30.0, 6.0)
    _wire(ax, [(30.0, y_tweeter), (34.0, y_tweeter)], color=tweeter_color)
    _zobel(ax, 44.0, 13.0, "R5", "6.4 Ω", "C7", "2.2 µF", tweeter_color)
    _wire(ax, [(34.0, y_tweeter), (44.0, y_tweeter), (44.0, 15.6)], color=tweeter_color)
    _wire(ax, [(44.0, 10.4), (44.0, 6.0)], color=tweeter_color)
    _ground(ax, 44.0, 6.0)
    _wire(ax, [(34.0, y_tweeter), (58.0, y_tweeter)], color=tweeter_color)
    _driver(ax, 58.0, 12.0, "T", "Tweeter", "6 Ω", tweeter_color)
    ax.text(56.2, 16.2, "+", ha="right", va="center", fontsize=7.5, fontweight="bold")
    ax.text(56.2, 7.8, "−", ha="right", va="center", fontsize=7.5, fontweight="bold")
    _wire(ax, [(58.0, y_tweeter), (58.0, 15.4)], color=tweeter_color)
    _wire(ax, [(58.0, 8.6), (58.0, 6.0)], color=tweeter_color)
    _ground(ax, 58.0, 6.0)

    ax.text(
        2.0,
        1.2,
        "All ground symbols are common.  Midrange reversed (− to crossover, + to ground).\n"
        "Zobel networks are across the driver terminals.",
        ha="left",
        va="bottom",
        fontsize=7.3,
        color="0.35",
        linespacing=1.35,
    )


def page_schematic(pdf: PdfPages, page: int, total: int) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    add_header(
        fig,
        "Proposed 3-way passive crossover",
        "LR2 at 306 Hz and ~2.0 kHz  ·  midrange reversed  ·  Zobel networks included",
    )

    ax_schematic = fig.add_axes([0.02, 0.06, 0.62, 0.82])
    draw_schematic(ax_schematic)

    ax_table = fig.add_axes([0.66, 0.10, 0.32, 0.76])
    ax_table.axis("off")
    ax_table.set_title("Component list", loc="left", fontsize=12, fontweight="bold", pad=8)
    rows = [
        ["L1", "8.2 mH", "Woofer LP series"],
        ["C1", "33 µF", "Woofer LP shunt"],
        ["L2", "8.2 mH", "Mid HP shunt"],
        ["C2", "33 µF", "Mid HP series"],
        ["L3", "1.2 mH", "Mid LP series"],
        ["C3", "5.1 µF", "Mid LP shunt"],
        ["L4", "1.0 mH", "Tweeter HP shunt"],
        ["C4", "6.8 µF", "Tweeter HP series"],
        ["R1", "1.5 Ω", "Tweeter L-pad series"],
        ["R2", "18 Ω", "Tweeter L-pad shunt"],
        ["R3", "9.1 Ω", "Woofer Zobel"],
        ["C5", "3.3 µF", "Woofer Zobel"],
        ["R4", "8.8 Ω", "Mid Zobel"],
        ["C6", "3.3 µF", "Mid Zobel"],
        ["R5", "6.4 Ω", "Tweeter Zobel"],
        ["C7", "2.2 µF", "Tweeter Zobel"],
    ]
    table = ax_table.table(
        cellText=rows,
        colLabels=["Ref", "Value", "Function"],
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        colWidths=[0.16, 0.22, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.8)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.82")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#e8eef5")
            cell.set_text_props(fontweight="bold")

    fig.text(
        0.66,
        0.075,
        "Calculated LR2: 306 Hz (W/M),\n"
        "2034 Hz (mid LP), 1930 Hz (tweeter HP).\n"
        "Use low-DCR inductors (L1 < 0.5 Ω;\n"
        "L2–L4 < 0.3 Ω), good-quality capacitors\n"
        "and 10 W resistors.",
        fontsize=6.8,
        color="0.38",
        va="bottom",
        linespacing=1.35,
    )
    add_footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 4: predicted performance
# ---------------------------------------------------------------------------


def page_predicted(
    pdf: PdfPages,
    data: list[DriverData],
    simulation: dict[str, Any],
    page: int,
    total: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.07, right=0.90, top=0.86, bottom=0.15, hspace=0.35, wspace=0.38)
    add_header(
        fig,
        "Predicted performance",
        "Simulation using the measured complex impedance and measured SPL phase  ·  midrange reversed",
    )

    ax_response, ax_transfer, ax_sum, ax_impedance = axes.flat
    by_key = {driver.key: driver for driver in data}

    # Individual filtered responses
    for key, filtered, label in (
        ("bas", simulation["filtered_woofer_db"], "Woofer"),
        ("mid", simulation["filtered_mid_db"], "Midrange"),
        ("high", simulation["filtered_tweeter_db"], "Tweeter"),
    ):
        driver = by_key[key]
        ax_response.plot(
            GRID,
            driver.spl_db - simulation["summed_offset"],
            color=driver.color,
            alpha=0.20,
            linewidth=0.8,
        )
        ax_response.plot(
            GRID,
            filtered,
            color=driver.color,
            linewidth=1.6,
            label=label,
        )
    ax_response.plot(GRID, simulation["summed_db"], color="k", linewidth=2.2, label="Sum")
    ax_response.set_title("Predicted acoustic response", loc="left", fontsize=11, fontweight="bold")
    ax_response.set_ylabel("Relative SPL (dB)")
    ax_response.set_ylim(-30.0, 10.0)
    ax_response.legend(fontsize=7.5, loc="lower right", ncol=2)

    # Electrical transfer functions
    for key, transfer, label in (
        ("bas", simulation["h_woofer"], "Woofer LP"),
        ("mid", simulation["h_mid"], "Mid HP+LP"),
        ("high", simulation["h_tweeter"], "Tweeter HP"),
    ):
        ax_transfer.plot(
            GRID,
            20.0 * np.log10(np.abs(transfer)),
            color=by_key[key].color,
            linewidth=1.6,
            label=label,
        )
    ax_transfer.set_title("Electrical transfer functions", loc="left", fontsize=11, fontweight="bold")
    ax_transfer.set_ylabel("|H| (dB)")
    ax_transfer.set_ylim(-60.0, 5.0)
    ax_transfer.legend(fontsize=7.5, loc="lower left")

    # Summed response
    ax_sum.plot(GRID, simulation["summed_db"], color="k", linewidth=2.2)
    ax_sum.axhspan(-3.0, 3.0, color="#2ca02c", alpha=0.08)
    ax_sum.axhline(0.0, color="0.55", linewidth=0.8)
    ax_sum.set_title("Summed response (normalized)", loc="left", fontsize=11, fontweight="bold")
    ax_sum.set_ylabel("Relative SPL (dB)")
    ax_sum.set_ylim(-10.0, 10.0)
    ax_sum.text(
        0.03,
        0.05,
        f"300 Hz – 10 kHz:  {simulation['response_min']:+.1f} … {simulation['response_max']:+.1f} dB",
        transform=ax_sum.transAxes,
        fontsize=7.5,
        color="0.25",
        va="bottom",
    )

    # System impedance
    z_abs = np.abs(simulation["z_total"])
    z_phase = np.angle(simulation["z_total"], deg=True)
    ax_impedance.plot(GRID, z_abs, color="#6a3d9a", linewidth=2.0, label="|Z|")
    ax_impedance.set_title("System impedance", loc="left", fontsize=11, fontweight="bold")
    ax_impedance.set_ylabel("|Z| (Ω)", color="#6a3d9a")
    ax_impedance.tick_params(axis="y", colors="#6a3d9a")
    ax_impedance.set_ylim(0.0, float(z_abs.max()) * 1.08)
    ax_impedance.legend(fontsize=7.5, loc="upper right")

    ax_phase = ax_impedance.twinx()
    ax_phase.plot(GRID, z_phase, color="#b15928", linewidth=1.1, linestyle="--", alpha=0.75)
    ax_phase.set_ylabel("Phase (deg)", color="#b15928")
    ax_phase.tick_params(axis="y", colors="#b15928")
    ax_phase.set_ylim(-90.0, 90.0)
    ax_impedance.text(
        0.03,
        0.05,
        f"Zmin = {simulation['z_min']:.1f} Ω at {simulation['z_min_freq']:.0f} Hz",
        transform=ax_impedance.transAxes,
        fontsize=7.5,
        color="0.25",
        va="bottom",
    )

    for ax in axes.flat:
        setup_frequency_axis(ax)
        add_crossover_lines(ax, labels=ax in (ax_response, ax_transfer))

    for ax in (ax_sum, ax_impedance):
        ax.set_xlabel("Frequency (Hz)")

    add_note(
        fig,
        "Simulation assumes ideal drivers with the measured magnitude/phase and a common time reference.  "
        "The final acoustic sum must be verified with a microphone measurement.",
        y=0.065,
    )
    add_footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 5: BOM and implementation notes
# ---------------------------------------------------------------------------


def page_bom(pdf: PdfPages, page: int, total: int) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    add_header(
        fig,
        "Bill of materials and implementation notes",
        "Component values, wiring and a short setup checklist",
    )

    ax_table = fig.add_axes([0.045, 0.27, 0.55, 0.58])
    ax_table.axis("off")
    ax_table.set_title("Bill of materials", loc="left", fontsize=12, fontweight="bold", pad=8)
    rows = [
        ["L1", "8.2 mH", "Woofer LP series; low-DCR iron/ferrite, DCR < 0.5 Ω", "1"],
        ["C1", "33 µF", "Woofer LP shunt; MKP or high-quality bipolar electrolytic", "1"],
        ["L2", "8.2 mH", "Mid HP shunt; air-core, DCR < 0.3 Ω", "1"],
        ["C2", "33 µF", "Mid HP series; MKP or high-quality bipolar electrolytic", "1"],
        ["L3", "1.2 mH", "Mid LP series; air-core, DCR < 0.2 Ω", "1"],
        ["C3", "5.1 µF", "Mid LP shunt; MKP 100 V", "1"],
        ["L4", "1.0 mH", "Tweeter HP shunt; air-core, DCR < 0.2 Ω", "1"],
        ["C4", "6.8 µF", "Tweeter HP series; MKP 100 V", "1"],
        ["R1", "1.5 Ω", "Tweeter L-pad series; 10 W wirewound", "1"],
        ["R2", "18 Ω", "Tweeter L-pad shunt; 10 W wirewound", "1"],
        ["R3", "9.1 Ω", "Woofer Zobel; 5–10 W", "1"],
        ["C5", "3.3 µF", "Woofer Zobel; MKP", "1"],
        ["R4", "8.8 Ω", "Mid Zobel; 5–10 W", "1"],
        ["C6", "3.3 µF", "Mid Zobel; MKP", "1"],
        ["R5", "6.4 Ω", "Tweeter Zobel; 5–10 W", "1"],
        ["C7", "2.2 µF", "Tweeter Zobel; MKP", "1"],
    ]
    table = ax_table.table(
        cellText=rows,
        colLabels=["Ref", "Value", "Type / notes", "Qty"],
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        colWidths=[0.09, 0.13, 0.68, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1.0, 1.32)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.82")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#e8eef5")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#fafbfc")

    # Wiring
    fig.text(0.62, 0.78, "Wiring", fontsize=12, fontweight="bold")
    wiring = (
        "Input +  →  all three branch inputs\n"
        "Input −  →  common ground\n\n"
        "Woofer +  →  L1 output\n"
        "Woofer −  →  ground\n\n"
        "Mid −  →  crossover output\n"
        "Mid +  →  ground\n"
        "  (reversed polarity; or reverse the\n"
        "   mid wires at the driver)\n\n"
        "Tweeter +  →  R1 / R2 junction\n"
        "Tweeter −  →  ground"
    )
    fig.text(0.62, 0.745, wiring, fontsize=8.0, va="top", linespacing=1.45, family="DejaVu Sans Mono")

    # Checklist
    fig.text(0.62, 0.45, "Setup checklist", fontsize=12, fontweight="bold")
    checklist = (
        "1. Measure each driver's impedance in the final box.\n"
        "2. Verify the crossover regions with DATS or REW.\n"
        "3. Measure the summed response at 1 m on the tweeter axis.\n"
        "4. Adjust R2 for treble level: 22 Ω = louder, 15 Ω = softer.\n"
        "5. If the mid/tweeter sum has a dip, compare both tweeter polarities.\n"
        "6. Check that the minimum impedance stays above ~4 Ω.\n"
        "7. Listen before finalising; measurements are a guide."
    )
    fig.text(0.62, 0.415, checklist, fontsize=7.8, va="top", linespacing=1.45)

    # Caveats
    fig.text(0.045, 0.22, "Caveats", fontsize=12, fontweight="bold")
    caveats = (
        "• SPL is relative / uncalibrated.  The L-pad value is based on the relative levels in "
        "these files and may need adjustment if the absolute sensitivities differ.\n"
        "• The impedance files may be box-loaded; the T/S values are rough estimates, not "
        "free-air parameters.\n"
        "• Component tolerances and inductor DCR will shift the crossover frequencies.  "
        "Measure the finished crossover if possible.\n"
        "• This is a starting point for voicing, not a guaranteed flat system.  The final "
        "acoustic result depends on the box, baffle, room and driver unit tolerances."
    )
    fig.text(0.045, 0.185, caveats, fontsize=7.8, va="top", linespacing=1.45)

    add_footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Design a 3-way passive crossover and write an illustrated PDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="Mätdata", help="Directory with the measurement files")
    parser.add_argument("--output-dir", default="analysis", help="Directory for the PDF and CSV files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        data_dir = resolve_data_dir(args.data_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    data, errors = load_driver_data(data_dir)
    for path, error in errors:
        print(f"Warning: {path}: {error}", file=sys.stderr)

    if len(data) < 3:
        print("Error: need SPL and impedance files for three drivers.", file=sys.stderr)
        return 1

    simulation = simulate_crossover(data)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "3_way_crossover_design.pdf"

    total_pages = 5
    with PdfPages(pdf_path) as pdf:
        pdf.infodict()["Title"] = "3-Way Passive Crossover — Design Proposal"
        pdf.infodict()["Author"] = "OpenCode speaker analysis"
        pdf.infodict()["Subject"] = "Passive 3-way crossover based on measured driver data"
        page_summary(pdf, data, simulation, 1, total_pages)
        page_measurements(pdf, data, 2, total_pages)
        page_schematic(pdf, 3, total_pages)
        page_predicted(pdf, data, simulation, 4, total_pages)
        page_bom(pdf, 5, total_pages)

    # Machine-readable extras.
    bom = pd.DataFrame(
        [
            ("L1", "8.2 mH", "Woofer LP series", 1),
            ("C1", "33 uF", "Woofer LP shunt", 1),
            ("L2", "8.2 mH", "Mid HP shunt", 1),
            ("C2", "33 uF", "Mid HP series", 1),
            ("L3", "1.2 mH", "Mid LP series", 1),
            ("C3", "5.1 uF", "Mid LP shunt", 1),
            ("L4", "1.0 mH", "Tweeter HP shunt", 1),
            ("C4", "6.8 uF", "Tweeter HP series", 1),
            ("R1", "1.5 ohm", "Tweeter L-pad series", 1),
            ("R2", "18 ohm", "Tweeter L-pad shunt", 1),
            ("R3", "9.1 ohm", "Woofer Zobel", 1),
            ("C5", "3.3 uF", "Woofer Zobel", 1),
            ("R4", "8.8 ohm", "Mid Zobel", 1),
            ("C6", "3.3 uF", "Mid Zobel", 1),
            ("R5", "6.4 ohm", "Tweeter Zobel", 1),
            ("C7", "2.2 uF", "Tweeter Zobel", 1),
        ],
        columns=["Ref", "Value", "Function", "Qty"],
    )
    bom.to_csv(output_dir / "crossover_bom.csv", index=False)

    response = pd.DataFrame(
        {
            "frequency_hz": GRID,
            "woofer_filtered_db": simulation["filtered_woofer_db"],
            "mid_filtered_db": simulation["filtered_mid_db"],
            "tweeter_filtered_db": simulation["filtered_tweeter_db"],
            "summed_db": simulation["summed_db"],
            "woofer_h_db": 20.0 * np.log10(np.abs(simulation["h_woofer"])),
            "mid_h_db": 20.0 * np.log10(np.abs(simulation["h_mid"])),
            "tweeter_h_db": 20.0 * np.log10(np.abs(simulation["h_tweeter"])),
            "system_z_ohm": np.abs(simulation["z_total"]),
            "system_z_phase_deg": np.angle(simulation["z_total"], deg=True),
        }
    )
    response.to_csv(output_dir / "crossover_simulation.csv", index=False)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {output_dir / 'crossover_bom.csv'}")
    print(f"Wrote {output_dir / 'crossover_simulation.csv'}")
    print(
        f"Design: LR2 at {FC_LOW:.1f} Hz and ~{FC_HIGH_MID:.0f} Hz; "
        f"predicted sum {simulation['response_min']:+.1f}…{simulation['response_max']:+.1f} dB "
        f"(300 Hz–10 kHz); Zmin {simulation['z_min']:.1f} Ω."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
