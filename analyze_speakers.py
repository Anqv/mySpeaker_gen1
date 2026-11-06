#!/usr/bin/env python3
"""Analyse and plot loudspeaker driver measurements exported from LSPCad.

The measurement files are plain text tables with three columns:

    Freq    Mag    Phase

Files whose name ends in ``_imp`` are treated as impedance measurements
(ohm / degrees); all other files are treated as SPL measurements (dB /
degrees).  A driver called ``bas`` therefore normally has ``bas.txt`` (SPL)
and ``bas_imp.txt`` (impedance).

The script produces:

* combined SPL / phase / impedance plots
* one overview plot per driver
* a normalised SPL plot for crossover work
* CSV, Excel and Markdown summaries
* an optional interactive Plotly HTML report

Example
-------
    python analyze_speakers.py
    python analyze_speakers.py --data-dir Mätdata --output-dir analysis
    python analyze_speakers.py --smooth 24 --band 100 10000
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import matplotlib

# Use a non-interactive backend when there is no display (servers, CI, WSL).
if not os.environ.get("DISPLAY") and sys.platform not in ("darwin", "win32"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

try:  # Optional, only used for a slightly nicer default style.
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
except Exception:  # pragma: no cover - styling must never break analysis
    sns = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except Exception:  # pragma: no cover - HTML output is optional
    HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DRIVER_ORDER = ["bas", "mid", "high"]

DRIVER_LABELS = {
    "bas": "Woofer (bas)",
    "mid": "Midrange (mid)",
    "high": "Tweeter (high)",
}

DRIVER_COLORS = {
    "bas": "#1f77b4",
    "mid": "#2ca02c",
    "high": "#d62728",
}

# Sensible default analysis bands per driver.  A tweeter's "passband" is not
# 100 Hz - 10 kHz, so using one global band would give misleading statistics.
DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "bas": (30.0, 1000.0),
    "mid": (100.0, 5000.0),
    "high": (2000.0, 20000.0),
}

FALLBACK_BAND = (100.0, 10000.0)

PLOT_TITLES = {
    "01_spl_magnitude": "SPL magnitude - all drivers",
    "02_spl_phase": "SPL phase - all drivers",
    "03_impedance_magnitude": "Impedance magnitude - all drivers",
    "04_impedance_phase": "Impedance phase - all drivers",
    "05_spl_normalized": "Normalised SPL",
    "driver_bas": "Woofer (bas) - overview",
    "driver_mid": "Midrange (mid) - overview",
    "driver_high": "Tweeter (high) - overview",
}

THIRD_OCTAVE_CENTERS = [
    10.0,
    12.5,
    16.0,
    20.0,
    25.0,
    31.5,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
    12500.0,
    16000.0,
    20000.0,
]

_IMPEDANCE_SUFFIX = re.compile(r"^(?P<driver>.+?)(?:_imp|[-_ ]impedance)$", re.I)
_SPL_SUFFIX = re.compile(r"^(?P<driver>.+?)(?:_spl|[-_ ]spl)$", re.I)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class Measurement:
    """One parsed measurement file."""

    driver: str
    kind: str  # "spl" or "impedance"
    path: Path
    freq: np.ndarray
    mag: np.ndarray
    phase: np.ndarray

    @property
    def label(self) -> str:
        return driver_label(self.driver)


def driver_label(driver: str) -> str:
    return DRIVER_LABELS.get(driver.lower(), driver.capitalize())


def driver_color(driver: str) -> str:
    return DRIVER_COLORS.get(driver.lower(), "#7f7f7f")


def ordered_drivers(drivers: Iterable[str]) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        lower = name.lower()
        try:
            return (DRIVER_ORDER.index(lower), lower)
        except ValueError:
            return (len(DRIVER_ORDER), lower)

    return sorted(drivers, key=key)


def resolve_data_dir(raw: str | Path) -> Path:
    """Resolve a path, matching the directory name case-insensitively.

    This makes ``--data-dir mätdata`` work on Linux even when the directory is
    actually called ``Mätdata``.
    """

    path = Path(raw).expanduser()
    if path.is_dir():
        return path

    parent = path.parent if str(path.parent) else Path(".")
    target = path.name.lower()
    try:
        for child in parent.iterdir():
            if child.is_dir() and child.name.lower() == target:
                return child
    except OSError:
        pass

    raise FileNotFoundError(
        f"Data directory not found: {path} (also tried a case-insensitive match)"
    )


def classify_file(path: Path) -> tuple[str, str]:
    """Return ``(driver, kind)`` for a measurement filename."""

    stem = path.stem
    match = _IMPEDANCE_SUFFIX.match(stem)
    if match:
        return match.group("driver").lower(), "impedance"

    match = _SPL_SUFFIX.match(stem)
    if match:
        return match.group("driver").lower(), "spl"

    return stem.lower(), "spl"


def read_measurement(path: Path, driver: str, kind: str) -> Measurement:
    """Read a whitespace-separated LSPCad measurement file."""

    df: pd.DataFrame | None = None
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(
                path,
                sep=r"\s+",
                engine="python",
                encoding=encoding,
            )
            break
        except UnicodeDecodeError as exc:  # try the next encoding
            last_error = exc

    if df is None:
        raise ValueError(f"Could not decode {path}: {last_error}")

    if df.empty:
        raise ValueError(f"No data rows in {path}")

    # Normalise column names.  LSPCad writes "Freq", "Mag", "Phase".
    df.columns = [str(column).strip().lower() for column in df.columns]
    rename: dict[str, str] = {}
    for column in df.columns:
        if column.startswith("freq"):
            rename[column] = "freq"
        elif column.startswith("mag") or column in {"z", "ohm", "impedance"}:
            rename[column] = "mag"
        elif column.startswith("phase") or column.startswith("ang"):
            rename[column] = "phase"
    df = df.rename(columns=rename)

    if "freq" not in df.columns or "mag" not in df.columns:
        # Fall back to positional columns for header-less or unusual exports.
        if len(df.columns) < 2:
            raise ValueError(f"Expected at least two columns in {path}")
        names = ["freq", "mag", "phase"][: len(df.columns)]
        df = df.iloc[:, : len(names)].copy()
        df.columns = names

    if "phase" not in df.columns:
        df["phase"] = 0.0

    df = df[["freq", "mag", "phase"]].apply(pd.to_numeric, errors="coerce")
    df["phase"] = df["phase"].fillna(0.0)
    df = df.dropna(subset=["freq", "mag"])
    df = df[np.isfinite(df["freq"]) & np.isfinite(df["mag"]) & (df["freq"] > 0)]
    df = df.sort_values("freq").drop_duplicates("freq")

    if len(df) < 5:
        raise ValueError(f"Not enough valid data rows in {path}")

    return Measurement(
        driver=driver,
        kind=kind,
        path=path,
        freq=df["freq"].to_numpy(dtype=float),
        mag=df["mag"].to_numpy(dtype=float),
        phase=df["phase"].to_numpy(dtype=float),
    )


def discover_measurements(
    data_dir: Path,
) -> tuple[dict[str, dict[str, Measurement]], list[tuple[Path, Exception]]]:
    """Group measurement files by driver and measurement kind."""

    by_driver: dict[str, dict[str, Measurement]] = {}
    errors: list[tuple[Path, Exception]] = []

    extensions = {".txt", ".dat", ".csv"}
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        driver, kind = classify_file(path)
        try:
            measurement = read_measurement(path, driver, kind)
        except Exception as exc:  # keep going if one file is malformed
            errors.append((path, exc))
            continue

        bucket = by_driver.setdefault(driver, {})
        if kind in bucket:
            errors.append(
                (
                    path,
                    ValueError(
                        f"Duplicate {kind} measurement for '{driver}'; "
                        f"using {path.name}, ignoring {bucket[kind].path.name}"
                    ),
                )
            )
            continue
        bucket[kind] = measurement

    return by_driver, errors


# ---------------------------------------------------------------------------
# Signal analysis
# ---------------------------------------------------------------------------


def band_for_driver(driver: str, override: tuple[float, float] | None) -> tuple[float, float]:
    if override is not None:
        return override
    return DEFAULT_BANDS.get(driver.lower(), FALLBACK_BAND)


def fractional_octave_smooth(
    freq: np.ndarray, values: np.ndarray, fraction: int
) -> np.ndarray:
    """Smooth values in fractional-octave bands and interpolate back.

    ``fraction=12`` means 1/12-octave smoothing.  ``fraction <= 0`` returns
    the original data.
    """

    if fraction <= 0 or len(freq) < 3:
        return values.astype(float, copy=True)

    log_freq = np.log2(freq)
    bands = np.floor(log_freq * fraction).astype(np.int64)
    frame = pd.DataFrame({"band": bands, "log_freq": log_freq, "value": values})
    grouped = (
        frame.groupby("band", as_index=False)
        .agg(log_freq=("log_freq", "mean"), value=("value", "mean"))
        .sort_values("log_freq")
    )
    return np.interp(
        log_freq,
        grouped["log_freq"].to_numpy(dtype=float),
        grouped["value"].to_numpy(dtype=float),
    )


def smoothing_label(fraction: int) -> str:
    if fraction <= 0:
        return "raw (no smoothing)"
    return f"1/{fraction}-octave smoothing"


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []

    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _interpolate_crossing(
    freq: np.ndarray, values: np.ndarray, level: float, i: int, j: int
) -> float:
    """Interpolate the frequency where the curve crosses *level*."""

    y_i, y_j = float(values[i]), float(values[j])
    if not (math.isfinite(y_i) and math.isfinite(y_j)) or y_i == y_j:
        return float(freq[i])

    fraction = (level - y_i) / (y_j - y_i)
    fraction = min(max(fraction, 0.0), 1.0)
    return float(
        math.exp(
            math.log(float(freq[i]))
            + fraction * (math.log(float(freq[j])) - math.log(float(freq[i])))
        )
    )


def _merge_close_runs(
    runs: list[tuple[int, int]],
    freq: np.ndarray,
    max_gap_octaves: float,
) -> list[tuple[int, int]]:
    """Merge runs separated by less than *max_gap_octaves*.

    A narrow notch (for example a 1/5-octave dip) should not split a driver's
    passband into two separate regions for the -3 dB calculation.
    """

    if not runs:
        return []

    merged = [runs[0]]
    for start, end in runs[1:]:
        previous_start, previous_end = merged[-1]
        if freq[start] > freq[previous_end]:
            gap = math.log2(float(freq[start]) / float(freq[previous_end]))
        else:
            gap = 0.0

        if gap <= max_gap_octaves:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))

    return merged


def find_cutoffs(
    freq: np.ndarray,
    values: np.ndarray,
    reference: float,
    drop_db: float,
    max_gap_octaves: float = 0.25,
) -> tuple[float | None, float | None]:
    """Find the lower/upper frequency where the curve drops *drop_db*.

    The widest contiguous region above the threshold is used, which avoids
    picking a narrow noise spike outside the driver's passband.  Runs that are
    separated by a narrow notch are merged first.
    """

    threshold = reference - drop_db
    runs = _merge_close_runs(
        _contiguous_runs(values >= threshold),
        freq,
        max_gap_octaves=max_gap_octaves,
    )
    if not runs:
        return None, None

    def run_span(run: tuple[int, int]) -> float:
        start, end = run
        if end <= start:
            return 0.0
        return math.log10(float(freq[end]) / float(freq[start]))

    start, end = max(
        runs,
        key=lambda run: (run_span(run), float(np.mean(values[run[0] : run[1] + 1]))),
    )

    low: float | None = None
    high: float | None = None
    if start > 0:
        low = _interpolate_crossing(freq, values, threshold, start - 1, start)
    if end < len(freq) - 1:
        high = _interpolate_crossing(freq, values, threshold, end, end + 1)

    return low, high


def bandwidth_octaves(low: float | None, high: float | None) -> float | None:
    if low is None or high is None or low <= 0 or high <= low:
        return None
    return math.log2(high / low)


def analyze_spl(
    measurement: Measurement,
    band: tuple[float, float],
    smooth_fraction: int,
) -> dict[str, Any]:
    """Compute level statistics and -3/-6 dB limits for an SPL curve."""

    freq = measurement.freq
    spl = measurement.mag
    smoothed = fractional_octave_smooth(freq, spl, smooth_fraction)

    mask = (freq >= band[0]) & (freq <= band[1])
    if not mask.any():
        mask = np.ones_like(freq, dtype=bool)

    in_band = spl[mask]
    reference = float(np.median(in_band))
    f3_low, f3_high = find_cutoffs(freq, smoothed, reference, 3.0)
    f6_low, f6_high = find_cutoffs(freq, smoothed, reference, 6.0)

    return {
        "spl_points": len(freq),
        "spl_f_min_hz": float(freq.min()),
        "spl_f_max_hz": float(freq.max()),
        "band_low_hz": float(band[0]),
        "band_high_hz": float(band[1]),
        "spl_band_mean_db": float(np.mean(in_band)),
        "spl_band_median_db": reference,
        "spl_band_std_db": float(np.std(in_band, ddof=1))
        if len(in_band) > 1
        else 0.0,
        "spl_band_min_db": float(np.min(in_band)),
        "spl_band_max_db": float(np.max(in_band)),
        "spl_band_pp_db": float(np.max(in_band) - np.min(in_band)),
        "spl_ref_db": reference,
        "f3_low_hz": f3_low,
        "f3_high_hz": f3_high,
        "f6_low_hz": f6_low,
        "f6_high_hz": f6_high,
        "bandwidth_3db_oct": bandwidth_octaves(f3_low, f3_high),
        "bandwidth_6db_oct": bandwidth_octaves(f6_low, f6_high),
    }


def _crossing_left(
    freq: np.ndarray, values: np.ndarray, level: float, left: int, peak: int
) -> float | None:
    for i in range(peak - 1, left - 1, -1):
        if values[i] < level <= values[i + 1]:
            return _interpolate_crossing(freq, values, level, i, i + 1)
    return None


def _crossing_right(
    freq: np.ndarray, values: np.ndarray, level: float, peak: int, right: int
) -> float | None:
    for i in range(peak, right):
        if values[i] >= level > values[i + 1]:
            return _interpolate_crossing(freq, values, level, i, i + 1)
    return None


def estimate_le(freq: np.ndarray, impedance: np.ndarray, re: float) -> float | None:
    """Rough voice-coil inductance estimate from the 10-20 kHz region."""

    mask = (freq >= 10000.0) & (freq <= 20000.0)
    if mask.sum() < 3 or re <= 0:
        return None

    omega = 2.0 * math.pi * freq[mask]
    reactance = np.sqrt(np.maximum(impedance[mask] ** 2 - re**2, 0.0))
    inductance = float(np.median(reactance / omega))
    return inductance * 1000.0 if math.isfinite(inductance) else None


def analyze_impedance(
    measurement: Measurement,
    fs_max: float = 2000.0,
) -> dict[str, Any]:
    """Extract impedance features and rough Thiele-Small parameters.

    The Q values are estimated from the impedance curve using the classic
    half-power method.  They are only meaningful for free-air measurements;
    a driver measured in a box will show the box-loaded resonance instead.
    """

    freq = measurement.freq
    impedance = measurement.mag
    phase = measurement.phase

    min_index = int(np.argmin(impedance))
    re = float(impedance[min_index])
    re_freq = float(freq[min_index])

    z_max = float(np.max(impedance))
    z_max_freq = float(freq[int(np.argmax(impedance))])

    prominence = max(0.05 * (z_max - float(np.min(impedance))), 0.05)
    peaks, _ = find_peaks(impedance, prominence=prominence)
    low_peaks = [int(p) for p in peaks if freq[p] <= fs_max]

    if low_peaks:
        fs_index = max(low_peaks, key=lambda i: impedance[i])
    else:
        below = np.flatnonzero(freq <= fs_max)
        if below.size:
            fs_index = int(below[np.argmax(impedance[below])])
        else:
            fs_index = int(np.argmax(impedance))

    fs = float(freq[fs_index])
    z_fs = float(impedance[fs_index])

    qms = qes = qts = None
    if z_fs > re * 1.02:
        ratio = z_fs / re
        half_level = math.sqrt(re * z_fs)
        # The nearest half-level crossing on either side of the resonance is
        # used.  Searching the whole curve (instead of stopping at a local
        # minimum) is important because measurement wiggles can create tiny
        # local minima that never fall below the half-power level.
        f1 = _crossing_left(freq, impedance, half_level, 0, fs_index)
        f2 = _crossing_right(
            freq, impedance, half_level, fs_index, len(freq) - 1
        )

        if f1 is not None and f2 is not None and f2 > f1:
            qms = fs * math.sqrt(ratio - 1.0) / (f2 - f1)
            qes = qms / (ratio - 1.0)
            if qms > 0 and qes > 0:
                qts = (qes * qms) / (qes + qms)
            else:
                qms = qes = qts = None

    z_at_1k = float(
        np.interp(
            math.log10(1000.0),
            np.log10(freq),
            impedance,
        )
    )

    return {
        "imp_points": len(freq),
        "imp_f_min_hz": float(freq.min()),
        "imp_f_max_hz": float(freq.max()),
        "z_min_ohm": re,
        "z_min_freq_hz": re_freq,
        "z_max_ohm": z_max,
        "z_max_freq_hz": z_max_freq,
        "z_1k_ohm": z_at_1k,
        "z_phase_min_deg": float(np.min(phase)),
        "z_phase_max_deg": float(np.max(phase)),
        "fs_hz": fs,
        "z_fs_ohm": z_fs,
        "re_ohm": re,
        "re_freq_hz": re_freq,
        "qms": qms,
        "qes": qes,
        "qts": qts,
        "le_mh": estimate_le(freq, impedance, re),
    }


def analyze_all(
    drivers: dict[str, dict[str, Measurement]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, dict[str, Any]]]]:
    """Analyse every driver and return SPL/Impedance tables plus details."""

    spl_rows: list[dict[str, Any]] = []
    impedance_rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, dict[str, Any]]] = {}

    override = tuple(args.band) if args.band else None

    for driver in ordered_drivers(drivers):
        details[driver] = {}
        data = drivers[driver]

        if "spl" in data:
            band = band_for_driver(driver, override)
            result = analyze_spl(data["spl"], band, args.smooth)
            details[driver]["spl"] = result
            spl_rows.append({"driver": driver, **result})

        if "impedance" in data:
            result = analyze_impedance(data["impedance"], fs_max=args.fs_max)
            details[driver]["impedance"] = result
            impedance_rows.append({"driver": driver, **result})

    return (
        pd.DataFrame(spl_rows),
        pd.DataFrame(impedance_rows),
        details,
    )


def build_summary(
    drivers: dict[str, dict[str, Measurement]],
    spl_df: pd.DataFrame,
    impedance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine file metadata and analysis results into one summary table."""

    rows: list[dict[str, Any]] = []
    for driver in ordered_drivers(drivers):
        data = drivers[driver]
        row: dict[str, Any] = {
            "driver": driver,
            "label": driver_label(driver),
            "spl_file": data["spl"].path.name if "spl" in data else None,
            "impedance_file": data["impedance"].path.name
            if "impedance" in data
            else None,
        }

        if not spl_df.empty:
            match = spl_df[spl_df["driver"] == driver]
            if not match.empty:
                row.update(
                    {
                        key: value
                        for key, value in match.iloc[0].items()
                        if key != "driver"
                    }
                )

        if not impedance_df.empty:
            match = impedance_df[impedance_df["driver"] == driver]
            if not match.empty:
                row.update(
                    {
                        key: value
                        for key, value in match.iloc[0].items()
                        if key != "driver"
                    }
                )

        rows.append(row)

    return pd.DataFrame(rows)


def third_octave_levels(
    drivers: dict[str, dict[str, Measurement]],
) -> pd.DataFrame:
    """Average SPL in standard third-octave bands for each driver."""

    rows: list[dict[str, Any]] = []
    for center in THIRD_OCTAVE_CENTERS:
        low = center / (2.0 ** (1.0 / 6.0))
        high = center * (2.0 ** (1.0 / 6.0))
        row: dict[str, Any] = {"freq_hz": center}
        for driver in ordered_drivers(drivers):
            measurement = drivers[driver].get("spl")
            if measurement is None:
                continue
            selected = measurement.mag[
                (measurement.freq >= low) & (measurement.freq <= high)
            ]
            row[driver] = float(np.mean(selected)) if selected.size else None
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _format_frequency(value: float, _position: int | None = None) -> str:
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def style_frequency_axis(ax: plt.Axes, f_min: float, f_max: float) -> None:
    ax.set_xscale("log")
    ax.set_xlim(f_min, f_max)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    ax.xaxis.set_major_formatter(FuncFormatter(_format_frequency))
    ax.xaxis.set_minor_formatter(NullFormatter())


def frequency_limits(
    measurements: Iterable[Measurement | None],
    floor_hz: float = 10.0,
    ceiling_hz: float = 20000.0,
) -> tuple[float, float]:
    freqs = [
        measurement.freq
        for measurement in measurements
        if measurement is not None and len(measurement.freq)
    ]
    if not freqs:
        return floor_hz, ceiling_hz

    lowest = min(float(np.min(values)) for values in freqs)
    highest = max(float(np.max(values)) for values in freqs)
    f_min = max(floor_hz, 10.0 ** math.floor(math.log10(lowest)))
    f_max = min(ceiling_hz, 10.0 ** math.ceil(math.log10(highest)))
    if f_max <= f_min:
        f_max = f_min * 10.0
    return f_min, f_max


def _finish_figure(fig: Figure, path: Path, args: argparse.Namespace) -> None:
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)


def plot_spl_all(
    drivers: dict[str, dict[str, Measurement]],
    details: dict[str, dict[str, dict[str, Any]]],
    args: argparse.Namespace,
    output_dir: Path,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    plotted = False

    for driver in ordered_drivers(drivers):
        measurement = drivers[driver].get("spl")
        if measurement is None:
            continue
        smoothed = fractional_octave_smooth(
            measurement.freq, measurement.mag, args.smooth
        )
        ax.plot(
            measurement.freq,
            smoothed,
            color=driver_color(driver),
            linewidth=1.8,
            label=driver_label(driver),
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    style_frequency_axis(
        ax,
        *frequency_limits(
            [drivers[d].get("spl") for d in drivers],
            floor_hz=10.0,
        ),
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SPL (dB, as measured)")
    ax.set_title(f"Measured SPL - driver comparison ({smoothing_label(args.smooth)})")
    ax.legend(loc="best")
    path = output_dir / "01_spl_magnitude.png"
    _finish_figure(fig, path, args)
    return path


def plot_spl_phase_all(
    drivers: dict[str, dict[str, Measurement]],
    args: argparse.Namespace,
    output_dir: Path,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    plotted = False

    for driver in ordered_drivers(drivers):
        measurement = drivers[driver].get("spl")
        if measurement is None:
            continue
        ax.plot(
            measurement.freq,
            measurement.phase,
            color=driver_color(driver),
            linewidth=1.1,
            label=driver_label(driver),
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    style_frequency_axis(
        ax,
        *frequency_limits(
            [drivers[d].get("spl") for d in drivers],
            floor_hz=10.0,
        ),
    )
    ax.axhline(0.0, color="0.45", linewidth=0.8, zorder=0)
    ax.set_ylim(-180.0, 180.0)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (degrees)")
    ax.set_title("Measured SPL phase (wrapped, as exported)")
    ax.legend(loc="best")
    path = output_dir / "02_spl_phase.png"
    _finish_figure(fig, path, args)
    return path


def plot_impedance_all(
    drivers: dict[str, dict[str, Measurement]],
    details: dict[str, dict[str, dict[str, Any]]],
    args: argparse.Namespace,
    output_dir: Path,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    plotted = False
    maximum = 0.0

    for driver in ordered_drivers(drivers):
        measurement = drivers[driver].get("impedance")
        if measurement is None:
            continue
        analysis = details.get(driver, {}).get("impedance", {})
        color = driver_color(driver)
        ax.plot(
            measurement.freq,
            measurement.mag,
            color=color,
            linewidth=1.6,
            label=driver_label(driver),
        )
        maximum = max(maximum, float(np.max(measurement.mag)))

        if analysis:
            ax.scatter(
                [analysis["fs_hz"]],
                [analysis["z_fs_ohm"]],
                s=48,
                facecolors="white",
                edgecolors=color,
                linewidths=1.6,
                zorder=5,
            )
            ax.annotate(
                f"Fs {analysis['fs_hz']:.1f} Hz",
                (analysis["fs_hz"], analysis["z_fs_ohm"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    style_frequency_axis(
        ax,
        *frequency_limits(
            [drivers[d].get("impedance") for d in drivers],
            floor_hz=5.0,
        ),
    )
    ax.set_ylim(0.0, maximum * 1.12 if maximum > 0 else None)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Impedance |Z| (ohm)")
    ax.set_title("Measured impedance magnitude (circles mark estimated Fs)")
    ax.legend(loc="upper left")
    path = output_dir / "03_impedance_magnitude.png"
    _finish_figure(fig, path, args)
    return path


def plot_impedance_phase_all(
    drivers: dict[str, dict[str, Measurement]],
    args: argparse.Namespace,
    output_dir: Path,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    plotted = False

    for driver in ordered_drivers(drivers):
        measurement = drivers[driver].get("impedance")
        if measurement is None:
            continue
        ax.plot(
            measurement.freq,
            measurement.phase,
            color=driver_color(driver),
            linewidth=1.1,
            label=driver_label(driver),
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    style_frequency_axis(
        ax,
        *frequency_limits(
            [drivers[d].get("impedance") for d in drivers],
            floor_hz=5.0,
        ),
    )
    ax.axhline(0.0, color="0.45", linewidth=0.8, zorder=0)
    ax.set_ylim(-180.0, 180.0)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (degrees)")
    ax.set_title("Measured impedance phase (wrapped, as exported)")
    ax.legend(loc="best")
    path = output_dir / "04_impedance_phase.png"
    _finish_figure(fig, path, args)
    return path


def plot_normalized_spl(
    drivers: dict[str, dict[str, Measurement]],
    details: dict[str, dict[str, dict[str, Any]]],
    args: argparse.Namespace,
    output_dir: Path,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(11, 6.0))
    plotted = False

    for driver in ordered_drivers(drivers):
        measurement = drivers[driver].get("spl")
        analysis = details.get(driver, {}).get("spl")
        if measurement is None or not analysis:
            continue
        smoothed = fractional_octave_smooth(
            measurement.freq, measurement.mag, args.smooth
        )
        ax.plot(
            measurement.freq,
            smoothed - analysis["spl_ref_db"],
            color=driver_color(driver),
            linewidth=1.7,
            label=driver_label(driver),
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    style_frequency_axis(
        ax,
        *frequency_limits(
            [drivers[d].get("spl") for d in drivers],
            floor_hz=10.0,
        ),
    )
    ax.axhline(0.0, color="0.3", linewidth=1.0)
    ax.axhline(3.0, color="0.5", linewidth=0.8, linestyle="--")
    ax.axhline(-3.0, color="0.5", linewidth=0.8, linestyle="--")
    ax.set_ylim(-30.0, 15.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Relative SPL (dB)")
    ax.set_title("SPL normalised to each driver's passband median (+/-3 dB dashed)")
    ax.legend(loc="lower left")
    path = output_dir / "05_spl_normalized.png"
    _finish_figure(fig, path, args)
    return path


def plot_driver_detail(
    driver: str,
    data: dict[str, Measurement],
    details: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> Path | None:
    if "spl" not in data or "impedance" not in data:
        return None

    spl_measurement = data["spl"]
    impedance_measurement = data["impedance"]
    spl_analysis = details.get("spl", {})
    impedance_analysis = details.get("impedance", {})
    color = driver_color(driver)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), constrained_layout=True)
    fig.suptitle(f"{driver_label(driver)} - measurement overview", fontsize=14)

    # SPL magnitude
    ax = axes[0, 0]
    smoothed = fractional_octave_smooth(
        spl_measurement.freq, spl_measurement.mag, args.smooth
    )
    ax.plot(spl_measurement.freq, smoothed, color=color, linewidth=1.8)
    if spl_analysis:
        ax.axvspan(
            spl_analysis["band_low_hz"],
            spl_analysis["band_high_hz"],
            color=color,
            alpha=0.08,
            linewidth=0,
        )
        for cutoff in (spl_analysis.get("f3_low_hz"), spl_analysis.get("f3_high_hz")):
            if cutoff is not None:
                ax.axvline(cutoff, color=color, linewidth=0.9, linestyle="--")
    style_frequency_axis(
        ax,
        *frequency_limits([spl_measurement], floor_hz=10.0),
    )
    ax.set_title("SPL magnitude")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SPL (dB)")

    # SPL phase
    ax = axes[0, 1]
    ax.plot(spl_measurement.freq, spl_measurement.phase, color=color, linewidth=1.0)
    style_frequency_axis(
        ax,
        *frequency_limits([spl_measurement], floor_hz=10.0),
    )
    ax.axhline(0.0, color="0.45", linewidth=0.8, zorder=0)
    ax.set_ylim(-180.0, 180.0)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_title("SPL phase")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (degrees)")

    # Impedance magnitude
    ax = axes[1, 0]
    ax.plot(
        impedance_measurement.freq,
        impedance_measurement.mag,
        color=color,
        linewidth=1.6,
    )
    if impedance_analysis:
        ax.scatter(
            [impedance_analysis["fs_hz"]],
            [impedance_analysis["z_fs_ohm"]],
            s=55,
            facecolors="white",
            edgecolors=color,
            linewidths=1.7,
            zorder=5,
        )
        ax.annotate(
            f"Fs {impedance_analysis['fs_hz']:.1f} Hz\n"
            f"Re {impedance_analysis['re_ohm']:.2f} ohm",
            (impedance_analysis["fs_hz"], impedance_analysis["z_fs_ohm"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    style_frequency_axis(
        ax,
        *frequency_limits([impedance_measurement], floor_hz=5.0),
    )
    ax.set_ylim(
        0.0,
        float(np.max(impedance_measurement.mag)) * 1.12,
    )
    ax.set_title("Impedance magnitude")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|Z| (ohm)")

    # Impedance phase
    ax = axes[1, 1]
    ax.plot(
        impedance_measurement.freq,
        impedance_measurement.phase,
        color=color,
        linewidth=1.0,
    )
    style_frequency_axis(
        ax,
        *frequency_limits([impedance_measurement], floor_hz=5.0),
    )
    ax.axhline(0.0, color="0.45", linewidth=0.8, zorder=0)
    ax.set_ylim(-180.0, 180.0)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_title("Impedance phase")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (degrees)")

    path = output_dir / f"driver_{driver}.png"
    _finish_figure(fig, path, args)
    return path


def write_interactive_html(
    drivers: dict[str, dict[str, Measurement]],
    details: dict[str, dict[str, dict[str, Any]]],
    args: argparse.Namespace,
    output_dir: Path,
) -> Path | None:
    if not HAS_PLOTLY:
        return None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "SPL magnitude",
            "SPL phase",
            "Impedance magnitude",
            "Impedance phase",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
    )

    for driver in ordered_drivers(drivers):
        data = drivers[driver]
        color = driver_color(driver)
        label = driver_label(driver)

        if "spl" in data:
            measurement = data["spl"]
            smoothed = fractional_octave_smooth(
                measurement.freq, measurement.mag, args.smooth
            )
            fig.add_trace(
                go.Scatter(
                    x=measurement.freq,
                    y=smoothed,
                    name=label,
                    legendgroup=driver,
                    line={"color": color, "width": 2},
                    hovertemplate="%{x:.1f} Hz<br>%{y:.2f} dB<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=measurement.freq,
                    y=measurement.phase,
                    name=label,
                    legendgroup=driver,
                    showlegend=False,
                    line={"color": color, "width": 1},
                    hovertemplate="%{x:.1f} Hz<br>%{y:.1f} deg<extra></extra>",
                ),
                row=1,
                col=2,
            )

        if "impedance" in data:
            measurement = data["impedance"]
            fig.add_trace(
                go.Scatter(
                    x=measurement.freq,
                    y=measurement.mag,
                    name=label,
                    legendgroup=driver,
                    showlegend=False,
                    line={"color": color, "width": 1.6},
                    hovertemplate="%{x:.1f} Hz<br>%{y:.2f} ohm<extra></extra>",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=measurement.freq,
                    y=measurement.phase,
                    name=label,
                    legendgroup=driver,
                    showlegend=False,
                    line={"color": color, "width": 1},
                    hovertemplate="%{x:.1f} Hz<br>%{y:.1f} deg<extra></extra>",
                ),
                row=2,
                col=2,
            )

    for row in (1, 2):
        for column in (1, 2):
            fig.update_xaxes(
                type="log",
                title_text="Frequency (Hz)",
                row=row,
                col=column,
            )

    fig.update_yaxes(title_text="SPL (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (deg)", range=[-180, 180], row=1, col=2)
    fig.update_yaxes(title_text="|Z| (ohm)", row=2, col=1)
    fig.update_yaxes(title_text="Phase (deg)", range=[-180, 180], row=2, col=2)
    fig.update_layout(
        height=850,
        title_text="Loudspeaker driver measurements",
        legend_title_text="Driver",
        hovermode="x unified",
        template="plotly_white",
    )

    path = output_dir / "interactive_measurements.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# ---------------------------------------------------------------------------
# Reports and exports
# ---------------------------------------------------------------------------


def write_combined_measurements(
    drivers: dict[str, dict[str, Measurement]],
    output_dir: Path,
) -> Path:
    frames: list[pd.DataFrame] = []
    for driver in ordered_drivers(drivers):
        for kind, measurement in drivers[driver].items():
            frames.append(
                pd.DataFrame(
                    {
                        "driver": driver,
                        "measurement": kind,
                        "freq_hz": measurement.freq,
                        "magnitude": measurement.mag,
                        "phase_deg": measurement.phase,
                        "source_file": measurement.path.name,
                    }
                )
            )

    path = output_dir / "measurements_combined.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def write_excel(
    output_dir: Path,
    summary: pd.DataFrame,
    spl_df: pd.DataFrame,
    impedance_df: pd.DataFrame,
    third_octave: pd.DataFrame,
) -> Path | None:
    try:
        path = output_dir / "summary.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            spl_df.to_excel(writer, sheet_name="SPL", index=False)
            impedance_df.to_excel(writer, sheet_name="Impedance", index=False)
            third_octave.to_excel(writer, sheet_name="Third octave", index=False)
        return path
    except Exception as exc:  # optional dependency / file lock
        print(f"Warning: could not write Excel summary: {exc}", file=sys.stderr)
        return None


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data._"
    clean = frame.round(2).astype(object).where(pd.notna(frame), None)
    try:
        return clean.to_markdown(index=False, missingval="-")
    except TypeError:  # older pandas without the missingval argument
        return clean.to_markdown(index=False)


def write_report(
    output_dir: Path,
    data_dir: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    spl_df: pd.DataFrame,
    impedance_df: pd.DataFrame,
    plot_paths: list[Path],
    html_path: Path | None,
    generated: datetime,
) -> Path:
    file_columns = [
        "driver",
        "label",
        "spl_file",
        "impedance_file",
        "spl_points",
        "imp_points",
    ]
    file_columns = [column for column in file_columns if column in summary.columns]

    spl_columns = [
        "driver",
        "band_low_hz",
        "band_high_hz",
        "spl_band_mean_db",
        "spl_band_std_db",
        "spl_band_min_db",
        "spl_band_max_db",
        "f3_low_hz",
        "f3_high_hz",
        "f6_low_hz",
        "f6_high_hz",
    ]
    spl_columns = [column for column in spl_columns if column in spl_df.columns]

    impedance_columns = [
        "driver",
        "re_ohm",
        "fs_hz",
        "z_fs_ohm",
        "z_min_ohm",
        "z_max_ohm",
        "z_max_freq_hz",
        "z_1k_ohm",
        "qms",
        "qes",
        "qts",
        "le_mh",
    ]
    impedance_columns = [
        column for column in impedance_columns if column in impedance_df.columns
    ]

    lines = [
        "# Loudspeaker driver measurement analysis",
        "",
        f"Generated: {generated:%Y-%m-%d %H:%M:%S}",
        "",
        f"- Data directory: `{data_dir}`",
        f"- Analysis band override: "
        + (
            f"{args.band[0]:g} - {args.band[1]:g} Hz"
            if args.band
            else "per-driver defaults"
        ),
        f"- Smoothing: {smoothing_label(args.smooth)}",
        "",
        "## Measurement files",
        "",
        _markdown_table(summary[file_columns]),
        "",
        "## SPL summary",
        "",
        _markdown_table(spl_df[spl_columns]),
        "",
        "## Impedance summary",
        "",
        _markdown_table(impedance_df[impedance_columns]),
        "",
        "## Notes",
        "",
        "- SPL values are plotted exactly as exported. If the measurement was "
        "not calibrated, the levels are relative rather than absolute dB SPL.",
        "- The -3 dB / -6 dB limits are relative to the median level in each "
        "driver's analysis band. The widest contiguous region above the "
        "threshold is used, and narrow notches are merged. A missing upper "
        "limit means the response stays within the threshold beyond the "
        "measured range.",
        "- Fs is the largest impedance peak below the `--fs-max` limit "
        "(default 2000 Hz). Additional low-frequency peaks may be box or "
        "measurement artefacts.",
        "- Fs, Re and the Q values are rough estimates from the impedance "
        "curve. They are only meaningful for free-air measurements; a driver "
        "measured in a box shows the box-loaded resonance.",
        "- `Re` is taken as the minimum impedance magnitude in the measured "
        "range. `Le` is a rough estimate from the 10-20 kHz region.",
        "",
        "## Plots",
        "",
    ]

    for path in plot_paths:
        if path is None:
            continue
        title = PLOT_TITLES.get(path.stem, path.stem.replace("_", " ").title())
        lines.extend([f"### {title}", "", f"![{title}]({path.name})", ""])

    if html_path is not None:
        lines.extend(
            [
                "## Interactive report",
                "",
                f"[Open the interactive Plotly report]({html_path.name})",
                "",
            ]
        )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse and plot loudspeaker driver measurements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="Mätdata",
        help="Directory containing the measurement files",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis",
        help="Directory for plots and reports",
    )
    parser.add_argument(
        "--band",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Override the analysis band for every driver (Hz)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=12,
        metavar="N",
        help="1/N-octave smoothing for plots; 0 disables smoothing",
    )
    parser.add_argument(
        "--fs-max",
        type=float,
        default=2000.0,
        metavar="HZ",
        help="Upper frequency for the resonance (Fs) search",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="DPI for saved PNG plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show each plot interactively (requires a GUI backend)",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Do not write the interactive Plotly HTML report",
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Do not write summary.xlsx",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generated = datetime.now()

    if args.show and matplotlib.get_backend().lower() == "agg":
        print(
            "Warning: --show was requested but the Matplotlib backend is "
            "'Agg' (no display). Plots will only be saved to disk.",
            file=sys.stderr,
        )

    try:
        data_dir = resolve_data_dir(args.data_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    drivers, errors = discover_measurements(data_dir)
    if not drivers:
        print(f"Error: no measurement files found in {data_dir}", file=sys.stderr)
        return 1

    for path, error in errors:
        print(f"Warning: {path.name}: {error}", file=sys.stderr)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    spl_df, impedance_df, details = analyze_all(drivers, args)
    summary = build_summary(drivers, spl_df, impedance_df)
    third_octave = third_octave_levels(drivers)

    summary.to_csv(output_dir / "summary.csv", index=False)
    third_octave.to_csv(output_dir / "third_octave_levels.csv", index=False)
    write_combined_measurements(drivers, output_dir)

    plot_paths: list[Path | None] = [
        plot_spl_all(drivers, details, args, output_dir),
        plot_spl_phase_all(drivers, args, output_dir),
        plot_impedance_all(drivers, details, args, output_dir),
        plot_impedance_phase_all(drivers, args, output_dir),
        plot_normalized_spl(drivers, details, args, output_dir),
    ]

    for driver in ordered_drivers(drivers):
        plot_paths.append(
            plot_driver_detail(
                driver,
                drivers[driver],
                details[driver],
                args,
                output_dir,
            )
        )

    html_path: Path | None = None
    if not args.no_html:
        html_path = write_interactive_html(drivers, details, args, output_dir)

    if not args.no_excel:
        write_excel(output_dir, summary, spl_df, impedance_df, third_octave)

    report_path = write_report(
        output_dir,
        data_dir,
        args,
        summary,
        spl_df,
        impedance_df,
        [path for path in plot_paths if path is not None],
        html_path,
        generated,
    )

    print(f"Data directory: {data_dir}")
    print(f"Drivers found: {', '.join(ordered_drivers(drivers))}")
    print(f"Output directory: {output_dir}")
    if not spl_df.empty:
        print("\nSPL summary:")
        print(spl_df.round(2).to_string(index=False))
    if not impedance_df.empty:
        print("\nImpedance summary:")
        print(impedance_df.round(2).to_string(index=False))
    print(f"\nReport: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
