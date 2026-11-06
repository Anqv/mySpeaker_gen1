#!/usr/bin/env python3
"""Interactive checkbox viewer for loudspeaker measurements.

The upper panel shows SPL measurements, the lower panel shows impedance
measurements.  Tick the files in the left-hand list to show or hide them.

Run with:

    source .venv/bin/activate
    python plot_speaker_gui.py
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction, QKeySequence
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover - only hit when the GUI extra is absent
    print(
        "PySide6 is required for the graphical viewer.\n"
        "Install it with:  .venv/bin/pip install PySide6",
        file=sys.stderr,
    )
    raise SystemExit(1)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from analyze_speakers import (
    Measurement,
    discover_measurements,
    driver_color,
    driver_label,
    fractional_octave_smooth,
    frequency_limits,
    ordered_drivers,
    resolve_data_dir,
    smoothing_label,
    style_frequency_axis,
)


@dataclass
class FileEntry:
    measurement: Measurement
    checkbox: QCheckBox


class SpeakerViewer(QMainWindow):
    """Two stacked plots with a checkbox for every measurement file."""

    def __init__(self, data_dir: Path, output_dir: Path):
        super().__init__()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.entries: list[FileEntry] = []
        self._updating = False

        self.setWindowTitle("Speaker measurement viewer")
        self.resize(1420, 900)
        self._build_ui()
        self.reload_data(select_all=True)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        controls = QWidget()
        controls.setMinimumWidth(290)
        controls.setMaximumWidth(430)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(8)

        heading = QLabel("Measurement files")
        heading_font = heading.font()
        heading_font.setPointSize(13)
        heading_font.setBold(True)
        heading.setFont(heading_font)
        controls_layout.addWidget(heading)

        self.spl_group = QGroupBox("SPL (dB)")
        self.spl_layout = QVBoxLayout(self.spl_group)
        self.imp_group = QGroupBox("Impedance (ohm)")
        self.imp_layout = QVBoxLayout(self.imp_group)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(self.spl_group)
        scroll_layout.addWidget(self.imp_group)
        scroll_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_content)
        controls_layout.addWidget(scroll, 1)

        button_row = QHBoxLayout()
        self.all_button = QPushButton("All")
        self.none_button = QPushButton("None")
        self.reload_button = QPushButton("Reload")
        self.all_button.setToolTip("Select every measurement file")
        self.none_button.setToolTip("Clear all selections")
        self.reload_button.setToolTip("Re-read the measurement directory")
        button_row.addWidget(self.all_button)
        button_row.addWidget(self.none_button)
        button_row.addWidget(self.reload_button)
        controls_layout.addLayout(button_row)

        options = QGroupBox("Options")
        options_layout = QVBoxLayout(options)

        self.phase_check = QCheckBox("Show phase (dashed, right axis)")
        self.phase_check.setChecked(False)
        options_layout.addWidget(self.phase_check)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("SPL smoothing:"))
        self.smooth_combo = QComboBox()
        for label, value in (
            ("Off (raw)", 0),
            ("1/3 octave", 3),
            ("1/6 octave", 6),
            ("1/12 octave", 12),
            ("1/24 octave", 24),
            ("1/48 octave", 48),
        ):
            self.smooth_combo.addItem(label, value)
        self.smooth_combo.setCurrentIndex(3)  # 1/12 octave
        smooth_row.addWidget(self.smooth_combo, 1)
        options_layout.addLayout(smooth_row)

        self.save_button = QPushButton("Save plots...")
        self.save_button.setToolTip("Save the current two-panel view")
        options_layout.addWidget(self.save_button)

        controls_layout.addWidget(options)

        self.figure = Figure(figsize=(9.0, 7.0), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        splitter.addWidget(controls)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1100])

        self.all_button.clicked.connect(lambda: self.set_all(True))
        self.none_button.clicked.connect(lambda: self.set_all(False))
        self.reload_button.clicked.connect(
            lambda: self.reload_data(select_all=False)
        )
        self.phase_check.toggled.connect(self.update_plots)
        self.smooth_combo.currentIndexChanged.connect(self.update_plots)
        self.save_button.clicked.connect(self.save_plots)

        self._build_menu()
        self.statusBar().showMessage("Ready")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        reload_action = QAction("Reload measurements", self)
        reload_action.setShortcut(QKeySequence.Refresh)
        reload_action.triggered.connect(
            lambda: self.reload_data(select_all=False)
        )
        file_menu.addAction(reload_action)

        save_action = QAction("Save plots...", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_plots)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _clear_layout(layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                child = item.layout()
                if child is not None:
                    SpeakerViewer._clear_layout(child)  # type: ignore[arg-type]

    def reload_data(self, select_all: bool = False) -> None:
        checked_names = {
            entry.measurement.path.name
            for entry in self.entries
            if entry.checkbox.isChecked()
        }

        self._clear_layout(self.spl_layout)
        self._clear_layout(self.imp_layout)
        self.entries.clear()

        try:
            drivers, errors = discover_measurements(self.data_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Could not read data", str(exc))
            return

        for path, error in errors:
            print(f"Warning: {path.name}: {error}", file=sys.stderr)

        spl_measurements: list[Measurement] = []
        impedance_measurements: list[Measurement] = []

        for driver in ordered_drivers(drivers):
            for kind, target in (
                ("spl", spl_measurements),
                ("impedance", impedance_measurements),
            ):
                measurement = drivers[driver].get(kind)
                if measurement is not None:
                    target.append(measurement)

        for measurement in spl_measurements:
            self._add_checkbox(
                self.spl_layout, measurement, checked_names, select_all
            )
        for measurement in impedance_measurements:
            self._add_checkbox(
                self.imp_layout, measurement, checked_names, select_all
            )

        if not spl_measurements:
            self.spl_layout.addWidget(QLabel("No SPL files found"))
        if not impedance_measurements:
            self.imp_layout.addWidget(QLabel("No impedance files found"))

        self.update_plots()

    def _add_checkbox(
        self,
        layout: QVBoxLayout,
        measurement: Measurement,
        checked_names: set[str],
        select_all: bool,
    ) -> None:
        checkbox = QCheckBox(
            f"{measurement.path.name}  -  {driver_label(measurement.driver)}"
        )
        checkbox.setChecked(select_all or measurement.path.name in checked_names)
        checkbox.setToolTip(str(measurement.path))
        checkbox.setStyleSheet(
            f"QCheckBox {{ color: {driver_color(measurement.driver)}; }}"
        )
        checkbox.toggled.connect(self.update_plots)
        layout.addWidget(checkbox)
        self.entries.append(FileEntry(measurement, checkbox))

    def set_all(self, checked: bool) -> None:
        for entry in self.entries:
            entry.checkbox.blockSignals(True)
            entry.checkbox.setChecked(checked)
            entry.checkbox.blockSignals(False)
        self.update_plots()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _selected_measurements(self) -> tuple[list[Measurement], list[Measurement]]:
        spl: list[Measurement] = []
        impedance: list[Measurement] = []
        for entry in self.entries:
            if not entry.checkbox.isChecked():
                continue
            if entry.measurement.kind == "spl":
                spl.append(entry.measurement)
            else:
                impedance.append(entry.measurement)
        return spl, impedance

    def _frequency_limits(self, measurements: list[Measurement]) -> tuple[float, float]:
        if not measurements:
            return 10.0, 20000.0
        return frequency_limits(measurements, floor_hz=5.0)

    def update_plots(self) -> None:
        if self._updating:
            return

        self._updating = True
        try:
            spl_measurements, impedance_measurements = self._selected_measurements()
            smooth_fraction = int(self.smooth_combo.currentData())
            show_phase = self.phase_check.isChecked()

            self.figure.clear()
            ax_spl = self.figure.add_subplot(2, 1, 1)
            ax_impedance = self.figure.add_subplot(2, 1, 2, sharex=ax_spl)
            ax_spl_phase = ax_spl.twinx()
            ax_impedance_phase = ax_impedance.twinx()

            f_min, f_max = self._frequency_limits(
                spl_measurements + impedance_measurements
            )

            for axis in (ax_spl, ax_impedance):
                style_frequency_axis(axis, f_min, f_max)
                axis.grid(True, which="major", alpha=0.30)
                axis.grid(True, which="minor", alpha=0.12)
                axis.tick_params(labelsize=9)

            ax_spl.tick_params(labelbottom=False)
            ax_impedance.set_xlabel("Frequency (Hz)", fontsize=10)

            # Upper panel: SPL
            for measurement in spl_measurements:
                color = driver_color(measurement.driver)
                smoothed = fractional_octave_smooth(
                    measurement.freq,
                    measurement.mag,
                    smooth_fraction,
                )
                ax_spl.plot(
                    measurement.freq,
                    smoothed,
                    color=color,
                    linewidth=1.7,
                    label=driver_label(measurement.driver),
                )
                if show_phase:
                    ax_spl_phase.plot(
                        measurement.freq,
                        measurement.phase,
                        color=color,
                        linewidth=0.9,
                        linestyle="--",
                        alpha=0.55,
                    )

            ax_spl.set_ylabel("SPL (dB)", fontsize=10)
            if spl_measurements:
                ax_spl.set_title(
                    f"SPL magnitude  -  {len(spl_measurements)} selected  "
                    f"({smoothing_label(smooth_fraction)})",
                    loc="left",
                    fontsize=11,
                    fontweight="bold",
                )
                ax_spl.legend(loc="best", fontsize=8, framealpha=0.90)
            else:
                ax_spl.set_title(
                    "SPL magnitude",
                    loc="left",
                    fontsize=11,
                    fontweight="bold",
                )
                ax_spl.text(
                    0.5,
                    0.5,
                    "No SPL files selected",
                    transform=ax_spl.transAxes,
                    ha="center",
                    va="center",
                    color="0.55",
                    fontsize=11,
                )

            # Lower panel: impedance
            for measurement in impedance_measurements:
                color = driver_color(measurement.driver)
                ax_impedance.plot(
                    measurement.freq,
                    measurement.mag,
                    color=color,
                    linewidth=1.7,
                    label=driver_label(measurement.driver),
                )
                if show_phase:
                    ax_impedance_phase.plot(
                        measurement.freq,
                        measurement.phase,
                        color=color,
                        linewidth=0.9,
                        linestyle="--",
                        alpha=0.55,
                    )

            ax_impedance.set_ylabel("Impedance |Z| (ohm)", fontsize=10)
            if impedance_measurements:
                maximum = max(
                    float(measurement.mag.max())
                    for measurement in impedance_measurements
                )
                ax_impedance.set_ylim(0.0, maximum * 1.12)
                ax_impedance.set_title(
                    f"Impedance magnitude  -  {len(impedance_measurements)} selected",
                    loc="left",
                    fontsize=11,
                    fontweight="bold",
                )
                ax_impedance.legend(loc="best", fontsize=8, framealpha=0.90)
            else:
                ax_impedance.set_title(
                    "Impedance magnitude",
                    loc="left",
                    fontsize=11,
                    fontweight="bold",
                )
                ax_impedance.text(
                    0.5,
                    0.5,
                    "No impedance files selected",
                    transform=ax_impedance.transAxes,
                    ha="center",
                    va="center",
                    color="0.55",
                    fontsize=11,
                )

            # Phase axes
            if show_phase:
                for phase_axis in (ax_spl_phase, ax_impedance_phase):
                    phase_axis.set_ylim(-180.0, 180.0)
                    phase_axis.set_yticks([-180, -90, 0, 90, 180])
                    phase_axis.set_ylabel(
                        "Phase (deg)", color="0.40", fontsize=9
                    )
                    phase_axis.tick_params(
                        axis="y", labelsize=8, colors="0.40"
                    )
                    phase_axis.spines["right"].set_color("0.65")
            else:
                ax_spl_phase.set_visible(False)
                ax_impedance_phase.set_visible(False)

            self.canvas.draw_idle()
            self.statusBar().showMessage(
                f"{len(spl_measurements)} SPL and "
                f"{len(impedance_measurements)} impedance file(s) selected"
            )
        finally:
            self._updating = False

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_plots(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        default_path = self.output_dir / "selected_measurements.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save plots",
            str(default_path),
            "PNG image (*.png);;PDF document (*.pdf);;SVG image (*.svg)",
        )
        if not path:
            return

        try:
            self.figure.savefig(path, dpi=200, bbox_inches="tight")
        except Exception as exc:
            QMessageBox.critical(self, "Could not save plots", str(exc))
            return

        self.statusBar().showMessage(f"Saved {path}", 5000)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive checkbox viewer for loudspeaker measurements.",
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
        help="Default directory for saved plots",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if (
        not os.environ.get("DISPLAY")
        and "QT_QPA_PLATFORM" not in os.environ
        and sys.platform not in ("darwin", "win32")
    ):
        print(
            "Error: no graphical display is available.\n"
            "Run this from a desktop session, or use the static script:\n"
            "  python analyze_speakers.py",
            file=sys.stderr,
        )
        return 1

    try:
        data_dir = resolve_data_dir(args.data_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("Speaker measurement viewer")
    window = SpeakerViewer(data_dir, Path(args.output_dir).expanduser())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
