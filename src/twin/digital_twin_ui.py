from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DIGITAL_TWIN_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "digital_twin"
)

EVAL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "twin_evaluation"
)

STATE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "unified_state_trajectories"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "twin_models"
)

STATE_DIM = 64

DEFAULT_PARTICIPANTS = [
    "UoM2401",
    "UoM2405",
]

HORIZONS = [1, 5, 10, 30, 60]


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(value: Any) -> float | None:
    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except Exception:
        return None


def format_number(value: Any, digits: int = 6) -> str:
    number = safe_float(value)

    if number is None:
        return "N/A"

    return f"{number:.{digits}f}"


def find_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:

    lowered = {
        str(column).lower(): column
        for column in columns
    }

    for candidate in candidates:

        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    for column in columns:

        low = str(column).lower()

        for candidate in candidates:

            if candidate.lower() in low:
                return column

    return None


def discover_state_columns(df: pd.DataFrame) -> list[str]:

    columns = []

    for column in df.columns:

        name = str(column).lower()

        if (
            name.startswith("state_")
            or name.startswith("current_state_")
            or name.startswith("next_state_")
            or name.startswith("predicted_state_")
            or name.startswith("actual_state_")
            or name.startswith("what_if_state_")
        ):
            columns.append(column)

    return columns


def load_json(path: Path) -> dict:

    if not path.exists():
        return {}

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            data = json.load(handle)

        return data if isinstance(data, dict) else {}

    except Exception:
        return {}


# ============================================================================
# ARTIFACT LOADING
# ============================================================================

class DigitalTwinArtifacts:

    def __init__(self, participant: str):

        self.participant = participant

        self.prediction_path = (
            DIGITAL_TWIN_ROOT
            / f"{participant}_prediction.csv"
        )

        self.what_if_path = (
            DIGITAL_TWIN_ROOT
            / f"{participant}_what_if.csv"
        )

        self.comparison_path = (
            DIGITAL_TWIN_ROOT
            / f"{participant}_comparison.json"
        )

        self.state_path = (
            STATE_ROOT
            / f"{participant}_unified_state.csv"
        )

        self.prediction = pd.DataFrame()
        self.what_if = pd.DataFrame()
        self.comparison = {}

        self.load()

    def load(self):

        if self.prediction_path.exists():

            self.prediction = pd.read_csv(
                self.prediction_path
            )

        if self.what_if_path.exists():

            self.what_if = pd.read_csv(
                self.what_if_path
            )

        self.comparison = load_json(
            self.comparison_path
        )

    @property
    def available(self) -> bool:

        return (
            not self.prediction.empty
            or not self.what_if.empty
        )

    def summary(self) -> dict[str, Any]:

        result = {
            "participant": self.participant,
            "prediction_rows": len(self.prediction),
            "what_if_rows": len(self.what_if),
            "prediction_file": str(
                self.prediction_path
            ),
            "what_if_file": str(
                self.what_if_path
            ),
        }

        return result


# ============================================================================
# METRIC EXTRACTION
# ============================================================================

def extract_metrics(
    comparison: dict,
) -> dict[int, dict[str, float]]:

    result: dict[int, dict[str, float]] = {}

    def consume(obj: Any):

        if isinstance(obj, dict):

            horizon = (
                obj.get("horizon")
                or obj.get("H")
                or obj.get("h")
            )

            if horizon is not None:

                try:
                    horizon = int(horizon)
                except Exception:
                    horizon = None

            if horizon is not None:

                metrics = {}

                for metric in (
                    "mse",
                    "rmse",
                    "mae",
                    "r2",
                ):

                    if metric in obj:

                        value = safe_float(
                            obj[metric]
                        )

                        if value is not None:
                            metrics[metric] = value

                if metrics:
                    result[horizon] = metrics

            for value in obj.values():
                consume(value)

        elif isinstance(obj, list):

            for value in obj:
                consume(value)

    consume(comparison)

    return result


# ============================================================================
# PREDICTION DATA EXTRACTION
# ============================================================================

def build_prediction_series(
    df: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:

    if df.empty:
        return None

    columns = list(df.columns)

    horizon_column = find_column(
        columns,
        [
            "horizon",
            "prediction_horizon",
            "H",
        ],
    )

    work = df

    if horizon_column is not None:

        numeric_horizon = pd.to_numeric(
            work[horizon_column],
            errors="coerce",
        )

        subset = work[
            numeric_horizon == horizon
        ]

        if not subset.empty:
            work = subset

    timestamp_column = find_column(
        list(work.columns),
        [
            "timestamp",
            "current_timestamp",
            "next_timestamp",
            "predicted_timestamp",
        ],
    )

    if timestamp_column is not None:

        timestamps = pd.to_datetime(
            work[timestamp_column],
            errors="coerce",
        )

        x = np.arange(
            len(work),
            dtype=float,
        )

    else:

        x = np.arange(
            len(work),
            dtype=float,
        )

    actual_candidates = [
        column
        for column in work.columns
        if "actual" in str(column).lower()
        and pd.api.types.is_numeric_dtype(
            work[column]
        )
    ]

    predicted_candidates = [
        column
        for column in work.columns
        if (
            "predicted" in str(column).lower()
            or "prediction" in str(column).lower()
        )
        and pd.api.types.is_numeric_dtype(
            work[column]
        )
    ]

    if actual_candidates and predicted_candidates:

        actual = pd.to_numeric(
            work[actual_candidates].mean(axis=1),
            errors="coerce",
        ).to_numpy()

        predicted = pd.to_numeric(
            work[predicted_candidates].mean(axis=1),
            errors="coerce",
        ).to_numpy()

        mask = (
            np.isfinite(x)
            & np.isfinite(actual)
            & np.isfinite(predicted)
        )

        return (
            x[mask],
            actual[mask],
            predicted[mask],
        )

    # ------------------------------------------------------------------
    # Generic state-column fallback.
    #
    # If the artifact stores state vectors in separate columns, compare
    # corresponding actual/predicted state dimensions.
    # ------------------------------------------------------------------

    actual_state = [
        column
        for column in work.columns
        if str(column).lower().startswith(
            "actual_state_"
        )
    ]

    predicted_state = [
        column
        for column in work.columns
        if str(column).lower().startswith(
            "predicted_state_"
        )
    ]

    if actual_state and predicted_state:

        actual_state = sorted(
            actual_state,
            key=str,
        )

        predicted_state = sorted(
            predicted_state,
            key=str,
        )

        common = min(
            len(actual_state),
            len(predicted_state),
        )

        actual = work[
            actual_state[:common]
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).mean(axis=1).to_numpy()

        predicted = work[
            predicted_state[:common]
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).mean(axis=1).to_numpy()

        mask = (
            np.isfinite(x)
            & np.isfinite(actual)
            & np.isfinite(predicted)
        )

        return (
            x[mask],
            actual[mask],
            predicted[mask],
        )

    return None


# ============================================================================
# WHAT-IF DATA EXTRACTION
# ============================================================================

def build_what_if_series(
    df: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:

    if df.empty:
        return None

    work = df.copy()

    horizon_column = find_column(
        list(work.columns),
        [
            "horizon",
            "prediction_horizon",
            "H",
        ],
    )

    if horizon_column is not None:

        h = pd.to_numeric(
            work[horizon_column],
            errors="coerce",
        )

        subset = work[h == horizon]

        if not subset.empty:
            work = subset

    x = np.arange(
        len(work),
        dtype=float,
    )

    baseline_candidates = [
        column
        for column in work.columns
        if (
            "baseline" in str(column).lower()
            or "original" in str(column).lower()
        )
        and pd.api.types.is_numeric_dtype(
            work[column]
        )
    ]

    what_if_candidates = [
        column
        for column in work.columns
        if (
            "what_if" in str(column).lower()
            or "whatif" in str(column).lower()
            or "perturbed" in str(column).lower()
            or "scenario" in str(column).lower()
        )
        and pd.api.types.is_numeric_dtype(
            work[column]
        )
    ]

    if baseline_candidates and what_if_candidates:

        baseline = work[
            baseline_candidates
        ].mean(axis=1).to_numpy()

        what_if = work[
            what_if_candidates
        ].mean(axis=1).to_numpy()

        mask = (
            np.isfinite(x)
            & np.isfinite(baseline)
            & np.isfinite(what_if)
        )

        return (
            x[mask],
            baseline[mask],
            what_if[mask],
        )

    # Generic numerical fallback.
    numeric_columns = [
        column
        for column in work.columns
        if pd.api.types.is_numeric_dtype(
            work[column]
        )
    ]

    if len(numeric_columns) >= 2:

        baseline = pd.to_numeric(
            work[numeric_columns[0]],
            errors="coerce",
        ).to_numpy()

        what_if = pd.to_numeric(
            work[numeric_columns[1]],
            errors="coerce",
        ).to_numpy()

        mask = (
            np.isfinite(x)
            & np.isfinite(baseline)
            & np.isfinite(what_if)
        )

        return (
            x[mask],
            baseline[mask],
            what_if[mask],
        )

    return None


# ============================================================================
# UI
# ============================================================================

class DigitalTwinUI(tk.Tk):

    def __init__(self):

        super().__init__()

        self.title(
            "T1D-UOM Digital Twin — Prediction & What-if"
        )

        self.geometry(
            "1450x900"
        )

        self.minsize(
            1100,
            700,
        )

        self.participant_var = tk.StringVar(
            value=DEFAULT_PARTICIPANTS[0]
        )

        self.horizon_var = tk.IntVar(
            value=5
        )

        self.status_var = tk.StringVar(
            value="Ready"
        )

        self.metric_var = tk.StringVar(
            value=""
        )

        self.artifacts: DigitalTwinArtifacts | None = None

        self._build_ui()

        self.load_participant()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):

        top = ttk.Frame(
            self,
            padding=12,
        )

        top.pack(
            side=tk.TOP,
            fill=tk.X,
        )

        title = ttk.Label(
            top,
            text="T1D-UOM DIGITAL TWIN",
            font=(
                "Segoe UI",
                22,
                "bold",
            ),
        )

        title.pack(
            anchor=tk.W
        )

        subtitle = ttk.Label(
            top,
            text=(
                "Unified Patient State → TwinDynamics → "
                "Prediction / What-if Simulation"
            ),
            font=(
                "Segoe UI",
                11,
            ),
        )

        subtitle.pack(
            anchor=tk.W,
            pady=(2, 10),
        )

        controls = ttk.Frame(
            top
        )

        controls.pack(
            fill=tk.X
        )

        ttk.Label(
            controls,
            text="Held-out participant:",
        ).pack(
            side=tk.LEFT
        )

        participant_box = ttk.Combobox(
            controls,
            textvariable=self.participant_var,
            values=DEFAULT_PARTICIPANTS,
            state="readonly",
            width=14,
        )

        participant_box.pack(
            side=tk.LEFT,
            padx=(8, 20),
        )

        ttk.Label(
            controls,
            text="Horizon:",
        ).pack(
            side=tk.LEFT
        )

        horizon_box = ttk.Combobox(
            controls,
            textvariable=self.horizon_var,
            values=HORIZONS,
            state="readonly",
            width=8,
        )

        horizon_box.pack(
            side=tk.LEFT,
            padx=8,
        )

        ttk.Button(
            controls,
            text="Load / Refresh",
            command=self.load_participant,
        ).pack(
            side=tk.LEFT,
            padx=10,
        )

        ttk.Label(
            controls,
            textvariable=self.status_var,
        ).pack(
            side=tk.RIGHT
        )

        # --------------------------------------------------------------
        # Architecture banner
        # --------------------------------------------------------------

        architecture = ttk.LabelFrame(
            self,
            text="Locked Digital Twin Architecture",
            padding=10,
        )

        architecture.pack(
            fill=tk.X,
            padx=12,
            pady=(0, 8),
        )

        ttk.Label(
            architecture,
            text=(
                "64-D Unified Patient State"
                "   →   TwinDynamics"
                "   →   Simulated State"
                "   →   Prediction / What-if"
            ),
            font=(
                "Segoe UI",
                11,
                "bold",
            ),
        ).pack()

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------

        metrics_frame = ttk.LabelFrame(
            self,
            text="Held-out Evaluation",
            padding=10,
        )

        metrics_frame.pack(
            fill=tk.X,
            padx=12,
            pady=(0, 8),
        )

        self.metrics_text = tk.Text(
            metrics_frame,
            height=7,
            width=120,
            wrap=tk.WORD,
            font=(
                "Consolas",
                10,
            ),
        )

        self.metrics_text.pack(
            fill=tk.X
        )

        # --------------------------------------------------------------
        # Notebook
        # --------------------------------------------------------------

        notebook = ttk.Notebook(
            self
        )

        notebook.pack(
            fill=tk.BOTH,
            expand=True,
            padx=12,
            pady=(0, 12),
        )

        prediction_tab = ttk.Frame(
            notebook
        )

        what_if_tab = ttk.Frame(
            notebook
        )

        artifact_tab = ttk.Frame(
            notebook
        )

        notebook.add(
            prediction_tab,
            text="Prediction",
        )

        notebook.add(
            what_if_tab,
            text="What-if Simulation",
        )

        notebook.add(
            artifact_tab,
            text="Artifacts / Evidence",
        )

        # --------------------------------------------------------------
        # Prediction plot
        # --------------------------------------------------------------

        self.prediction_figure = Figure(
            figsize=(10, 6),
            dpi=100,
        )

        self.prediction_axis = (
            self.prediction_figure.add_subplot(111)
        )

        self.prediction_canvas = (
            FigureCanvasTkAgg(
                self.prediction_figure,
                master=prediction_tab,
            )
        )

        self.prediction_canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True,
        )

        # --------------------------------------------------------------
        # What-if plot
        # --------------------------------------------------------------

        self.what_if_figure = Figure(
            figsize=(10, 6),
            dpi=100,
        )

        self.what_if_axis = (
            self.what_if_figure.add_subplot(111)
        )

        self.what_if_canvas = (
            FigureCanvasTkAgg(
                self.what_if_figure,
                master=what_if_tab,
            )
        )

        self.what_if_canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True,
        )

        # --------------------------------------------------------------
        # Artifact evidence
        # --------------------------------------------------------------

        self.artifact_text = tk.Text(
            artifact_tab,
            wrap=tk.WORD,
            font=(
                "Consolas",
                10,
            ),
        )

        self.artifact_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10,
        )

    # ------------------------------------------------------------------
    # Load participant
    # ------------------------------------------------------------------

    def load_participant(self):

        participant = (
            self.participant_var.get()
            .strip()
        )

        if not participant:
            return

        try:

            self.status_var.set(
                "Loading artifacts..."
            )

            self.update_idletasks()

            artifacts = DigitalTwinArtifacts(
                participant
            )

            if not artifacts.available:

                raise RuntimeError(
                    "No Digital Twin prediction/what-if "
                    "artifacts were found for "
                    f"{participant}."
                )

            self.artifacts = artifacts

            self._update_metrics()

            self._update_prediction_plot()

            self._update_what_if_plot()

            self._update_artifact_panel()

            self.status_var.set(
                f"{participant} loaded"
            )

        except Exception as exc:

            self.status_var.set(
                "Load failed"
            )

            messagebox.showerror(
                "Digital Twin UI",
                str(exc),
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _update_metrics(self):

        if self.artifacts is None:
            return

        self.metrics_text.delete(
            "1.0",
            tk.END,
        )

        comparison = self.artifacts.comparison

        metrics = extract_metrics(
            comparison
        )

        lines = [
            f"Participant: {self.artifacts.participant}",
            "",
            "Prediction metrics:",
            "",
            " Horizon        MSE          RMSE          MAE           R²",
            " ----------------------------------------------------------------",
        ]

        for horizon in HORIZONS:

            row = metrics.get(
                horizon,
                {},
            )

            lines.append(
                f" H={horizon:>3}   "
                f"{format_number(row.get('mse')):>12}   "
                f"{format_number(row.get('rmse')):>12}   "
                f"{format_number(row.get('mae')):>12}   "
                f"{format_number(row.get('r2')):>10}"
            )

        lines.extend(
            [
                "",
                "The prediction target is the 64-dimensional "
                "Unified Patient State.",
                "These metrics must not be interpreted as direct "
                "clinical glucose-error metrics.",
            ]
        )

        self.metrics_text.insert(
            tk.END,
            "\n".join(lines),
        )

    # ------------------------------------------------------------------
    # Prediction plot
    # ------------------------------------------------------------------

    def _update_prediction_plot(self):

        if self.artifacts is None:
            return

        horizon = int(
            self.horizon_var.get()
        )

        axis = self.prediction_axis

        axis.clear()

        result = build_prediction_series(
            self.artifacts.prediction,
            horizon,
        )

        if result is None:

            axis.text(
                0.5,
                0.5,
                (
                    "Prediction artifact was generated, "
                    "but its columns could not be mapped "
                    "automatically for plotting."
                ),
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

        else:

            x, actual, predicted = result

            axis.plot(
                x,
                actual,
                label="Actual Unified State",
            )

            axis.plot(
                x,
                predicted,
                label="Digital Twin Prediction",
            )

            axis.set_title(
                f"{self.artifacts.participant} — "
                f"Digital Twin Prediction, H={horizon}"
            )

            axis.set_xlabel(
                "Evaluation sequence"
            )

            axis.set_ylabel(
                "Mean latent-state value"
            )

            axis.legend()

            axis.grid(
                True,
                alpha=0.25,
            )

        self.prediction_figure.tight_layout()

        self.prediction_canvas.draw()

    # ------------------------------------------------------------------
    # What-if plot
    # ------------------------------------------------------------------

    def _update_what_if_plot(self):

        if self.artifacts is None:
            return

        horizon = int(
            self.horizon_var.get()
        )

        axis = self.what_if_axis

        axis.clear()

        result = build_what_if_series(
            self.artifacts.what_if,
            horizon,
        )

        if result is None:

            axis.text(
                0.5,
                0.5,
                (
                    "What-if artifact was generated, "
                    "but its columns could not be mapped "
                    "automatically for plotting."
                ),
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

        else:

            x, baseline, what_if = result

            axis.plot(
                x,
                baseline,
                label="Baseline Twin",
            )

            axis.plot(
                x,
                what_if,
                label="What-if Twin",
            )

            axis.plot(
                x,
                what_if - baseline,
                label="Difference",
                linestyle="--",
            )

            axis.set_title(
                f"{self.artifacts.participant} — "
                f"Latent-State What-if, H={horizon}"
            )

            axis.set_xlabel(
                "Simulation sequence"
            )

            axis.set_ylabel(
                "Latent-state value / effect"
            )

            axis.legend()

            axis.grid(
                True,
                alpha=0.25,
            )

        self.what_if_figure.tight_layout()

        self.what_if_canvas.draw()

    # ------------------------------------------------------------------
    # Artifact panel
    # ------------------------------------------------------------------

    def _update_artifact_panel(self):

        if self.artifacts is None:
            return

        self.artifact_text.delete(
            "1.0",
            tk.END,
        )

        lines = [
            "DIGITAL TWIN ARTIFACT EVIDENCE",
            "=" * 72,
            "",
            f"Participant: {self.artifacts.participant}",
            "",
            "Prediction artifact:",
            f"  {self.artifacts.prediction_path}",
            "",
            "What-if artifact:",
            f"  {self.artifacts.what_if_path}",
            "",
            "Comparison artifact:",
            f"  {self.artifacts.comparison_path}",
            "",
            "Unified Patient State source:",
            f"  {self.artifacts.state_path}",
            "",
            "TwinDynamics checkpoint:",
            f"  {MODEL_ROOT / 'twin_dynamics_best.pt'}",
            "",
            "Architecture:",
            "  Unified Patient State [64]",
            "        ↓",
            "  DigitalTwin",
            "        ↓",
            "  TwinDynamics [64 → 64 residual transition]",
            "        ↓",
            "  Simulated State",
            "       ↙      ↘",
            " Prediction   What-if",
            "",
            "Held-out participants:",
            "  UoM2401",
            "  UoM2405",
            "",
            "Important interpretation:",
            "  The state is a learned 64-dimensional latent state.",
            "  What-if results represent latent-state perturbation,",
            "  not a direct physiological intervention.",
            "",
            "Source data policy:",
            "  Source modification : NO",
            "  Retraining          : NO",
            "  Resampling          : NO",
            "  Interpolation       : NO",
            "  Imputation          : NO",
            "  Normalization       : NO",
        ]

        self.artifact_text.insert(
            tk.END,
            "\n".join(lines),
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("T1D-UOM DIGITAL TWIN UI")
    print("=" * 80)
    print()
    print(
        "Unified Patient State -> TwinDynamics -> "
        "Prediction / What-if"
    )
    print()
    print(
        f"Digital Twin artifacts: {DIGITAL_TWIN_ROOT}"
    )
    print(
        f"Model checkpoint       : "
        f"{MODEL_ROOT / 'twin_dynamics_best.pt'}"
    )
    print()

    if not DIGITAL_TWIN_ROOT.exists():

        print(
            "ERROR: Digital Twin artifact directory does not exist:"
        )

        print(
            DIGITAL_TWIN_ROOT
        )

        sys.exit(1)

    app = DigitalTwinUI()

    app.mainloop()


if __name__ == "__main__":
    main()