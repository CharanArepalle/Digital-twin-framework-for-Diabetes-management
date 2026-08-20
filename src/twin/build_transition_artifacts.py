"""
T1D-UOM Digital Twin Transition Artifact Builder
================================================

Architecture
------------

    Unified Patient State
            |
            v
    Adjacent Temporal Pairs
            |
            v
       TwinTransition
            |
            v
      TwinDynamics
            |
            v
      Simulated State
         /       \
        v         v
   Prediction   What-if

Purpose
-------
Build persistent transition artifacts from already-generated Unified
Patient State trajectories.

This module operates ONLY on generated Unified Patient State CSV files.

It does NOT:

    - access raw source CSV files;
    - modify source CSV files;
    - resample;
    - interpolate;
    - impute;
    - normalize;
    - engineer features;
    - encode categorical variables;
    - generate patient states;
    - train TwinDynamics;
    - implement Prediction;
    - implement What-if;
    - implement UI.

Locked transition contract
--------------------------
For consecutive rows i and i+1:

    same participant
    AND next_timestamp > current_timestamp

        =>
            current_state = state(i)
            next_state    = state(i+1)
            delta_t       = next_timestamp - current_timestamp

Otherwise no transition is created.

Participant boundaries are NEVER crossed.

The existing repository transition implementation remains the
authoritative in-memory contract:

    src.twin.transitions.build_transition_dataset

This script only provides:

    CSV -> validated transition artifact

for subsequent TwinDynamics training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.twin.transitions import (
    TwinTransitionDataset,
    build_transition_dataset,
)


# ============================================================================
# LOCKED PROJECT CONSTANTS
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

STATE_ROOT = (
    ROOT
    / "data"
    / "derived"
    / "unified_state_trajectories"
)

TRANSITION_ROOT = (
    ROOT
    / "data"
    / "derived"
    / "twin_transitions"
)

STATE_DIM = 64

STATE_PREFIX = "state_"

REQUIRED_METADATA_COLUMNS = (
    "participant_id",
    "timestamp",
)


# ============================================================================
# FROZEN COHORT
# ============================================================================

FROZEN_COHORT = (
    "UoM2301",
    "UoM2302",
    "UoM2304",
    "UoM2305",
    "UoM2306",
    "UoM2307",
    "UoM2308",
    "UoM2309",
    "UoM2313",
    "UoM2401",
    "UoM2405",
)

EXCLUDED_INCOMPLETE_COHORT = {
    "UoM2314": "Required sleep time-series source is missing.",
    "UoM2403": "Required sleep time-series source is missing.",
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class ParticipantTrajectory:
    participant_id: str
    path: Path
    timestamps: tuple[pd.Timestamp, ...]
    states: torch.Tensor


@dataclass(frozen=True)
class TransitionBuildResult:
    participant_id: str
    source_path: Path
    output_path: Path
    state_count: int
    transition_count: int
    skipped_non_increasing: int
    state_dim: int


# ============================================================================
# GENERAL VALIDATION
# ============================================================================

def fail(message: str) -> "None":
    raise RuntimeError(message)


def finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{name} must be numeric; received {value!r}."
        ) from exc

    if not math.isfinite(result):
        raise RuntimeError(
            f"{name} must be finite; received {value!r}."
        )

    return result


def validate_participant_id(
    participant_id: object,
    *,
    expected: str | None = None,
) -> str:
    if not isinstance(participant_id, str):
        raise RuntimeError(
            "participant_id must be a string."
        )

    if not participant_id.strip():
        raise RuntimeError(
            "participant_id must not be empty."
        )

    if expected is not None and participant_id != expected:
        raise RuntimeError(
            f"Participant mismatch: expected {expected!r}, "
            f"received {participant_id!r}."
        )

    return participant_id


# ============================================================================
# STATE COLUMN DISCOVERY
# ============================================================================

def expected_state_columns() -> list[str]:
    return [
        f"{STATE_PREFIX}{index:02d}"
        for index in range(STATE_DIM)
    ]


def discover_state_columns(
    df: pd.DataFrame,
    *,
    path: Path,
) -> list[str]:

    expected = expected_state_columns()

    missing = [
        column
        for column in expected
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{path}: missing Unified Patient State columns: "
            f"{missing}"
        )

    # Reject unexpected state columns such as state_64.
    discovered = sorted(
        [
            column
            for column in df.columns
            if column.startswith(STATE_PREFIX)
        ]
    )

    if discovered != expected:
        raise RuntimeError(
            f"{path}: invalid Unified Patient State column contract.\n"
            f"Expected exactly {expected}.\n"
            f"Received {discovered}."
        )

    return expected


# ============================================================================
# STATE TRAJECTORY LOADING
# ============================================================================

def load_state_trajectory(
    path: Path,
    *,
    expected_participant: str | None = None,
) -> ParticipantTrajectory:

    if not path.exists():
        raise RuntimeError(
            f"Unified state trajectory does not exist:\n{path}"
        )

    if path.suffix.lower() != ".csv":
        raise RuntimeError(
            f"Expected CSV trajectory: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise RuntimeError(
            f"{path}: trajectory is empty."
        )

    for column in REQUIRED_METADATA_COLUMNS:
        if column not in df.columns:
            raise RuntimeError(
                f"{path}: missing required column '{column}'."
            )

    state_columns = discover_state_columns(
        df,
        path=path,
    )

    participant_values = (
        df["participant_id"]
        .astype(str)
        .tolist()
    )

    unique_participants = sorted(
        set(participant_values)
    )

    if len(unique_participants) != 1:
        raise RuntimeError(
            f"{path}: trajectory must contain exactly one participant. "
            f"Found: {unique_participants}"
        )

    participant_id = validate_participant_id(
        unique_participants[0],
        expected=expected_participant,
    )

    # ------------------------------------------------------------------
    # Timestamp validation
    # ------------------------------------------------------------------

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        bad_indices = (
            timestamps[timestamps.isna()]
            .index
            .tolist()[:10]
        )

        raise RuntimeError(
            f"{path}: invalid timestamps at row(s): "
            f"{bad_indices}"
        )

    # ------------------------------------------------------------------
    # State numeric conversion
    # ------------------------------------------------------------------

    state_frame = df[state_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if state_frame.isna().any().any():
        bad_rows = (
            state_frame.isna().any(axis=1)
            .loc[lambda x: x]
            .index
            .tolist()[:10]
        )

        raise RuntimeError(
            f"{path}: Unified Patient State contains non-numeric "
            f"or missing values at row(s): {bad_rows}"
        )

    state_array = state_frame.to_numpy(
        dtype=np.float32,
    )

    if not np.isfinite(state_array).all():
        raise RuntimeError(
            f"{path}: Unified Patient State contains NaN or Inf."
        )

    if state_array.ndim != 2:
        raise RuntimeError(
            f"{path}: invalid state matrix shape "
            f"{state_array.shape}."
        )

    if state_array.shape[1] != STATE_DIM:
        raise RuntimeError(
            f"{path}: state dimension is "
            f"{state_array.shape[1]}; expected {STATE_DIM}."
        )

    # ------------------------------------------------------------------
    # CRITICAL: verify the trajectory is already chronological.
    #
    # We DO NOT sort here.
    #
    # The repository transition contract explicitly requires the caller
    # to provide observations in intended temporal order.
    # ------------------------------------------------------------------

    timestamp_seconds = (
        timestamps.astype("int64")
        .to_numpy()
        / 1_000_000_000.0
    )

    if len(timestamp_seconds) > 1:
        differences = np.diff(
            timestamp_seconds
        )

        if np.any(differences <= 0.0):
            bad_positions = np.where(
                differences <= 0.0
            )[0]

            preview = bad_positions[:10].tolist()

            raise RuntimeError(
                f"{path}: timestamps are not strictly increasing. "
                f"Problem positions: {preview}"
            )

    return ParticipantTrajectory(
        participant_id=participant_id,
        path=path,
        timestamps=tuple(timestamps.tolist()),
        states=torch.from_numpy(
            state_array
        ),
    )


# ============================================================================
# TRANSITION CONSTRUCTION
# ============================================================================

def build_transitions(
    trajectory: ParticipantTrajectory,
) -> tuple[
    TwinTransitionDataset,
    int,
]:
    """
    Construct adjacent transitions using the repository's authoritative
    transition implementation.

    Because the input trajectory contains exactly one participant and
    timestamps have already been validated as strictly increasing,
    every adjacent pair is expected to become a valid transition.
    """

    participant_ids = [
        trajectory.participant_id
        for _ in trajectory.timestamps
    ]

    dataset = build_transition_dataset(
        states=trajectory.states,
        timestamps=trajectory.timestamps,
        participant_ids=participant_ids,
    )

    expected_count = max(
        trajectory.states.shape[0] - 1,
        0,
    )

    if len(dataset) != expected_count:
        raise RuntimeError(
            f"{trajectory.path}: transition count mismatch. "
            f"Expected {expected_count}, "
            f"received {len(dataset)}."
        )

    return (
        dataset,
        expected_count - len(dataset),
    )


# ============================================================================
# TRANSITION CSV WRITING
# ============================================================================

def transition_columns() -> list[str]:
    return (
        [
            "participant_id",
            "current_timestamp",
            "next_timestamp",
            "delta_t_seconds",
        ]
        + [
            f"current_{STATE_PREFIX}{index:02d}"
            for index in range(STATE_DIM)
        ]
        + [
            f"next_{STATE_PREFIX}{index:02d}"
            for index in range(STATE_DIM)
        ]
    )


def _transition_item_get(item, key: str, index: int):
    """Read a field from the repository's actual transition item."""
    if isinstance(item, dict) and key in item:
        return item[key]

    try:
        return item[key]
    except (KeyError, TypeError, IndexError):
        pass

    if hasattr(item, key):
        return getattr(item, key)

    raise RuntimeError(
        f"Transition {index} does not expose required field '{key}'. "
        f"Received type: {type(item).__name__}."
    )


def _transition_state_array(value, *, name: str, index: int):
    """Validate and convert one 64-dimensional transition state."""
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.float32)

    tensor = value.detach().cpu().reshape(-1)

    if tensor.numel() != STATE_DIM:
        raise RuntimeError(
            f"Transition {index}: {name} has {tensor.numel()} values; "
            f"expected {STATE_DIM}."
        )

    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(
            f"Transition {index}: {name} contains NaN or Inf."
        )

    return tensor.numpy().astype(np.float32, copy=False)


def _transition_delta_seconds(value, *, index: int) -> float:
    """Validate one scalar delta_t."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() != 1:
            raise RuntimeError(
                f"Transition {index}: delta_t must be scalar; "
                f"received shape {tuple(tensor.shape)}."
            )
        value = tensor.item()

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Transition {index}: delta_t is not numeric: {value!r}"
        ) from exc

    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError(
            f"Transition {index}: delta_t must be finite and > 0; "
            f"received {value!r}."
        )

    return result


def dataset_to_dataframe(
    dataset: TwinTransitionDataset,
    *,
    participant_id: str,
    timestamps: tuple[pd.Timestamp, ...],
) -> pd.DataFrame:
    """
    Serialize the repository transition dataset.

    The authoritative repository implementation currently exposes
    transition items containing:

        current_state
        next_state
        delta_t

    Participant/timestamp metadata therefore comes from the already
    validated Unified Patient State trajectory.
    """

    expected_count = max(len(timestamps) - 1, 0)

    if len(dataset) != expected_count:
        raise RuntimeError(
            f"{participant_id}: transition count mismatch. "
            f"Expected {expected_count}, received {len(dataset)}."
        )

    rows = []

    for index in range(len(dataset)):
        item = dataset[index]

        current_state = _transition_state_array(
            _transition_item_get(item, "current_state", index),
            name="current_state",
            index=index,
        )

        next_state = _transition_state_array(
            _transition_item_get(item, "next_state", index),
            name="next_state",
            index=index,
        )

        delta_t = _transition_delta_seconds(
            _transition_item_get(item, "delta_t", index),
            index=index,
        )

        current_timestamp = timestamps[index]
        next_timestamp = timestamps[index + 1]

        calculated_delta = (
            next_timestamp - current_timestamp
        ).total_seconds()

        if calculated_delta <= 0.0:
            raise RuntimeError(
                f"{participant_id}: transition {index} has "
                "non-positive timestamp difference."
            )

        if not math.isclose(
            delta_t,
            calculated_delta,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                f"{participant_id}: transition {index} delta_t mismatch. "
                f"Repository={delta_t}, timestamps={calculated_delta}."
            )

        row = {
            "participant_id": participant_id,
            "current_timestamp": current_timestamp.isoformat(sep=" "),
            "next_timestamp": next_timestamp.isoformat(sep=" "),
            "delta_t_seconds": delta_t,
        }

        for dimension in range(STATE_DIM):
            row[f"current_state_{dimension:02d}"] = float(
                current_state[dimension]
            )

        for dimension in range(STATE_DIM):
            row[f"next_state_{dimension:02d}"] = float(
                next_state[dimension]
            )

        rows.append(row)

    columns = (
        [
            "participant_id",
            "current_timestamp",
            "next_timestamp",
            "delta_t_seconds",
        ]
        + [f"current_state_{i:02d}" for i in range(STATE_DIM)]
        + [f"next_state_{i:02d}" for i in range(STATE_DIM)]
    )

    return pd.DataFrame(rows, columns=columns)


# ============================================================================
# ARTIFACT VALIDATION
# ============================================================================

def validate_transition_dataframe(
    df: pd.DataFrame,
    *,
    expected_participant: str | None = None,
) -> None:
    """
    Validate a serialized TwinDynamics transition dataframe.

    Metadata is validated as metadata. Only the 128 state columns are
    converted to numeric values and checked for finite values.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Transition artifact must be a pandas DataFrame."
        )

    if df.empty:
        raise RuntimeError(
            "Transition dataframe is empty."
        )

    metadata_columns = [
        "participant_id",
        "current_timestamp",
        "next_timestamp",
        "delta_t_seconds",
    ]

    missing_metadata = [
        c for c in metadata_columns if c not in df.columns
    ]

    if missing_metadata:
        raise RuntimeError(
            "Transition dataframe is missing required metadata columns: "
            + ", ".join(missing_metadata)
        )

    current_columns = [
        f"current_state_{i:02d}"
        for i in range(STATE_DIM)
    ]

    next_columns = [
        f"next_state_{i:02d}"
        for i in range(STATE_DIM)
    ]

    missing_current = [
        c for c in current_columns if c not in df.columns
    ]
    missing_next = [
        c for c in next_columns if c not in df.columns
    ]

    if missing_current:
        raise RuntimeError(
            "Transition dataframe is missing current-state columns: "
            + ", ".join(missing_current)
        )

    if missing_next:
        raise RuntimeError(
            "Transition dataframe is missing next-state columns: "
            + ", ".join(missing_next)
        )

    expected_columns = (
        metadata_columns + current_columns + next_columns
    )

    unexpected = [
        c for c in df.columns if c not in expected_columns
    ]

    if unexpected:
        raise RuntimeError(
            "Transition dataframe contains unexpected columns: "
            + ", ".join(unexpected)
        )

    participants = df["participant_id"].astype(str).str.strip()

    if participants.eq("").any():
        raise RuntimeError(
            "Transition dataframe contains an empty participant_id."
        )

    if expected_participant is not None:
        unexpected_participants = sorted(
            set(participants.tolist()) - {expected_participant}
        )
        if unexpected_participants:
            raise RuntimeError(
                f"Unexpected participant(s): {unexpected_participants}"
            )

    current_timestamps = pd.to_datetime(
        df["current_timestamp"],
        errors="coerce",
    )
    next_timestamps = pd.to_datetime(
        df["next_timestamp"],
        errors="coerce",
    )

    if current_timestamps.isna().any():
        raise RuntimeError(
            "Transition dataframe contains invalid current_timestamp values."
        )

    if next_timestamps.isna().any():
        raise RuntimeError(
            "Transition dataframe contains invalid next_timestamp values."
        )

    delta_t = pd.to_numeric(
        df["delta_t_seconds"],
        errors="coerce",
    )

    if delta_t.isna().any():
        raise RuntimeError(
            "Transition dataframe contains non-numeric delta_t_seconds values."
        )

    delta_values = delta_t.to_numpy(dtype=np.float64)

    if not np.isfinite(delta_values).all():
        raise RuntimeError(
            "Transition dataframe contains non-finite delta_t_seconds values."
        )

    if (delta_values <= 0.0).any():
        raise RuntimeError(
            "Transition dataframe contains non-positive delta_t_seconds values."
        )

    actual_delta = (
        next_timestamps - current_timestamps
    ).dt.total_seconds().to_numpy(dtype=np.float64)

    if not np.all(actual_delta > 0.0):
        raise RuntimeError(
            "Transition dataframe contains a non-forward temporal transition."
        )

    if not np.allclose(
        actual_delta,
        delta_values,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "delta_t_seconds does not match timestamp differences."
        )

    current_state = df[current_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    next_state = df[next_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if current_state.isna().any().any():
        raise RuntimeError(
            "Transition current-state values contain missing/non-numeric values."
        )

    if next_state.isna().any().any():
        raise RuntimeError(
            "Transition next-state values contain missing/non-numeric values."
        )

    current_array = current_state.to_numpy(dtype=np.float64)
    next_array = next_state.to_numpy(dtype=np.float64)

    if not np.isfinite(current_array).all():
        raise RuntimeError(
            "Transition current-state values contain NaN or Inf."
        )

    if not np.isfinite(next_array).all():
        raise RuntimeError(
            "Transition next-state values contain NaN or Inf."
        )

    if current_array.shape[1] != STATE_DIM:
        raise RuntimeError(
            f"Current-state dimension mismatch: "
            f"expected {STATE_DIM}, received {current_array.shape[1]}."
        )

    if next_array.shape[1] != STATE_DIM:
        raise RuntimeError(
            f"Next-state dimension mismatch: "
            f"expected {STATE_DIM}, received {next_array.shape[1]}."
        )


# ============================================================================
# PERSISTENCE
# ============================================================================

def output_path_for(
    participant_id: str,
) -> Path:

    return (
        TRANSITION_ROOT
        / f"{participant_id}_twin_transitions.csv"
    )


def write_metadata(
    results: Iterable[TransitionBuildResult],
) -> Path:

    results = tuple(results)

    TRANSITION_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        TRANSITION_ROOT
        / "transition_dataset_metadata.json"
    )

    payload = {
        "format": "t1d_uom_twin_transition_artifact_v1",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "architecture": (
            "Unified Patient State -> "
            "Temporal Transition Pairs -> TwinDynamics"
        ),
        "state_dimension": STATE_DIM,
        "transition_rule": (
            "Adjacent rows within the same participant; "
            "next_timestamp must be strictly greater than "
            "current_timestamp."
        ),
        "participant_boundary_crossing": False,
        "source_modification": False,
        "resampling": False,
        "interpolation": False,
        "imputation": False,
        "normalization": False,
        "feature_engineering": False,
        "delta_t_is_metadata_only": True,
        "frozen_cohort": list(FROZEN_COHORT),
        "participants": [
            {
                "participant_id": result.participant_id,
                "source": str(result.source_path),
                "output": str(result.output_path),
                "state_count": result.state_count,
                "transition_count": result.transition_count,
                "skipped_non_increasing": (
                    result.skipped_non_increasing
                ),
                "state_dim": result.state_dim,
            }
            for result in results
        ],
        "total_states": sum(
            result.state_count
            for result in results
        ),
        "total_transitions": sum(
            result.transition_count
            for result in results
        ),
    }

    metadata_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metadata_path


# ============================================================================
# PARTICIPANT DISCOVERY
# ============================================================================

def discover_state_files() -> dict[str, Path]:

    if not STATE_ROOT.exists():
        raise RuntimeError(
            "Unified state trajectory directory does not exist:\n"
            f"{STATE_ROOT}"
        )

    result: dict[str, Path] = {}

    for participant_id in FROZEN_COHORT:

        path = (
            STATE_ROOT
            / f"{participant_id}_unified_state.csv"
        )

        if path.exists():
            result[participant_id] = path

    return result


# ============================================================================
# SELF TEST
# ============================================================================

def self_test() -> None:

    print()
    print("=" * 80)
    print("TWIN TRANSITION ARTIFACT BUILDER SELF-TEST")
    print("=" * 80)

    states = torch.tensor(
        [
            [1.0] * STATE_DIM,
            [2.0] * STATE_DIM,
            [3.0] * STATE_DIM,
        ],
        dtype=torch.float32,
    )

    timestamps = (
        pd.Timestamp("2026-01-01 00:00:00"),
        pd.Timestamp("2026-01-01 00:05:00"),
        pd.Timestamp("2026-01-01 00:10:00"),
    )

    participant_ids = [
        "SELFTEST",
        "SELFTEST",
        "SELFTEST",
    ]

    dataset = build_transition_dataset(
        states=states,
        timestamps=timestamps,
        participant_ids=participant_ids,
    )

    if len(dataset) != 2:
        raise RuntimeError(
            f"Expected 2 transitions; received {len(dataset)}."
        )

    frame = dataset_to_dataframe(
        dataset,
        participant_id="SELFTEST",
        timestamps=timestamps,
    )

    validate_transition_dataframe(
        frame,
        expected_participant="SELFTEST",
    )

    if len(frame) != 2:
        raise RuntimeError(
            "Serialized self-test transition count is incorrect."
        )

    if not np.allclose(
        frame["delta_t_seconds"].to_numpy(dtype=np.float64),
        [300.0, 300.0],
    ):
        raise RuntimeError(
            "Self-test delta_t validation failed."
        )

    # Participant boundary test.
    boundary_dataset = build_transition_dataset(
        states=states,
        timestamps=timestamps,
        participant_ids=["A", "A", "B"],
    )

    if len(boundary_dataset) != 1:
        raise RuntimeError(
            "Participant boundary protection failed."
        )

    # Finite state test.
    item = dataset[0]

    current_state = _transition_state_array(
        _transition_item_get(item, "current_state", 0),
        name="current_state",
        index=0,
    )

    next_state = _transition_state_array(
        _transition_item_get(item, "next_state", 0),
        name="next_state",
        index=0,
    )

    if not np.isfinite(current_state).all():
        raise RuntimeError(
            "Current-state finite validation failed."
        )

    if not np.isfinite(next_state).all():
        raise RuntimeError(
            "Next-state finite validation failed."
        )

    print()
    print("State dimension          : 64")
    print("Adjacent transition rule : PASS")
    print("Participant boundary     : PASS")
    print("Positive delta_t         : PASS")
    print("Finite-state validation  : PASS")
    print("Repository contract      : PASS")
    print()
    print("SELF-TEST                : PASS")
    print()


# ============================================================================
# BUILD ONE PARTICIPANT
# ============================================================================

def build_one(
    participant_id: str,
    source_path: Path,
) -> TransitionBuildResult:

    print()
    print(
        f"[{participant_id}] Loading Unified Patient State..."
    )
    print(
        f"  source: {source_path}"
    )

    trajectory = load_state_trajectory(
        source_path,
        expected_participant=participant_id,
    )

    print(
        f"  states: {trajectory.states.shape[0]}"
    )
    print(
        f"  state dimension: {trajectory.states.shape[1]}"
    )

    dataset, skipped = build_transitions(
        trajectory
    )

    if skipped != 0:
        raise RuntimeError(
            f"{participant_id}: {skipped} transitions were skipped "
            "after chronological validation."
        )

    frame = dataset_to_dataframe(
        dataset,
        participant_id=participant_id,
        timestamps=trajectory.timestamps,
    )

    validate_transition_dataframe(
        frame,
        expected_participant=participant_id,
    )

    TRANSITION_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_path_for(
        participant_id
    )

    # Write only the derived transition artifact.
    # No source trajectory is modified.
    frame.to_csv(
        output_path,
        index=False,
    )

    # Read-back validation.
    written = pd.read_csv(
        output_path
    )

    validate_transition_dataframe(
        written,
        expected_participant=participant_id,
    )

    expected_transition_count = max(
        trajectory.states.shape[0] - 1,
        0,
    )

    if len(written) != expected_transition_count:
        raise RuntimeError(
            f"{participant_id}: output transition count mismatch. "
            f"Expected {expected_transition_count}, "
            f"received {len(written)}."
        )

    print(
        f"  transitions: {len(written)}"
    )
    print(
        f"  output: {output_path}"
    )
    print(
        f"[{participant_id}] PASS"
    )

    return TransitionBuildResult(
        participant_id=participant_id,
        source_path=source_path,
        output_path=output_path,
        state_count=int(
            trajectory.states.shape[0]
        ),
        transition_count=int(
            len(written)
        ),
        skipped_non_increasing=skipped,
        state_dim=int(
            trajectory.states.shape[1]
        ),
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build T1D-UOM TwinDynamics transition artifacts "
            "from Unified Patient State trajectories."
        )
    )

    parser.add_argument(
        "--participant",
        type=str,
        default=None,
        help="Build one participant, e.g. UoM2301.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Build artifacts for the frozen cohort.",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run contract self-test.",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.participant and args.all:
        parser.error(
            "Use either --participant or --all, not both."
        )

    if not args.participant and not args.all:
        parser.error(
            "Specify --participant UoMxxxx or --all."
        )

    print()
    print("=" * 80)
    print("T1D-UOM TWIN TRANSITION ARTIFACT BUILDER")
    print("=" * 80)
    print()
    print("Architecture:")
    print(
        "  Unified Patient State -> "
        "Temporal Transition Pairs -> TwinDynamics"
    )
    print()
    print(
        f"Unified state dimension : {STATE_DIM}"
    )
    print(
        "Transition rule        : adjacent same-participant states"
    )
    print(
        "Participant crossing   : NO"
    )
    print(
        "delta_t as model input : NO"
    )
    print()
    print("Frozen source policy:")
    print("  Source modification : NO")
    print("  Resampling          : NO")
    print("  Interpolation       : NO")
    print("  Imputation          : NO")
    print("  Normalization       : NO")
    print("  Feature engineering : NO")
    print()

    if args.participant:

        participant_id = validate_participant_id(
            args.participant
        )

        if participant_id not in FROZEN_COHORT:
            raise RuntimeError(
                f"{participant_id} is not in the frozen project cohort.\n"
                f"Frozen cohort: {', '.join(FROZEN_COHORT)}"
            )

        source_path = (
            STATE_ROOT
            / f"{participant_id}_unified_state.csv"
        )

        result = build_one(
            participant_id,
            source_path,
        )

        metadata_path = write_metadata(
            [result]
        )

        print()
        print("=" * 80)
        print("TWIN TRANSITION ARTIFACT BUILD: PASS")
        print("=" * 80)
        print(
            f"Participant       : {participant_id}"
        )
        print(
            f"States            : {result.state_count}"
        )
        print(
            f"Transitions       : {result.transition_count}"
        )
        print(
            f"Metadata          : {metadata_path}"
        )
        print()

        return

    # ------------------------------------------------------------------
    # --all
    # ------------------------------------------------------------------

    files = discover_state_files()

    missing = [
        participant
        for participant in FROZEN_COHORT
        if participant not in files
    ]

    if missing:
        print(
            "Missing final Unified Patient State trajectories:"
        )

        for participant in missing:
            print(
                f"  - {participant}"
            )

        print()
        print(
            "Generate the missing Unified Patient State trajectories "
            "before building the complete transition dataset."
        )

        sys.exit(1)

    results: list[TransitionBuildResult] = []

    for index, participant_id in enumerate(
        FROZEN_COHORT,
        start=1,
    ):

        print()
        print(
            f"[{index:02d}/{len(FROZEN_COHORT):02d}] "
            f"{participant_id}"
        )

        try:
            results.append(
                build_one(
                    participant_id,
                    files[participant_id],
                )
            )

        except Exception as exc:

            print()
            print("=" * 80)
            print(
                f"TRANSITION BUILD FAILED: {participant_id}"
            )
            print("=" * 80)
            print(str(exc))
            print()

            raise

    metadata_path = write_metadata(
        results
    )

    total_states = sum(
        result.state_count
        for result in results
    )

    total_transitions = sum(
        result.transition_count
        for result in results
    )

    print()
    print("=" * 80)
    print("T1D-UOM TWIN TRANSITION ARTIFACT BUILD: PASS")
    print("=" * 80)
    print()
    print(
        f"Participants processed : {len(results)}"
    )
    print(
        f"Total states           : {total_states}"
    )
    print(
        f"Total transitions      : {total_transitions}"
    )
    print(
        f"Output directory       : {TRANSITION_ROOT}"
    )
    print(
        f"Metadata               : {metadata_path}"
    )
    print()
    print(
        "Frozen source data was not modified."
    )
    print()


if __name__ == "__main__":
    main()