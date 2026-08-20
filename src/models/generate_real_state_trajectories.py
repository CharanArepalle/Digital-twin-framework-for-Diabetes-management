"""
T1D-UOM REAL UNIFIED PATIENT STATE GENERATION
==============================================

Purpose
-------
Generate real-data Unified Patient State trajectories from the frozen
T1D-UOM sequence-input files.

LOCKED ARCHITECTURE
-------------------

    Glucose   -> GRU -> zG
    Insulin   -> GRU -> zI
    Nutrition -> GRU -> zN
    Activity  -> GRU -> zA
    Sleep     -> GRU -> zS
                         |
                    MLP Fusion
                         |
                Unified Patient State

This script is an integration/runtime script.

It does NOT:
    - modify source CSV files;
    - resample;
    - interpolate;
    - impute;
    - normalize;
    - fit categorical vocabularies from target patients;
    - create an additional Insulin GRU;
    - create an additional Sleep GRU;
    - modify FiveGRU;
    - modify MLPFusion;
    - implement Digital Twin dynamics;
    - implement Prediction;
    - implement What-if;
    - implement UI.

IMPORTANT
---------
The existing FiveGRUStatePipeline contract is:

    pipeline(FiveGRUInputBatch)

NOT:

    pipeline(glucose, insulin, nutrition, activity, sleep)

The pipeline returns:

    UnifiedPatientState

whose tensor is:

    patient_state.state

The generator therefore constructs FiveGRUInputBatch explicitly and
extracts UnifiedPatientState.state explicitly.

Runtime representation contract
--------------------------------

    Glucose:
        1 numeric feature
        1 -> 1

    Insulin:
        dose + event_type
        1 + 1 -> 2

    Nutrition:
        4 numeric features
        + meal_type embedding (10)
        + meal_tag embedding (10)
        4 -> 24

    Activity:
        10 numeric features
        + activity_type embedding (4)
        + intensity embedding (3)
        10 -> 17

    Sleep:
        six frozen sleep time-series numeric features
        6 -> 6

Categorical vocabulary policy
-----------------------------

Vocabulary is loaded ONLY from the persisted training-partition artifact.

No vocabulary is fitted from the target participant.

Raw categorical strings are not modified.

Missing / empty:
    -> MISSING_ID

Unseen:
    -> UNK_ID

Padding:
    -> PAD_ID

Causal policy
-------------

At timestamp t, each modality uses only observations whose timestamp
is <= t.

No future observation is used.

No artificial observations are created for a modality that has no
causal history yet.

Performance
-----------

The original implementation performed one complete model forward pass
per timestamp.

This implementation batches causal timestamps into chunks while keeping
the five modality branches independent.

Each branch receives:
    - its own padded history;
    - its own valid sequence length.

This preserves the FiveGRU independent-length contract while avoiding
tens of thousands of individual Python-level forward calls.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from src.models.five_gru import (
    ACTIVITY_DIM,
    GLUCOSE_DIM,
    INSULIN_DIM,
    NUTRITION_DIM,
    SLEEP_DIM,
    FiveGRU,
)
from src.models.five_gru_pipeline import (
    FiveGRUInputBatch,
    FiveGRUStatePipeline,
)
from src.models.mlp_fusion import MLPFusion
from src.models.input_adapters import (
    ActivityAdapter,
    IdentityNumericAdapter,
    InsulinAdapter,
    NutritionAdapter,
)
from src.models.patient_state import UnifiedPatientState
from src.representations.categorical_vocabulary import (
    CategoricalVocabulary,
    validate_vocabulary,
)


# ============================================================================
# PROJECT PATHS
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "t1d_uom_v1.0.3_sequence_inputs"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "unified_state_trajectories"
)

VOCAB_ROOT_CANDIDATES = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "categorical_vocabularies",
    PROJECT_ROOT
    / "data"
    / "processed"
    / "categorical_vocabularies",
    PROJECT_ROOT
    / "artifacts"
    / "categorical_vocabularies",
    PROJECT_ROOT
    / "models"
    / "categorical_vocabularies",
)


# ============================================================================
# FROZEN MODEL CONTRACT
# ============================================================================

HIDDEN_DIM = 64
MAX_HISTORY = 64

DEFAULT_CHUNK_SIZE = 256

REQUIRED_VOCAB_FIELDS = (
    "meal_type",
    "meal_tag",
    "activity_type",
    "intensity",
)

EXPECTED_RUNTIME_DIMS = {
    "glucose": 1,
    "insulin": 2,
    "nutrition": 24,
    "activity": 17,
    "sleep": 6,
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class ModalityData:
    """One causal modality observation."""

    timestamp: pd.Timestamp
    values: np.ndarray
    categorical: dict[str, int]


@dataclass(frozen=True)
class ParticipantData:
    """All frozen modality observations for one participant."""

    participant_id: str
    glucose: list[ModalityData]
    insulin: list[ModalityData]
    nutrition: list[ModalityData]
    activity: list[ModalityData]
    sleep: list[ModalityData]


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def parse_timestamp(
    series: pd.Series,
    *,
    field: str,
) -> pd.Series:
    """
    Parse frozen project timestamps robustly without modifying source data.

    The T1D-UOM sequence inputs contain day-first timestamps, and some
    participants/files can contain mixed timestamp representations.
    pandas 2.x may infer a single format for an entire Series unless
    format="mixed" is specified.  We therefore explicitly request
    per-value mixed-format parsing when supported.

    No source values are changed, resampled, interpolated, imputed,
    normalized, or otherwise transformed.
    """

    raw = series.astype(str).str.strip()

    try:
        # pandas >= 2.0
        parsed = pd.to_datetime(
            raw,
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
    except (TypeError, ValueError):
        # Compatibility fallback for older pandas versions.
        parsed = pd.to_datetime(
            raw,
            dayfirst=True,
            errors="coerce",
        )

    if parsed.isna().any():
        bad = (
            raw.loc[parsed.isna()]
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            f"{field}: invalid timestamps. "
            f"Examples: {bad}"
        )

    # Remove timezone information if pandas produced timezone-aware
    # timestamps.  The project uses naive chronological timestamps.
    try:
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass

    return parsed


def load_csv(path: Path) -> pd.DataFrame:
    """Read one frozen CSV without modifying it."""

    if not path.exists():
        raise RuntimeError(
            f"Missing source file: {path}"
        )

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
    )


def identify_participant(path: Path) -> str:
    """
    Identify participant from the actual frozen filename conventions.

    Examples:

        UoMActivity2301.csv
        UoMGlucose2301.csv
        UoMBolus2301.csv
        UoMBasal2401.csv
        UoMNutrition2301.csv
        UoMsleep2301.csv
    """

    stem = path.stem.strip()

    pattern = re.compile(
        r"^UoM(?:Activity|Glucose|Bolus|Basal|Nutrition|sleep)?"
        r"(\d{4})(?:.*)?$",
        re.IGNORECASE,
    )

    match = pattern.match(stem)

    if match:
        return f"UoM{match.group(1)}"

    fallback = re.search(
        r"UoM(\d{4})",
        stem,
        re.IGNORECASE,
    )

    if fallback:
        return f"UoM{fallback.group(1)}"

    raise RuntimeError(
        f"Unable to identify participant from filename:\n{path}"
    )


# ============================================================================
# PARTICIPANT FILE DISCOVERY
# ============================================================================

def discover_participant_files(
    participant_id: str,
) -> dict[str, list[Path]]:
    """
    Discover the frozen sequence-input files for one participant.

    Required runtime families:

        glucose
        basal
        bolus
        nutrition
        activity
        sleep

    Basal may legitimately be absent for participants such as UoM2301.
    """

    participant_id = participant_id.strip()

    result: dict[str, list[Path]] = {
        "glucose": [],
        "basal": [],
        "bolus": [],
        "nutrition": [],
        "activity": [],
        "sleep": [],
    }

    if not SOURCE_ROOT.exists():
        raise RuntimeError(
            "Frozen sequence-input directory does not exist:\n"
            f"{SOURCE_ROOT}"
        )

    for path in SOURCE_ROOT.rglob("*.csv"):
        if not path.is_file():
            continue

        try:
            pid = identify_participant(path)
        except RuntimeError:
            continue

        if pid != participant_id:
            continue

        relative = path.relative_to(SOURCE_ROOT)
        parts = [
            part.lower()
            for part in relative.parts
        ]

        name = path.name.lower()

        if parts and parts[0] == "glucose data":
            result["glucose"].append(path)

        elif parts and parts[0] == "nutrition data":
            result["nutrition"].append(path)

        elif parts and parts[0] == "activity data":
            result["activity"].append(path)

        elif (
            len(parts) >= 2
            and parts[0] == "insulin data"
            and parts[1] == "basal data"
        ):
            result["basal"].append(path)

        elif (
            len(parts) >= 2
            and parts[0] == "insulin data"
            and parts[1] == "bolus data"
        ):
            result["bolus"].append(path)

        elif (
            parts
            and parts[0] == "sleep data"
            and name.startswith("uomsleep")
        ):
            result["sleep"].append(path)

    for key in result:
        result[key].sort()

    print()
    print(
        f"[{participant_id}] Frozen source discovery:"
    )

    for key in (
        "glucose",
        "basal",
        "bolus",
        "nutrition",
        "activity",
        "sleep",
    ):
        paths = result[key]

        if paths:
            for path in paths:
                print(
                    f"  {key:10s}: "
                    f"{path.relative_to(PROJECT_ROOT)}"
                )
        else:
            print(
                f"  {key:10s}: MISSING"
            )

    return result


# ============================================================================
# CATEGORICAL VOCABULARY LOADING
# ============================================================================

def _vocab_from_mapping(
    field_name: str,
    mapping: Mapping[str, Any],
) -> CategoricalVocabulary:
    """
    Convert one persisted vocabulary mapping to the repository object.

    Supported forms:

        {
            "token_to_id": {...}
        }

    or:

        {
            "<PAD>": 0,
            "<UNK>": 1,
            "<MISSING>": 2,
            ...
        }
    """

    if "token_to_id" in mapping:
        token_to_id = mapping["token_to_id"]
    else:
        token_to_id = mapping

    if not isinstance(token_to_id, Mapping):
        raise RuntimeError(
            f"Vocabulary '{field_name}' has invalid token_to_id data."
        )

    converted = {
        str(key): int(value)
        for key, value in token_to_id.items()
    }

    vocabulary = CategoricalVocabulary(
        field_name=field_name,
        token_to_id=converted,
    )

    validate_vocabulary(vocabulary)

    return vocabulary


def load_vocabulary_artifact() -> dict[str, CategoricalVocabulary]:
    """
    Load the persisted training-partition vocabulary artifact.

    The current artifact structure is:

        {
            "format": ...,
            "training_participants": [...],
            "policy": {...},
            "vocabularies": {
                "meal_type": {...},
                "meal_tag": {...},
                "activity_type": {...},
                "intensity": {...}
            }
        }

    No vocabulary fitting is performed here.
    """

    combined_names = (
        "categorical_vocabularies.json",
        "vocabularies.json",
        "categorical_vocabularies.pkl",
        "vocabularies.pkl",
        "categorical_vocabularies.pickle",
        "vocabularies.pickle",
    )

    directories = [
        directory
        for directory in VOCAB_ROOT_CANDIDATES
        if directory.exists()
    ]

    # ------------------------------------------------------------------
    # Combined artifact.
    # ------------------------------------------------------------------

    for directory in directories:
        for filename in combined_names:
            path = directory / filename

            if not path.exists():
                continue

            if path.suffix.lower() == ".json":
                with path.open(
                    "r",
                    encoding="utf-8",
                ) as handle:
                    raw = json.load(handle)
            else:
                with path.open("rb") as handle:
                    raw = pickle.load(handle)

            if not isinstance(raw, Mapping):
                raise RuntimeError(
                    f"Vocabulary artifact {path} must contain a mapping."
                )

            # Current canonical artifact.
            if "vocabularies" in raw:
                vocabulary_container = raw["vocabularies"]

                if not isinstance(
                    vocabulary_container,
                    Mapping,
                ):
                    raise RuntimeError(
                        f"Vocabulary artifact {path} has invalid "
                        "'vocabularies' field."
                    )
            else:
                # Backward-compatible legacy artifact.
                vocabulary_container = raw

            result: dict[str, CategoricalVocabulary] = {}

            for field in REQUIRED_VOCAB_FIELDS:
                if field not in vocabulary_container:
                    raise RuntimeError(
                        f"Vocabulary artifact {path} is missing "
                        f"required vocabulary field '{field}'."
                    )

                value = vocabulary_container[field]

                if isinstance(
                    value,
                    CategoricalVocabulary,
                ):
                    vocabulary = value
                    validate_vocabulary(vocabulary)
                else:
                    if not isinstance(value, Mapping):
                        raise RuntimeError(
                            f"Vocabulary artifact {path} contains "
                            f"invalid data for '{field}': "
                            f"expected a mapping, received "
                            f"{type(value).__name__}."
                        )

                    vocabulary = _vocab_from_mapping(
                        field,
                        value,
                    )

                result[field] = vocabulary

            # Optional metadata validation.
            if "training_participants" in raw:
                participants = raw["training_participants"]

                if not isinstance(
                    participants,
                    list,
                ):
                    raise RuntimeError(
                        f"Vocabulary artifact {path} has invalid "
                        "'training_participants'."
                    )

            print(
                f"  Vocabulary artifact: {path}"
            )

            if "training_participants" in raw:
                print(
                    "  Vocabulary source: training-partition artifact"
                )
                print(
                    "  Training participants: "
                    + ", ".join(
                        str(item)
                        for item in raw["training_participants"]
                    )
                )

            return result

    # ------------------------------------------------------------------
    # Individual artifacts.
    # ------------------------------------------------------------------

    result: dict[str, CategoricalVocabulary] = {}

    for field in REQUIRED_VOCAB_FIELDS:
        found: Path | None = None

        names = (
            f"{field}.json",
            f"{field}.pkl",
            f"{field}.pickle",
            f"{field}_vocabulary.json",
            f"{field}_vocabulary.pkl",
            f"{field}_vocabulary.pickle",
        )

        for directory in directories:
            for filename in names:
                path = directory / filename

                if path.exists():
                    found = path
                    break

            if found is not None:
                break

        if found is None:
            continue

        if found.suffix.lower() == ".json":
            with found.open(
                "r",
                encoding="utf-8",
            ) as handle:
                raw = json.load(handle)
        else:
            with found.open("rb") as handle:
                raw = pickle.load(handle)

        if isinstance(
            raw,
            CategoricalVocabulary,
        ):
            vocabulary = raw
        else:
            if not isinstance(raw, Mapping):
                raise RuntimeError(
                    f"Vocabulary artifact {found} must contain "
                    "a mapping."
                )

            vocabulary = _vocab_from_mapping(
                field,
                raw,
            )

        validate_vocabulary(vocabulary)

        result[field] = vocabulary

    missing = [
        field
        for field in REQUIRED_VOCAB_FIELDS
        if field not in result
    ]

    if missing:
        searched = "\n".join(
            f"  {path}"
            for path in VOCAB_ROOT_CANDIDATES
        )

        raise RuntimeError(
            "Training-partition categorical vocabulary artifact was "
            "not found for: "
            + ", ".join(missing)
            + "\n\nSearched:\n"
            + searched
            + "\n\nThe generator refuses to fit vocabularies from "
              "the target participant because that would violate "
              "the frozen training-partition vocabulary contract."
        )

    return result


# ============================================================================
# GLUCOSE
# ============================================================================

def load_glucose(
    paths: list[Path],
) -> list[ModalityData]:

    records: list[ModalityData] = []

    for path in paths:
        df = load_csv(path)

        if "bg_ts" not in df.columns:
            raise RuntimeError(
                f"{path}: missing 'bg_ts'."
            )

        value_candidates = (
            "bg",
            "glucose",
            "value",
            "bg_value",
        )

        value_column = next(
            (
                column
                for column in value_candidates
                if column in df.columns
            ),
            None,
        )

        if value_column is None:
            raise RuntimeError(
                f"{path}: unable to identify glucose value column. "
                f"Available columns: {list(df.columns)}"
            )

        timestamps = parse_timestamp(
            df["bg_ts"],
            field=str(path),
        )

        values = pd.to_numeric(
            df[value_column],
            errors="coerce",
        )

        for timestamp, value in zip(
            timestamps,
            values,
        ):
            if pd.isna(value):
                continue

            value_float = float(value)

            if not np.isfinite(value_float):
                continue

            records.append(
                ModalityData(
                    timestamp=timestamp,
                    values=np.asarray(
                        [value_float],
                        dtype=np.float32,
                    ),
                    categorical={},
                )
            )

    records.sort(
        key=lambda record: record.timestamp
    )

    return records


# ============================================================================
# INSULIN
# ============================================================================

def load_insulin(
    paths: list[Path],
    event_type: int,
) -> list[ModalityData]:
    """
    Load basal or bolus insulin into the single frozen Insulin branch.

    Basal:
        basal_ts
        basal_dose

    Bolus:
        bolus_ts
        bolus_dose

    Runtime:
        [dose, event_type]
    """

    records: list[ModalityData] = []

    for path in paths:
        df = load_csv(path)

        timestamp_column = (
            "basal_ts"
            if event_type == InsulinAdapter.BASAL_EVENT_TYPE
            else "bolus_ts"
        )

        if timestamp_column not in df.columns:
            raise RuntimeError(
                f"{path}: missing '{timestamp_column}'."
            )

        # CRITICAL:
        # The actual frozen files use basal_dose / bolus_dose.
        if event_type == InsulinAdapter.BASAL_EVENT_TYPE:
            value_candidates = (
                "basal_dose",
                "dose",
                "insulin",
                "value",
                "units",
                "amount",
            )
        else:
            value_candidates = (
                "bolus_dose",
                "dose",
                "insulin",
                "value",
                "units",
                "amount",
            )

        value_column = next(
            (
                column
                for column in value_candidates
                if column in df.columns
            ),
            None,
        )

        if value_column is None:
            raise RuntimeError(
                f"{path}: unable to identify insulin dose column. "
                f"Expected one of {value_candidates}. "
                f"Available columns: {list(df.columns)}"
            )

        timestamps = parse_timestamp(
            df[timestamp_column],
            field=str(path),
        )

        values = pd.to_numeric(
            df[value_column],
            errors="coerce",
        )

        for timestamp, value in zip(
            timestamps,
            values,
        ):
            if pd.isna(value):
                continue

            value_float = float(value)

            if not np.isfinite(value_float):
                continue

            records.append(
                ModalityData(
                    timestamp=timestamp,
                    values=np.asarray(
                        [value_float],
                        dtype=np.float32,
                    ),
                    categorical={
                        "event_type": int(event_type),
                    },
                )
            )

    records.sort(
        key=lambda record: record.timestamp
    )

    return records


# ============================================================================
# NUTRITION
# ============================================================================

def load_nutrition(
    paths: list[Path],
    vocab: dict[str, CategoricalVocabulary],
) -> tuple[list[ModalityData], int]:

    records: list[ModalityData] = []
    excluded = 0

    numeric_columns = [
        "carbs_g",
        "prot_g",
        "fat_g",
        "fibre_g",
    ]

    required_columns = numeric_columns + [
        "meal_type",
        "meal_tag",
    ]

    for path in paths:
        df = load_csv(path)

        if "meal_ts" not in df.columns:
            raise RuntimeError(
                f"{path}: missing 'meal_ts'."
            )

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise RuntimeError(
                f"{path}: missing required nutrition columns: "
                f"{missing}"
            )

        timestamps = parse_timestamp(
            df["meal_ts"],
            field=str(path),
        )

        numeric = df[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        for index in range(len(df)):
            row_numeric = numeric.iloc[index]

            if row_numeric.isna().any():
                excluded += 1
                continue

            values = row_numeric.to_numpy(
                dtype=np.float32,
            )

            if not np.isfinite(values).all():
                excluded += 1
                continue

            records.append(
                ModalityData(
                    timestamp=timestamps.iloc[index],
                    values=values,
                    categorical={
                        "meal_type": vocab[
                            "meal_type"
                        ].id_for(
                            df.iloc[index]["meal_type"]
                        ),
                        "meal_tag": vocab[
                            "meal_tag"
                        ].id_for(
                            df.iloc[index]["meal_tag"]
                        ),
                    },
                )
            )

    records.sort(
        key=lambda record: record.timestamp
    )

    return records, excluded


# ============================================================================
# ACTIVITY
# ============================================================================

def load_activity(
    paths: list[Path],
    vocab: dict[str, CategoricalVocabulary],
) -> tuple[list[ModalityData], int]:

    records: list[ModalityData] = []
    excluded = 0

    numeric_columns = [
        "active_Kcal",
        "step_count",
        "distance_m",
        "duration_s",
        "active_time_s",
        "start_time_s",
        "start_time_offset_s",
        "met",
        "motion_intensity_mean",
        "motion_intensity_max",
    ]

    required_columns = numeric_columns + [
        "activity_type",
        "intensity",
    ]

    for path in paths:
        df = load_csv(path)

        if "activity_ts" not in df.columns:
            raise RuntimeError(
                f"{path}: missing 'activity_ts'."
            )

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise RuntimeError(
                f"{path}: missing required activity columns: "
                f"{missing}"
            )

        timestamps = parse_timestamp(
            df["activity_ts"],
            field=str(path),
        )

        numeric = df[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        for index in range(len(df)):
            row_numeric = numeric.iloc[index]

            if row_numeric.isna().any():
                excluded += 1
                continue

            values = row_numeric.to_numpy(
                dtype=np.float32,
            )

            if not np.isfinite(values).all():
                excluded += 1
                continue

            records.append(
                ModalityData(
                    timestamp=timestamps.iloc[index],
                    values=values,
                    categorical={
                        "activity_type": vocab[
                            "activity_type"
                        ].id_for(
                            df.iloc[index]["activity_type"]
                        ),
                        "intensity": vocab[
                            "intensity"
                        ].id_for(
                            df.iloc[index]["intensity"]
                        ),
                    },
                )
            )

    records.sort(
        key=lambda record: record.timestamp
    )

    return records, excluded


# ============================================================================
# SLEEP
# ============================================================================

def load_sleep(
    paths: list[Path],
) -> tuple[list[ModalityData], int]:
    """
    Load the frozen six-feature sleep time-series representation.

    Exact frozen features:

        step_count
        heart_rate
        current_activity_type_intensity
        stress_level_value
        sleep_level
        resting_heart_rate
    """

    records: list[ModalityData] = []
    excluded = 0

    numeric_columns = [
        "step_count",
        "heart_rate",
        "current_activity_type_intensity",
        "stress_level_value",
        "sleep_level",
        "resting_heart_rate",
    ]

    for path in paths:
        df = load_csv(path)

        if "sleep_ts" not in df.columns:
            raise RuntimeError(
                f"{path}: missing 'sleep_ts'."
            )

        missing = [
            column
            for column in numeric_columns
            if column not in df.columns
        ]

        if missing:
            raise RuntimeError(
                f"{path}: missing required sleep time-series "
                f"columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        timestamps = parse_timestamp(
            df["sleep_ts"],
            field=str(path),
        )

        numeric = df[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        for index in range(len(df)):
            row = numeric.iloc[index]

            if row.isna().any():
                excluded += 1
                continue

            values = row.to_numpy(
                dtype=np.float32,
            )

            if not np.isfinite(values).all():
                excluded += 1
                continue

            records.append(
                ModalityData(
                    timestamp=timestamps.iloc[index],
                    values=values,
                    categorical={},
                )
            )

    records.sort(
        key=lambda record: record.timestamp
    )

    return records, excluded


# ============================================================================
# PARTICIPANT LOADING
# ============================================================================

def load_participant_data(
    participant_id: str,
    vocab: dict[str, CategoricalVocabulary],
) -> ParticipantData:

    files = discover_participant_files(
        participant_id,
    )

    for key in (
        "glucose",
        "nutrition",
        "activity",
        "sleep",
    ):
        if not files[key]:
            raise RuntimeError(
                f"{participant_id}: {key} source is missing."
            )

    if not files["basal"] and not files["bolus"]:
        raise RuntimeError(
            f"{participant_id}: insulin source is missing."
        )

    glucose = load_glucose(
        files["glucose"]
    )

    basal = load_insulin(
        files["basal"],
        event_type=InsulinAdapter.BASAL_EVENT_TYPE,
    )

    bolus = load_insulin(
        files["bolus"],
        event_type=InsulinAdapter.BOLUS_EVENT_TYPE,
    )

    insulin = basal + bolus

    insulin.sort(
        key=lambda record: record.timestamp
    )

    nutrition, nutrition_excluded = load_nutrition(
        files["nutrition"],
        vocab,
    )

    activity, activity_excluded = load_activity(
        files["activity"],
        vocab,
    )

    sleep, sleep_excluded = load_sleep(
        files["sleep"],
    )

    print()
    print(
        f"[{participant_id}] Loading frozen source data..."
    )

    print(
        f"  glucose observations  : {len(glucose)}"
    )
    print(
        f"  insulin observations : {len(insulin)}"
    )
    print(
        f"  nutrition observations: {len(nutrition)}"
    )
    print(
        f"  activity observations : {len(activity)}"
    )
    print(
        f"  sleep observations    : {len(sleep)}"
    )

    print()
    print("  Missing-value handling:")
    print(
        "    incomplete rows are EXCLUDED, never imputed."
    )
    print(
        f"    nutrition: excluded {nutrition_excluded} row(s)"
    )
    print(
        f"    activity : excluded {activity_excluded} row(s)"
    )
    print(
        f"    sleep    : excluded {sleep_excluded} row(s)"
    )

    return ParticipantData(
        participant_id=participant_id,
        glucose=glucose,
        insulin=insulin,
        nutrition=nutrition,
        activity=activity,
        sleep=sleep,
    )


# ============================================================================
# ADAPTER CONSTRUCTION
# ============================================================================

def build_adapters(
    vocab: dict[str, CategoricalVocabulary],
) -> dict[str, Any]:

    glucose_adapter = IdentityNumericAdapter(
        input_dim=1,
    )

    insulin_adapter = InsulinAdapter()

    nutrition_adapter = NutritionAdapter(
        meal_type_cardinality=vocab[
            "meal_type"
        ].size,
        meal_tag_cardinality=vocab[
            "meal_tag"
        ].size,
        meal_type_embedding_dim=10,
        meal_tag_embedding_dim=10,
    )

    activity_adapter = ActivityAdapter(
        activity_type_cardinality=vocab[
            "activity_type"
        ].size,
        intensity_cardinality=vocab[
            "intensity"
        ].size,
        activity_type_embedding_dim=4,
        intensity_embedding_dim=3,
    )

    sleep_adapter = IdentityNumericAdapter(
        input_dim=6,
    )

    adapters = {
        "glucose": glucose_adapter,
        "insulin": insulin_adapter,
        "nutrition": nutrition_adapter,
        "activity": activity_adapter,
        "sleep": sleep_adapter,
    }

    # Hard runtime contract.
    assert adapters["glucose"].output_dim == 1
    assert adapters["insulin"].output_dim == 2
    assert adapters["nutrition"].output_dim == 24
    assert adapters["activity"].output_dim == 17
    assert adapters["sleep"].output_dim == 6

    return adapters


# ============================================================================
# MODEL CONSTRUCTION
# ============================================================================

def build_model() -> FiveGRUStatePipeline:
    """
    Build the locked FiveGRU -> MLPFusion -> UnifiedPatientState pipeline.
    """

    five_gru = FiveGRU(
        hidden_dim=HIDDEN_DIM,
    )

    fusion = MLPFusion(
        hidden_dim=HIDDEN_DIM,
    )

    pipeline = FiveGRUStatePipeline(
        five_gru=five_gru,
        fusion=fusion,
    )

    return pipeline


# ============================================================================
# CAUSAL HISTORY INDEXING
# ============================================================================

def _record_timestamps(
    records: list[ModalityData],
) -> list[pd.Timestamp]:
    return [
        record.timestamp
        for record in records
    ]


def _causal_slice(
    records: list[ModalityData],
    timestamp_list: list[pd.Timestamp],
    timestamp: pd.Timestamp,
) -> list[ModalityData]:
    """
    Return up to MAX_HISTORY observations with timestamp <= t.
    """

    end = bisect_right(
        timestamp_list,
        timestamp,
    )

    start = max(
        0,
        end - MAX_HISTORY,
    )

    return records[start:end]


# ============================================================================
# BATCH TENSOR BUILDING
# ============================================================================

def _pad_record_histories(
    histories: list[list[ModalityData]],
) -> tuple[Tensor, Tensor]:
    """
    Convert a list of variable-length histories to:

        values:
            [batch, max_length, feature_dim]

        lengths:
            [batch]

    Left-padding is used.

    The final timestep is therefore always the most recent causal
    observation for that sample.

    Padding values are zero and are ignored by FiveGRU through lengths.
    """

    if not histories:
        raise RuntimeError(
            "Cannot build tensors from an empty history batch."
        )

    non_empty = [
        history
        for history in histories
        if history
    ]

    if len(non_empty) != len(histories):
        raise RuntimeError(
            "Every emitted causal state must have a non-empty history "
            "for every modality."
        )

    batch_size = len(histories)

    feature_dim = histories[0][0].values.shape[0]

    max_length = max(
        len(history)
        for history in histories
    )

    values = np.zeros(
        (
            batch_size,
            max_length,
            feature_dim,
        ),
        dtype=np.float32,
    )

    lengths = np.zeros(
        batch_size,
        dtype=np.int64,
    )

    for batch_index, history in enumerate(
        histories
    ):
        length = len(history)

        lengths[batch_index] = length

        matrix = np.stack(
            [
                record.values
                for record in history
            ],
            axis=0,
        )

        values[
            batch_index,
            max_length - length:,
            :,
        ] = matrix

    return (
        torch.from_numpy(values),
        torch.from_numpy(lengths),
    )


def _pad_categorical_histories(
    histories: list[list[ModalityData]],
    field: str,
    *,
    max_length: int,
) -> Tensor:
    """
    Left-pad categorical IDs to match the corresponding numeric histories.

    PAD_ID = 0 is reserved for padding.
    """

    batch_size = len(histories)

    values = np.zeros(
        (
            batch_size,
            max_length,
        ),
        dtype=np.int64,
    )

    for batch_index, history in enumerate(
        histories
    ):
        length = len(history)

        ids = [
            int(record.categorical[field])
            for record in history
        ]

        values[
            batch_index,
            max_length - length:,
        ] = np.asarray(
            ids,
            dtype=np.int64,
        )

    return torch.from_numpy(values)


# ============================================================================
# PIPELINE FORWARD
# ============================================================================

def _run_pipeline(
    model: FiveGRUStatePipeline,
    *,
    glucose_input: Tensor,
    insulin_input: Tensor,
    nutrition_input: Tensor,
    activity_input: Tensor,
    sleep_input: Tensor,
    lengths: Mapping[str, Tensor],
) -> Tensor:
    """
    Run the actual locked pipeline.

    CRITICAL:

        FiveGRUStatePipeline.forward()
        accepts exactly ONE FiveGRUInputBatch.

    It returns:

        UnifiedPatientState

    and the actual tensor is:

        patient_state.state
    """

    inputs = FiveGRUInputBatch(
        glucose=glucose_input,
        insulin=insulin_input,
        nutrition=nutrition_input,
        activity=activity_input,
        sleep=sleep_input,
        lengths=lengths,
    )

    patient_state = model(
        inputs
    )

    if not isinstance(
        patient_state,
        UnifiedPatientState,
    ):
        raise RuntimeError(
            "FiveGRUStatePipeline returned an unexpected object. "
            "Expected UnifiedPatientState, received "
            f"{type(patient_state).__name__}."
        )

    state = patient_state.state

    if not isinstance(
        state,
        torch.Tensor,
    ):
        raise RuntimeError(
            "UnifiedPatientState.state is not a torch.Tensor."
        )

    if state.ndim != 2:
        raise RuntimeError(
            "UnifiedPatientState.state must have shape "
            f"[batch, state_dim]. Received {tuple(state.shape)}."
        )

    if state.shape[0] != glucose_input.shape[0]:
        raise RuntimeError(
            "Unified Patient State batch size mismatch."
        )

    if state.shape[1] != HIDDEN_DIM:
        raise RuntimeError(
            "Unified Patient State dimension mismatch. "
            f"Expected {HIDDEN_DIM}, received {state.shape[1]}."
        )

    if not torch.isfinite(state).all():
        raise RuntimeError(
            "Unified Patient State contains NaN or infinite values."
        )

    return state


# ============================================================================
# CAUSAL BATCH GENERATION
# ============================================================================

def generate_for_participant(
    participant_id: str,
    *,
    model: FiveGRUStatePipeline,
    adapters: dict[str, Any],
    vocab: dict[str, CategoricalVocabulary],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """
    Generate one causal Unified Patient State for every eligible timestamp.

    Chunked execution is used for performance.

    No timestamp is emitted until every one of the five modality branches
    has at least one causal observation.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive."
        )

    data = load_participant_data(
        participant_id,
        vocab,
    )

    timestamp_lists = {
        "glucose": _record_timestamps(
            data.glucose
        ),
        "insulin": _record_timestamps(
            data.insulin
        ),
        "nutrition": _record_timestamps(
            data.nutrition
        ),
        "activity": _record_timestamps(
            data.activity
        ),
        "sleep": _record_timestamps(
            data.sleep
        ),
    }

    timestamps = sorted(
        set(
            timestamp
            for records in (
                data.glucose,
                data.insulin,
                data.nutrition,
                data.activity,
                data.sleep,
            )
            for record in records
            for timestamp in (
                record.timestamp,
            )
        )
    )

    print()
    print(
        f"  Causal timestamps: {len(timestamps)}"
    )

    output_records: list[dict[str, Any]] = []

    model.eval()

    with torch.no_grad():

        total_chunks = (
            len(timestamps) + chunk_size - 1
        ) // chunk_size

        emitted = 0

        for chunk_number, chunk_start in enumerate(
            range(
                0,
                len(timestamps),
                chunk_size,
            ),
            start=1,
        ):

            chunk_timestamps = timestamps[
                chunk_start:
                chunk_start + chunk_size
            ]

            histories = {
                "glucose": [],
                "insulin": [],
                "nutrition": [],
                "activity": [],
                "sleep": [],
            }

            eligible_timestamps: list[pd.Timestamp] = []

            for timestamp in chunk_timestamps:

                glucose_history = _causal_slice(
                    data.glucose,
                    timestamp_lists["glucose"],
                    timestamp,
                )

                insulin_history = _causal_slice(
                    data.insulin,
                    timestamp_lists["insulin"],
                    timestamp,
                )

                nutrition_history = _causal_slice(
                    data.nutrition,
                    timestamp_lists["nutrition"],
                    timestamp,
                )

                activity_history = _causal_slice(
                    data.activity,
                    timestamp_lists["activity"],
                    timestamp,
                )

                sleep_history = _causal_slice(
                    data.sleep,
                    timestamp_lists["sleep"],
                    timestamp,
                )

                # Do not fabricate missing modality history.
                if not all(
                    (
                        glucose_history,
                        insulin_history,
                        nutrition_history,
                        activity_history,
                        sleep_history,
                    )
                ):
                    continue

                histories["glucose"].append(
                    glucose_history
                )
                histories["insulin"].append(
                    insulin_history
                )
                histories["nutrition"].append(
                    nutrition_history
                )
                histories["activity"].append(
                    activity_history
                )
                histories["sleep"].append(
                    sleep_history
                )

                eligible_timestamps.append(
                    timestamp
                )

            if not eligible_timestamps:
                continue

            # --------------------------------------------------------------
            # Numeric branches
            # --------------------------------------------------------------

            glucose_numeric, glucose_lengths = (
                _pad_record_histories(
                    histories["glucose"]
                )
            )

            insulin_dose, insulin_lengths = (
                _pad_record_histories(
                    histories["insulin"]
                )
            )

            nutrition_numeric, nutrition_lengths = (
                _pad_record_histories(
                    histories["nutrition"]
                )
            )

            activity_numeric, activity_lengths = (
                _pad_record_histories(
                    histories["activity"]
                )
            )

            sleep_numeric, sleep_lengths = (
                _pad_record_histories(
                    histories["sleep"]
                )
            )

            # --------------------------------------------------------------
            # Categorical branches
            # --------------------------------------------------------------

            nutrition_max_length = (
                nutrition_numeric.shape[1]
            )

            nutrition_meal_type = (
                _pad_categorical_histories(
                    histories["nutrition"],
                    "meal_type",
                    max_length=nutrition_max_length,
                )
            )

            nutrition_meal_tag = (
                _pad_categorical_histories(
                    histories["nutrition"],
                    "meal_tag",
                    max_length=nutrition_max_length,
                )
            )

            activity_max_length = (
                activity_numeric.shape[1]
            )

            activity_type = (
                _pad_categorical_histories(
                    histories["activity"],
                    "activity_type",
                    max_length=activity_max_length,
                )
            )

            activity_intensity = (
                _pad_categorical_histories(
                    histories["activity"],
                    "intensity",
                    max_length=activity_max_length,
                )
            )

            insulin_max_length = (
                insulin_dose.shape[1]
            )

            insulin_event = torch.zeros(
                (
                    len(histories["insulin"]),
                    insulin_max_length,
                ),
                dtype=torch.float32,
            )

            for batch_index, history in enumerate(
                histories["insulin"]
            ):
                length = len(history)

                insulin_event[
                    batch_index,
                    insulin_max_length - length:,
                ] = torch.tensor(
                    [
                        record.categorical["event_type"]
                        for record in history
                    ],
                    dtype=torch.float32,
                )

            # --------------------------------------------------------------
            # Adapter boundary
            # --------------------------------------------------------------

            glucose_input = adapters[
                "glucose"
            ](
                glucose_numeric
            )

            insulin_input = adapters[
                "insulin"
            ](
                insulin_dose,
                insulin_event,
            )

            nutrition_input = adapters[
                "nutrition"
            ](
                nutrition_numeric,
                nutrition_meal_type,
                nutrition_meal_tag,
            )

            activity_input = adapters[
                "activity"
            ](
                activity_numeric,
                activity_type,
                activity_intensity,
            )

            sleep_input = adapters[
                "sleep"
            ](
                sleep_numeric
            )

            # --------------------------------------------------------------
            # Frozen runtime assertions
            # --------------------------------------------------------------

            if glucose_input.shape[-1] != 1:
                raise RuntimeError(
                    "Glucose runtime dimension mismatch."
                )

            if insulin_input.shape[-1] != 2:
                raise RuntimeError(
                    "Insulin runtime dimension mismatch."
                )

            if nutrition_input.shape[-1] != 24:
                raise RuntimeError(
                    "Nutrition runtime dimension mismatch."
                )

            if activity_input.shape[-1] != 17:
                raise RuntimeError(
                    "Activity runtime dimension mismatch."
                )

            if sleep_input.shape[-1] != 6:
                raise RuntimeError(
                    "Sleep runtime dimension mismatch."
                )

            # --------------------------------------------------------------
            # Independent modality sequence lengths
            # --------------------------------------------------------------

            lengths = {
                "glucose": glucose_lengths,
                "insulin": insulin_lengths,
                "nutrition": nutrition_lengths,
                "activity": activity_lengths,
                "sleep": sleep_lengths,
            }

            # --------------------------------------------------------------
            # FiveGRU -> MLPFusion -> UnifiedPatientState
            # --------------------------------------------------------------

            state = _run_pipeline(
                model,
                glucose_input=glucose_input,
                insulin_input=insulin_input,
                nutrition_input=nutrition_input,
                activity_input=activity_input,
                sleep_input=sleep_input,
                lengths=lengths,
            )

            state_np = (
                state
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if state_np.shape[1] != HIDDEN_DIM:
                raise RuntimeError(
                    "Generated state dimension mismatch."
                )

            # --------------------------------------------------------------
            # Preserve timestamp ordering.
            # --------------------------------------------------------------

            for batch_index, timestamp in enumerate(
                eligible_timestamps
            ):
                row = {
                    "participant_id": participant_id,
                    "timestamp": timestamp.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }

                for dimension in range(
                    HIDDEN_DIM
                ):
                    row[
                        f"state_{dimension:02d}"
                    ] = float(
                        state_np[
                            batch_index,
                            dimension,
                        ]
                    )

                output_records.append(
                    row
                )

                emitted += 1

            print(
                f"  chunk {chunk_number}/{total_chunks} "
                f"| emitted {emitted} states",
                end="\r",
            )

    print()

    if not output_records:
        raise RuntimeError(
            f"{participant_id}: no complete causal states could be generated."
        )

    return output_records


# ============================================================================
# SELF TEST
# ============================================================================

def self_test() -> None:
    """
    Fast structural/runtime self-test.

    No dataset files are read.
    """

    print()
    print("=" * 80)
    print("STATE GENERATION MODEL SELF-TEST")
    print("=" * 80)

    torch.manual_seed(42)

    model = build_model()

    adapters = {
        "glucose": IdentityNumericAdapter(1),
        "insulin": InsulinAdapter(),
        "nutrition": NutritionAdapter(
            meal_type_cardinality=16,
            meal_tag_cardinality=1096,
        ),
        "activity": ActivityAdapter(
            activity_type_cardinality=9,
            intensity_cardinality=6,
        ),
        "sleep": IdentityNumericAdapter(6),
    }

    batch = 2

    # Deliberately use independent sequence lengths.
    glucose = torch.randn(
        batch,
        5,
        1,
    )

    insulin_dose = torch.randn(
        batch,
        4,
        1,
    )

    insulin_event = torch.randint(
        0,
        2,
        (
            batch,
            4,
        ),
    ).to(torch.float32)

    nutrition_numeric = torch.randn(
        batch,
        3,
        4,
    )

    nutrition_meal_type = torch.randint(
        0,
        16,
        (
            batch,
            3,
        ),
        dtype=torch.long,
    )

    nutrition_meal_tag = torch.randint(
        0,
        1096,
        (
            batch,
            3,
        ),
        dtype=torch.long,
    )

    activity_numeric = torch.randn(
        batch,
        6,
        10,
    )

    activity_type = torch.randint(
        0,
        9,
        (
            batch,
            6,
        ),
        dtype=torch.long,
    )

    activity_intensity = torch.randint(
        0,
        6,
        (
            batch,
            6,
        ),
        dtype=torch.long,
    )

    sleep = torch.randn(
        batch,
        7,
        6,
    )

    glucose_input = adapters[
        "glucose"
    ](
        glucose
    )

    insulin_input = adapters[
        "insulin"
    ](
        insulin_dose,
        insulin_event,
    )

    nutrition_input = adapters[
        "nutrition"
    ](
        nutrition_numeric,
        nutrition_meal_type,
        nutrition_meal_tag,
    )

    activity_input = adapters[
        "activity"
    ](
        activity_numeric,
        activity_type,
        activity_intensity,
    )

    sleep_input = adapters[
        "sleep"
    ](
        sleep
    )

    lengths = {
        "glucose": torch.tensor(
            [5, 4],
            dtype=torch.long,
        ),
        "insulin": torch.tensor(
            [4, 3],
            dtype=torch.long,
        ),
        "nutrition": torch.tensor(
            [3, 2],
            dtype=torch.long,
        ),
        "activity": torch.tensor(
            [6, 5],
            dtype=torch.long,
        ),
        "sleep": torch.tensor(
            [7, 6],
            dtype=torch.long,
        ),
    }

    inputs = FiveGRUInputBatch(
        glucose=glucose_input,
        insulin=insulin_input,
        nutrition=nutrition_input,
        activity=activity_input,
        sleep=sleep_input,
        lengths=lengths,
    )

    with torch.no_grad():
        patient_state = model(
            inputs
        )

    if not isinstance(
        patient_state,
        UnifiedPatientState,
    ):
        raise RuntimeError(
            "Self-test failed: pipeline did not return "
            "UnifiedPatientState."
        )

    if patient_state.state.shape != (
        batch,
        HIDDEN_DIM,
    ):
        raise RuntimeError(
            "Self-test failed: unexpected state shape "
            f"{tuple(patient_state.state.shape)}."
        )

    if not torch.isfinite(
        patient_state.state
    ).all():
        raise RuntimeError(
            "Self-test failed: state contains non-finite values."
        )

    print(
        f"FiveGRU hidden dimension : {HIDDEN_DIM}"
    )
    print(
        f"Unified state dimension  : {HIDDEN_DIM}"
    )
    print(
        f"Maximum causal history  : {MAX_HISTORY}"
    )
    print(
        "Five branches            : PASS"
    )
    print(
        "Nutrition runtime dim    : 24"
    )
    print(
        "Activity runtime dim     : 17"
    )
    print(
        "Insulin runtime dim      : 2"
    )
    print(
        "Sleep runtime dim        : 6"
    )
    print(
        "FiveGRUInputBatch        : PASS"
    )
    print(
        "UnifiedPatientState      : PASS"
    )
    print(
        "Architecture             : FiveGRU -> MLPFusion"
    )
    print(
        "SELF-TEST                : PASS"
    )
    print()


# ============================================================================
# PARTICIPANT DISCOVERY
# ============================================================================

def discover_participants() -> list[str]:
    participants: set[str] = set()

    if not SOURCE_ROOT.exists():
        raise RuntimeError(
            f"Sequence-input directory does not exist:\n"
            f"{SOURCE_ROOT}"
        )

    for path in SOURCE_ROOT.rglob("*.csv"):
        try:
            participants.add(
                identify_participant(path)
            )
        except RuntimeError:
            continue

    return sorted(participants)


# ============================================================================
# OUTPUT
# ============================================================================

def write_trajectory(
    participant_id: str,
    records: list[dict[str, Any]],
) -> Path:

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / f"{participant_id}_unified_state.csv"
    )

    df = pd.DataFrame(
        records
    )

    expected_columns = [
        "participant_id",
        "timestamp",
    ] + [
        f"state_{dimension:02d}"
        for dimension in range(HIDDEN_DIM)
    ]

    missing = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Generated trajectory is missing expected columns: "
            f"{missing}"
        )

    df = df[
        expected_columns
    ]

    df.to_csv(
        output_path,
        index=False,
    )

    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate real T1D-UOM unified patient-state trajectories."
        )
    )

    parser.add_argument(
        "--participant",
        type=str,
        default=None,
        help="Participant ID, e.g. UoM2301",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate states for all discovered participants.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Number of causal timestamps processed per model "
            f"forward chunk. Default: {DEFAULT_CHUNK_SIZE}."
        ),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.chunk_size <= 0:
        parser.error(
            "--chunk-size must be a positive integer."
        )

    print()
    print("=" * 80)
    print("T1D-UOM REAL UNIFIED PATIENT STATE GENERATION")
    print("=" * 80)

    print()
    print("Architecture:")
    print(
        "  FiveGRU -> MLPFusion -> Unified Patient State"
    )

    print()
    print(
        f"FiveGRU hidden dimension : {HIDDEN_DIM}"
    )
    print(
        f"Unified state dimension  : {HIDDEN_DIM}"
    )
    print(
        f"Maximum causal history  : {MAX_HISTORY}"
    )
    print(
        f"Causal batch chunk size : {args.chunk_size}"
    )

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    print()
    print(
        "Loading training-partition categorical vocabularies..."
    )

    try:
        vocab = load_vocabulary_artifact()
    except Exception as exc:
        print()
        print(
            "VOCABULARY LOADING FAILED"
        )
        print("=" * 80)
        print(str(exc))
        print()
        print(
            "Do NOT fit vocabularies from the target participant."
        )
        sys.exit(1)

    print()
    print(
        "Vocabulary cardinalities:"
    )

    for field in REQUIRED_VOCAB_FIELDS:
        print(
            f"  {field:16s}: "
            f"{vocab[field].size}"
        )

    # ------------------------------------------------------------------
    # Model and adapters
    # ------------------------------------------------------------------

    adapters = build_adapters(
        vocab
    )

    model = build_model()

    print()
    print(
        "Runtime modality contract:"
    )
    print(
        "  glucose   : 1"
    )
    print(
        "  insulin   : 2"
    )
    print(
        "  nutrition : 24"
    )
    print(
        "  activity  : 17"
    )
    print(
        "  sleep     : 6"
    )

    print()
    print(
        "Critical adapter path:"
    )
    print(
        "  Nutrition 4 -> NutritionAdapter -> 24"
    )
    print(
        "  Activity 10 -> ActivityAdapter -> 17"
    )
    print(
        "  Insulin 1+event -> InsulinAdapter -> 2"
    )
    print(
        "  Sleep 6 -> IdentityNumericAdapter -> 6"
    )

    # ------------------------------------------------------------------
    # Participants
    # ------------------------------------------------------------------

    if args.participant:
        participants = [
            args.participant.strip()
        ]

    elif args.all:
        participants = discover_participants()

    else:
        parser.error(
            "Specify --participant UoMxxxx or --all."
        )

    if not participants:
        raise RuntimeError(
            "No participants were selected."
        )

    print()
    print(
        f"Participants             : {len(participants)}"
    )

    print()
    print(
        "Frozen source policy:"
    )
    print(
        "  Source modification : NO"
    )
    print(
        "  Resampling          : NO"
    )
    print(
        "  Interpolation       : NO"
    )
    print(
        "  Imputation          : NO"
    )
    print(
        "  Normalization       : NO"
    )
    print(
        "  Vocabulary fitting  : NO"
    )

    failures: list[tuple[str, str]] = []

    for index, participant_id in enumerate(
        participants,
        start=1,
    ):

        print()
        print(
            f"[{index:02d}/{len(participants):02d}] "
            f"{participant_id}"
        )

        try:
            records = generate_for_participant(
                participant_id,
                model=model,
                adapters=adapters,
                vocab=vocab,
                chunk_size=args.chunk_size,
            )

            output_path = write_trajectory(
                participant_id,
                records,
            )

            print()
            print(
                f"[{participant_id}] SUCCESS"
            )
            print(
                f"  states generated : {len(records)}"
            )
            print(
                f"  state dimension  : {HIDDEN_DIM}"
            )
            print(
                f"  output           : {output_path}"
            )

        except Exception as exc:

            failures.append(
                (
                    participant_id,
                    str(exc),
                )
            )

            print()
            print("=" * 80)
            print(
                f"STATE GENERATION FAILED: "
                f"{participant_id}"
            )
            print("=" * 80)
            print(
                str(exc)
            )

    print()
    print("=" * 80)

    if failures:

        print(
            "REAL STATE GENERATION COMPLETED WITH FAILURES"
        )
        print("=" * 80)

        for participant_id, error in failures:
            print()
            print(
                f"{participant_id}: {error}"
            )

        print()
        print(
            "IMPORTANT: frozen source data was not modified."
        )

        sys.exit(1)

    print(
        "REAL STATE GENERATION COMPLETED SUCCESSFULLY"
    )
    print("=" * 80)
    print()
    print(
        f"Generated {len(participants)} participant trajectory "
        f"artifact(s)."
    )
    print(
        f"Output directory: {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()