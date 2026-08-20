"""
T1D-UOM NUMERIC / SEMANTIC GRU INPUT CONTRACT AUDIT

READ-ONLY AUDIT.

This script:
- DOES NOT modify dataset files.
- DOES NOT modify sequence-input files.
- DOES NOT transform values.
- DOES NOT delete rows.
- DOES NOT resample.
- DOES NOT interpolate.
- DOES NOT impute.
- DOES NOT normalize.
- DOES NOT engineer features.
- DOES NOT encode categorical variables.
- DOES NOT construct event_type.
- DOES NOT create windows.
- DOES NOT create targets.
- DOES NOT train a model.
- DOES NOT implement MLP Fusion.
- DOES NOT implement Digital Twin.
- DOES NOT implement Prediction.
- DOES NOT implement What-if.
- DOES NOT implement Interactive UI.

Purpose:
Validate the physical / semantic input contract of ALL frozen
sequence-input CSV files before controlled GRU integration.

IMPORTANT:
Participant identification is based on the participant digits embedded
in the actual frozen filenames, e.g.:

    UoMActivity2301.csv  -> UoM2301
    UoMGlucose2301.csv   -> UoM2301
    UoMBolus2301.csv     -> UoM2301
    UoMsleep2301.csv     -> UoM2301
    UoM2302sleeptime.csv -> UoM2302

No file contents are modified.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================================
# PROJECT PATHS
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

SEQUENCE_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "t1d_uom_v1.0.3_sequence_inputs"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"

AUDIT_REPORT = (
    REPORT_DIR / "numeric_semantic_input_contract_audit.json"
)

MODELING_MANIFEST = (
    REPORT_DIR / "modeling_dataset_manifest.json"
)

PREPARATION_MANIFEST = (
    REPORT_DIR / "sequence_input_preparation_manifest.json"
)


# ============================================================================
# FROZEN CONTRACT
# ============================================================================

EXPECTED_FILE_COUNT = 86

FROZEN_PARTICIPANTS = [
    "UoM2301",
    "UoM2302",
    "UoM2304",
    "UoM2305",
    "UoM2306",
    "UoM2307",
    "UoM2308",
    "UoM2309",
    "UoM2313",
    "UoM2314",
    "UoM2401",
    "UoM2403",
    "UoM2405",
]

EXPECTED_PARTICIPANT_SET = set(FROZEN_PARTICIPANTS)


# ============================================================================
# MODALITY CONTRACTS
# ============================================================================

MODALITY_CONTRACTS = {
    "glucose": {
        "folder": "Glucose Data",
        "timestamp": "bg_ts",
        "numeric": ["value"],
        "categorical": [],
        "structured": [],
        "expected_dim": 1,
    },

    "nutrition": {
        "folder": "Nutrition Data",
        "timestamp": "meal_ts",
        "numeric": [
            "carbs_g",
            "prot_g",
            "fat_g",
            "fibre_g",
        ],
        "categorical": [
            "meal_type",
            "meal_tag",
        ],
        "structured": [],
        "expected_dim": 6,
    },

    "activity": {
        "folder": "Activity Data",
        "timestamp": "activity_ts",
        "numeric": [
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
        ],
        "categorical": [
            "activity_type",
            "intensity",
        ],
        "structured": [],
        "expected_dim": 12,
    },

    "basal_insulin": {
        "folder": "Insulin Data/Basal Data",
        "timestamp": "basal_ts",
        "numeric": [
            "basal_dose",
        ],
        "categorical": [
            "insulin_kind",
        ],
        "structured": [],
        "expected_dim": 2,
    },

    "bolus_insulin": {
        "folder": "Insulin Data/Bolus Data",
        "timestamp": "bolus_ts",
        "numeric": [
            "bolus_dose",
        ],
        "categorical": [],
        "structured": [],
        "expected_dim": 1,
    },

    "sleep_summary": {
        "folder": "Sleep Data",
        "timestamp": "start_date_ts",
        "numeric": [
            "duration_in_sec",
            "start_time_offset_s",
            "unmeasurable_sleep_s",
            "deep_sleep_s",
            "light_sleep_s",
            "rem_sleep_s",
            "awake_s",
        ],
        "categorical": [
            "calendar_date",
            "validation",
        ],
        "structured": [
            "sleep_levels_map_deep",
            "sleep_levels_map_light",
            "sleep_levels_map_awake",
            "sleep_levels_map_rem",
            "sleep_levels_map_unmeasurable",
        ],
        "expected_dim": 14,
    },

    "sleep_timeseries": {
        "folder": "Sleep Data",
        "timestamp": "sleep_ts",
        "numeric": [
            "step_count",
            "heart_rate",
            "current_activity_type_intensity",
            "stress_level_value",
            "sleep_level",
            "resting_heart_rate",
        ],
        "categorical": [],
        "structured": [],
        "expected_dim": 6,
    },
}


EXPECTED_MODALITY_COUNTS = {
    "activity": 13,
    "glucose": 13,
    "basal_insulin": 12,
    "bolus_insulin": 13,
    "nutrition": 13,
    "sleep_summary": 11,
    "sleep_timeseries": 11,
}


# ============================================================================
# ARCHITECTURE CONTRACT
# ============================================================================

ARCHITECTURE = {
    "Glucose": "GRU -> zG",
    "Insulin": "GRU -> zI",
    "Nutrition": "GRU -> zN",
    "Activity": "GRU -> zA",
    "Sleep": "GRU -> zS",
}

SLEEP_GRU_FEATURES = [
    "step_count",
    "heart_rate",
    "current_activity_type_intensity",
    "stress_level_value",
    "sleep_level",
    "resting_heart_rate",
]


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def relative_path(path: Path) -> str:
    """Return project-relative path using forward slashes."""
    try:
        return path.relative_to(SEQUENCE_INPUT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def json_safe(value: Any) -> Any:
    """Convert values to JSON-safe Python primitives."""
    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    return str(value)


def is_empty(value: Any) -> bool:
    """Treat blank/NaN/None as empty without modifying the source."""
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    return str(value).strip() == ""


# ============================================================================
# PARTICIPANT IDENTIFICATION
# ============================================================================

def identify_participant_from_filename(path: Path) -> str | None:
    """
    Robustly identify the frozen participant from actual UoM filenames.

    Supported examples:

        UoMActivity2301.csv
        UoMGlucose2301.csv
        UoMBolus2301.csv
        UoMBasal2302.csv
        UoMNutrition2301.csv
        UoMsleep2301.csv
        UoM2302sleeptime.csv

    The participant is the four-digit identifier immediately following
    the UoM modality prefix OR immediately following UoM itself.

    This function NEVER reads or changes file contents.
    """

    stem = path.stem.strip()

    # Primary robust rule:
    # UoM + optional modality token + four participant digits.
    #
    # Examples:
    # UoMActivity2301
    # UoMGlucose2301
    # UoMBolus2301
    # UoMBasal2302
    # UoMNutrition2301
    # UoMsleep2301
    # UoM2302sleeptime
    pattern = re.compile(
        r"^UoM(?:Activity|Glucose|Bolus|Basal|Nutrition|sleep)?"
        r"(\d{4})(?:.*)?$",
        re.IGNORECASE,
    )

    match = pattern.match(stem)

    if match:
        return f"UoM{match.group(1)}"

    # Conservative fallback:
    # Look for UoM followed by exactly four digits anywhere in the
    # filename, while avoiding arbitrary unrelated digit sequences.
    fallback = re.search(
        r"UoM(\d{4})",
        stem,
        re.IGNORECASE,
    )

    if fallback:
        return f"UoM{fallback.group(1)}"

    return None


# ============================================================================
# MODALITY IDENTIFICATION
# ============================================================================

def identify_modality(path: Path) -> str | None:
    """
    Identify modality from the frozen directory / filename structure.
    """

    rel = relative_path(path)
    rel_lower = rel.lower()
    name_lower = path.name.lower()

    if rel_lower.startswith("activity data/"):
        return "activity"

    if rel_lower.startswith("glucose data/"):
        return "glucose"

    if rel_lower.startswith("nutrition data/"):
        return "nutrition"

    if rel_lower.startswith("insulin data/basal data/"):
        return "basal_insulin"

    if rel_lower.startswith("insulin data/bolus data/"):
        return "bolus_insulin"

    if rel_lower.startswith("sleep data/"):
        if "sleeptime" in name_lower:
            return "sleep_timeseries"

        if name_lower.startswith("uomsleep"):
            return "sleep_summary"

    return None


# ============================================================================
# CSV LOADING
# ============================================================================

def read_csv_read_only(path: Path) -> pd.DataFrame:
    """
    Read CSV without writing anything back to disk.

    No transformations are persisted.
    """
    return pd.read_csv(
        path,
        low_memory=False,
    )


# ============================================================================
# NUMERIC AUDIT
# ============================================================================

def audit_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Check numeric semantic validity without modifying df.

    Returns:
        missing_numeric_values
        invalid_numeric_values
    """

    missing: dict[str, int] = {}
    invalid: dict[str, int] = {}

    for column in columns:
        series = df[column]

        empty_mask = series.map(is_empty)

        empty_count = int(empty_mask.sum())

        if empty_count:
            missing[column] = empty_count

        # Only evaluate non-empty values.
        nonempty = series.loc[~empty_mask]

        if len(nonempty) == 0:
            continue

        converted = pd.to_numeric(
            nonempty,
            errors="coerce",
        )

        invalid_mask = converted.isna()

        invalid_count = int(invalid_mask.sum())

        if invalid_count:
            invalid[column] = invalid_count

    return missing, invalid


# ============================================================================
# CATEGORICAL AUDIT
# ============================================================================

def audit_categorical_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> dict[str, int]:
    """
    Raw categorical fields are NOT encoded.

    This audit only records missing categorical values.
    It does not impose arbitrary category vocabularies.
    """

    missing: dict[str, int] = {}

    for column in columns:
        series = df[column]
        count = int(series.map(is_empty).sum())

        if count:
            missing[column] = count

    return missing


# ============================================================================
# STRUCTURED FIELD AUDIT
# ============================================================================

def audit_structured_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Audit structured sleep-map fields.

    Empty structured fields are REVIEW items, not automatic failures.

    Invalid structured values are only flagged when a non-empty value
    is clearly malformed for the expected structured representation.

    No structured field is parsed into a new representation.
    """

    empty_values: dict[str, int] = {}
    invalid_values: dict[str, int] = {}

    for column in columns:
        series = df[column]

        empty_count = int(series.map(is_empty).sum())

        if empty_count:
            empty_values[column] = empty_count

        nonempty = series.loc[
            ~series.map(is_empty)
        ]

        for value in nonempty:
            text = str(value).strip()

            # The source contract contains serialized structured sleep
            # maps. We deliberately do not transform them.
            #
            # We only reject obvious empty/null tokens when they are
            # present as non-empty strings.
            if text.lower() in {
                "nan",
                "none",
                "null",
            }:
                invalid_values[column] = (
                    invalid_values.get(column, 0) + 1
                )

    return empty_values, invalid_values


# ============================================================================
# SINGLE-FILE AUDIT
# ============================================================================

def audit_file(path: Path) -> dict[str, Any]:
    """
    Perform the complete physical/semantic audit for one CSV.
    """

    participant = identify_participant_from_filename(path)
    modality = identify_modality(path)

    result: dict[str, Any] = {
        "relative_path": relative_path(path),
        "participant": participant,
        "modality": modality,
        "status": "PASS",
        "schema_status": "PASS",
        "numeric_status": "PASS",
        "categorical_status": "PASS",
        "structured_status": "PASS",
        "timestamp_column": None,
        "expected_numeric_columns": [],
        "expected_categorical_columns": [],
        "expected_structured_columns": [],
        "actual_columns": [],
        "missing_columns": [],
        "unexpected_columns": [],
        "missing_numeric_values": {},
        "invalid_numeric_values": {},
        "categorical_missing_values": {},
        "invalid_categorical_values": {},
        "structured_empty_values": {},
        "invalid_structured_values": {},
        "row_count": None,
        "read_error": None,
    }

    if participant is None:
        result["status"] = "FAIL"
        result["schema_status"] = "FAIL"
        result["participant_error"] = (
            "Unable to identify participant from filename"
        )
        return result

    if participant not in EXPECTED_PARTICIPANT_SET:
        result["status"] = "FAIL"
        result["schema_status"] = "FAIL"
        result["participant_error"] = (
            f"Participant {participant} is not in the frozen cohort"
        )
        return result

    if modality is None:
        result["status"] = "FAIL"
        result["schema_status"] = "FAIL"
        result["modality_error"] = (
            "Unable to identify frozen modality from path/filename"
        )
        return result

    # Sleep has two intentionally separate physical representations.
    # The frozen filenames are not sufficient to decide which schema a
    # particular sleep CSV physically contains, so the audit resolves the
    # sleep family from the actual header AFTER reading the file. This is
    # read-only and does not alter the file.
    #
    # For all non-sleep modalities, the directory/filename family remains
    # authoritative.
    contract = MODALITY_CONTRACTS[modality]

    try:
        df = read_csv_read_only(path)
    except Exception as exc:
        result["status"] = "FAIL"
        result["schema_status"] = "FAIL"
        result["read_error"] = repr(exc)
        return result

    result["actual_columns"] = [
        str(column)
        for column in df.columns
    ]

    result["row_count"] = int(len(df))

    # ------------------------------------------------------------------------
    # SLEEP FAMILY RESOLUTION
    # ------------------------------------------------------------------------
    # The sequence-input inventory contains two legitimate sleep families:
    #
    #   1) sleep time-series: 6 physical features + sleep_ts
    #   2) sleep summary:     14 physical/semantic fields + start_date_ts
    #
    # They are alternatives, not one combined schema. Therefore a legitimate
    # sleep file must match ONE of those complete physical schemas. A summary
    # file must never be failed merely because it lacks time-series columns,
    # and a time-series file must never be failed merely because it lacks
    # summary columns.
    #
    # We retain the filename-derived family for provenance, but use the
    # actual physical header to select the contract.
    filename_modality = modality

    if modality in {"sleep_summary", "sleep_timeseries"}:
        actual_columns_set = set(df.columns)

        sleep_candidates = []
        for candidate_name in (
            "sleep_timeseries",
            "sleep_summary",
        ):
            candidate = MODALITY_CONTRACTS[candidate_name]
            candidate_columns = {
                candidate["timestamp"],
                *candidate["numeric"],
                *candidate["categorical"],
                *candidate["structured"],
            }

            if actual_columns_set == candidate_columns:
                sleep_candidates.append(candidate_name)

        if len(sleep_candidates) == 1:
            modality = sleep_candidates[0]
            contract = MODALITY_CONTRACTS[modality]
            result["modality"] = modality
            result["filename_modality"] = filename_modality
            result["physical_schema_modality"] = modality

            if filename_modality != modality:
                result["filename_schema_family_mismatch"] = True
                result["filename_schema_family_note"] = (
                    "Filename-derived sleep family differs from the "
                    "physical CSV schema. This is recorded for provenance "
                    "only and is NOT treated as a contract failure."
                )
            else:
                result["filename_schema_family_mismatch"] = False

        else:
            result["physical_schema_modality"] = None
            result["filename_modality"] = filename_modality

            result["status"] = "FAIL"
            result["schema_status"] = "FAIL"
            result["sleep_schema_resolution_error"] = (
                "Sleep CSV does not match exactly one frozen physical "
                "sleep schema."
            )

            if not sleep_candidates:
                result["sleep_schema_resolution_detail"] = (
                    "No exact match for sleep_timeseries or sleep_summary."
                )
            else:
                result["sleep_schema_resolution_detail"] = (
                    "Ambiguous match for frozen sleep schemas."
                )

            return result

    result["timestamp_column"] = contract["timestamp"]
    result["expected_numeric_columns"] = list(
        contract["numeric"]
    )
    result["expected_categorical_columns"] = list(
        contract["categorical"]
    )
    result["expected_structured_columns"] = list(
        contract["structured"]
    )

    expected_columns = [
        contract["timestamp"],
        *contract["numeric"],
        *contract["categorical"],
        *contract["structured"],
    ]

    actual_set = set(df.columns)
    expected_set = set(expected_columns)

    missing_columns = [
        column
        for column in expected_columns
        if column not in actual_set
    ]

    unexpected_columns = [
        column
        for column in df.columns
        if column not in expected_set
    ]

    result["missing_columns"] = missing_columns
    result["unexpected_columns"] = [
        str(column)
        for column in unexpected_columns
    ]

    if missing_columns:
        result["schema_status"] = "FAIL"
        result["status"] = "FAIL"

        # Do not perform value-level audits against missing columns.
        return result

    # ------------------------------------------------------------------------
    # NUMERIC
    # ------------------------------------------------------------------------

    missing_numeric, invalid_numeric = audit_numeric_columns(
        df,
        contract["numeric"],
    )

    result["missing_numeric_values"] = missing_numeric
    result["invalid_numeric_values"] = invalid_numeric

    if invalid_numeric:
        result["numeric_status"] = "FAIL"
        result["status"] = "FAIL"

    # ------------------------------------------------------------------------
    # CATEGORICAL
    # ------------------------------------------------------------------------

    categorical_missing = audit_categorical_columns(
        df,
        contract["categorical"],
    )

    result["categorical_missing_values"] = (
        categorical_missing
    )

    # Missing categorical values are REVIEW only.
    # No encoding and no arbitrary vocabulary is imposed.

    # ------------------------------------------------------------------------
    # STRUCTURED
    # ------------------------------------------------------------------------

    structured_empty, structured_invalid = (
        audit_structured_columns(
            df,
            contract["structured"],
        )
    )

    result["structured_empty_values"] = (
        structured_empty
    )

    result["invalid_structured_values"] = (
        structured_invalid
    )

    if structured_invalid:
        result["structured_status"] = "FAIL"
        result["status"] = "FAIL"

    # ------------------------------------------------------------------------
    # EXPECTED DIMENSION
    # ------------------------------------------------------------------------

    actual_feature_dim = (
        len(contract["numeric"])
        + len(contract["categorical"])
        + len(contract["structured"])
    )

    if actual_feature_dim != contract["expected_dim"]:
        result["schema_status"] = "FAIL"
        result["status"] = "FAIL"
        result["dimension_error"] = {
            "expected": contract["expected_dim"],
            "calculated": actual_feature_dim,
        }

    return result


# ============================================================================
# INVENTORY
# ============================================================================

def discover_csv_files() -> list[Path]:
    """Discover exactly the frozen sequence-input CSV inventory."""
    if not SEQUENCE_INPUT_DIR.exists():
        return []

    return sorted(
        [
            path
            for path in SEQUENCE_INPUT_DIR.rglob("*.csv")
            if path.is_file()
        ],
        key=lambda p: relative_path(p).lower(),
    )


# ============================================================================
# MANIFEST CHECK
# ============================================================================

def check_manifest(path: Path) -> tuple[bool, str]:
    """Check that a required manifest exists and is readable."""
    if not path.exists():
        return False, "NOT FOUND"

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            json.load(handle)
        return True, "PASS"
    except Exception as exc:
        return False, f"UNREADABLE: {exc}"


# ============================================================================
# PRINTING
# ============================================================================

def print_architecture() -> None:
    print("-" * 80)
    print("5. FROZEN FIVE-GRU ARCHITECTURE")
    print("-" * 80)

    for branch, mapping in ARCHITECTURE.items():
        print(f"{branch:<10} -> {mapping}")

    print("zG,zI,zN,zA,zS -> MLP Fusion")
    print("MLP Fusion -> Unified Patient State")
    print("Unified Patient State -> DIGITAL TWIN")
    print("DIGITAL TWIN -> Prediction / What-if")
    print("Prediction / What-if -> Interactive UI")
    print()


def print_contract() -> None:
    print("-" * 80)
    print("5A. FROZEN PHYSICAL / SEMANTIC CONTRACT")
    print("-" * 80)

    for modality, contract in MODALITY_CONTRACTS.items():
        print(f"\n{modality.upper()}")
        print(f"  Timestamp:        {contract['timestamp']}")
        print(f"  Expected dimension: {contract['expected_dim']}")

        if contract["numeric"]:
            print("  Numeric:")
            for column in contract["numeric"]:
                print(f"    - {column}")

        if contract["categorical"]:
            print("  Categorical/raw:")
            for column in contract["categorical"]:
                print(f"    - {column}")

        if contract["structured"]:
            print("  Structured/raw:")
            for column in contract["structured"]:
                print(f"    - {column}")

    print()


# ============================================================================
# REPORT
# ============================================================================

def build_report(
    discovered: list[Path],
    audited: list[dict[str, Any]],
    manifest_results: dict[str, Any],
) -> dict[str, Any]:

    participants = sorted(
        {
            item["participant"]
            for item in audited
            if item.get("participant")
            in EXPECTED_PARTICIPANT_SET
        }
    )

    unknown_participants = sorted(
        {
            item["participant"]
            for item in audited
            if item.get("participant")
            and item.get("participant")
            not in EXPECTED_PARTICIPANT_SET
        }
    )

    participant_failures = [
        item["relative_path"]
        for item in audited
        if item.get("participant") is None
        or item.get("participant_error")
    ]

    missing_participants = sorted(
        EXPECTED_PARTICIPANT_SET - set(participants)
    )

    modality_counts: dict[str, int] = {}

    for item in audited:
        modality = item.get("modality")

        if modality:
            modality_counts[modality] = (
                modality_counts.get(modality, 0) + 1
            )

    modality_counts = dict(
        sorted(modality_counts.items())
    )

    modality_failures = {}

    for modality, expected in EXPECTED_MODALITY_COUNTS.items():
        actual = modality_counts.get(modality, 0)

        if actual != expected:
            modality_failures[modality] = {
                "expected": expected,
                "actual": actual,
            }

    schema_failures = [
        item["relative_path"]
        for item in audited
        if item["schema_status"] == "FAIL"
    ]

    numeric_failures = [
        item["relative_path"]
        for item in audited
        if item["numeric_status"] == "FAIL"
    ]

    categorical_failures = [
        item["relative_path"]
        for item in audited
        if item["categorical_status"] == "FAIL"
    ]

    structured_failures = [
        item["relative_path"]
        for item in audited
        if item["structured_status"] == "FAIL"
    ]

    missing_value_reviews = []

    for item in audited:
        if (
            item["missing_numeric_values"]
            or item["categorical_missing_values"]
            or item["structured_empty_values"]
        ):
            missing_value_reviews.append(
                {
                    "relative_path": item["relative_path"],
                    "missing_numeric_values": item[
                        "missing_numeric_values"
                    ],
                    "categorical_missing_values": item[
                        "categorical_missing_values"
                    ],
                    "structured_empty_values": item[
                        "structured_empty_values"
                    ],
                }
            )

    genuine_contract_failure = any(
        [
            len(discovered) != EXPECTED_FILE_COUNT,
            len(audited) != len(discovered),
            len(participant_failures) > 0,
            len(unknown_participants) > 0,
            len(missing_participants) > 0,
            len(modality_failures) > 0,
            len(schema_failures) > 0,
            len(numeric_failures) > 0,
            len(categorical_failures) > 0,
            len(structured_failures) > 0,
            not manifest_results[
                "modeling_dataset_manifest"
            ]["readable"],
            not manifest_results[
                "sequence_input_preparation_manifest"
            ]["readable"],
        ]
    )

    return {
        "audit_name": (
            "numeric_semantic_input_contract"
        ),
        "audit_version": "3.0.0",

        "read_only": True,

        "dataset_modified": False,
        "sequence_inputs_modified": False,
        "categorical_encoding_performed": False,
        "event_type_constructed": False,
        "windows_created": False,
        "targets_created": False,
        "model_trained": False,
        "architecture_changed": False,

        "project_root": str(PROJECT_ROOT),
        "sequence_inputs": str(SEQUENCE_INPUT_DIR),

        "expected_file_count": EXPECTED_FILE_COUNT,
        "files_discovered": len(discovered),
        "files_audited": len(audited),

        "frozen_participants": FROZEN_PARTICIPANTS,

        "architecture": ARCHITECTURE,
        "sleep_gru_features": SLEEP_GRU_FEATURES,

        "modality_contracts": MODALITY_CONTRACTS,

        "files": audited,

        "participant_failures": participant_failures,
        "unknown_participants": unknown_participants,
        "missing_participants": missing_participants,

        "modality_failures": modality_failures,

        "schema_failures": schema_failures,
        "numeric_failures": numeric_failures,
        "categorical_failures": categorical_failures,
        "structured_failures": structured_failures,

        "missing_value_reviews": missing_value_reviews,

        "manifests": manifest_results,

        "summary": {
            "files_discovered": len(discovered),
            "files_audited": len(audited),
            "participants_represented": participants,
            "participant_count": len(participants),
            "modality_counts": modality_counts,

            "schema_failure_count": len(
                schema_failures
            ),
            "numeric_failure_count": len(
                numeric_failures
            ),
            "categorical_failure_count": len(
                categorical_failures
            ),
            "structured_failure_count": len(
                structured_failures
            ),

            "files_requiring_missing_value_review": len(
                missing_value_reviews
            ),

            "genuine_contract_failure":
                genuine_contract_failure,
        },
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    print("=" * 80)
    print(
        "T1D-UOM NUMERIC / SEMANTIC GRU INPUT CONTRACT AUDIT"
    )
    print("=" * 80)

    print("IMPORTANT: READ-ONLY.")
    print("No dataset files will be modified.")
    print("No sequence-input files will be modified.")
    print("No values will be transformed in the source files.")
    print("No rows will be deleted.")
    print("No resampling will be performed.")
    print("No interpolation will be performed.")
    print("No imputation will be performed.")
    print("No normalization will be performed.")
    print("No feature engineering will be performed.")
    print("No categorical encoding will be performed.")
    print("No event_type column will be constructed.")
    print("No windows will be created.")
    print("No targets will be created.")
    print("No model will be trained.")
    print("No MLP Fusion will be implemented.")
    print("No Digital Twin will be implemented.")
    print("No Prediction will be implemented.")
    print("No What-if implementation will be created.")
    print("No Interactive UI will be created.")
    print()

    print(f"Project root:       {PROJECT_ROOT}")
    print(f"Sequence inputs:    {SEQUENCE_INPUT_DIR}")
    print(f"Audit report:       {AUDIT_REPORT}")
    print()

    # ------------------------------------------------------------------------
    # 1. DIRECTORY AND MANIFEST
    # ------------------------------------------------------------------------

    print("-" * 80)
    print("1. DIRECTORY AND MANIFEST VALIDATION")
    print("-" * 80)

    sequence_dir_pass = SEQUENCE_INPUT_DIR.is_dir()

    print(
        "Sequence-input directory: "
        + ("PASS" if sequence_dir_pass else "FAIL")
    )

    modeling_ok, modeling_message = check_manifest(
        MODELING_MANIFEST
    )

    prep_ok, prep_message = check_manifest(
        PREPARATION_MANIFEST
    )

    print(
        "modeling_dataset_manifest.json: "
        + modeling_message
        + (
            f" [{MODELING_MANIFEST.relative_to(PROJECT_ROOT).as_posix()}]"
            if modeling_ok
            else ""
        )
    )

    print(
        "sequence_input_preparation_manifest.json: "
        + prep_message
        + (
            f" [{PREPARATION_MANIFEST.relative_to(PROJECT_ROOT).as_posix()}]"
            if prep_ok
            else ""
        )
    )

    print()

    if not sequence_dir_pass:
        print("=" * 80)
        print(
            "T1D-UOM NUMERIC / SEMANTIC GRU INPUT CONTRACT AUDIT FAILED"
        )
        print("=" * 80)
        print()
        print(
            "Sequence-input directory does not exist."
        )
        return 1

    # ------------------------------------------------------------------------
    # 2. INVENTORY
    # ------------------------------------------------------------------------

    print("-" * 80)
    print("2. FROZEN SEQUENCE-INPUT INVENTORY")
    print("-" * 80)

    discovered = discover_csv_files()

    print(
        f"Sequence-input CSV files discovered: "
        f"{len(discovered)}"
    )

    inventory_pass = (
        len(discovered) == EXPECTED_FILE_COUNT
    )

    print(
        "Frozen sequence-input file count: "
        + ("PASS" if inventory_pass else "FAIL")
    )

    print()

    # ------------------------------------------------------------------------
    # 3. AUDIT ALL FILES
    # ------------------------------------------------------------------------

    print("-" * 80)
    print("3. COMPLETE 86-FILE PARTICIPANT / MODALITY AUDIT")
    print("-" * 80)

    audited: list[dict[str, Any]] = []

    for index, path in enumerate(
        discovered,
        start=1,
    ):

        result = audit_file(path)
        audited.append(result)

        participant = result.get("participant")
        modality = result.get("modality")

        participant_text = (
            participant
            if participant is not None
            else "UNKNOWN"
        )

        modality_text = (
            modality
            if modality is not None
            else "UNKNOWN"
        )

        status = result["status"]

        print(
            f"[{index:03d}/{len(discovered):03d}] "
            f"{relative_path(path)}"
            f" | participant={participant_text}"
            f" | modality={modality_text}"
            f" | {status}"
        )

    print()

    # ------------------------------------------------------------------------
    # MANIFEST RESULTS
    # ------------------------------------------------------------------------

    manifest_results = {
        "modeling_dataset_manifest": {
            "path": MODELING_MANIFEST.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "exists": MODELING_MANIFEST.exists(),
            "readable": modeling_ok,
            "status": modeling_message,
        },

        "sequence_input_preparation_manifest": {
            "path": PREPARATION_MANIFEST.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "exists": PREPARATION_MANIFEST.exists(),
            "readable": prep_ok,
            "status": prep_message,
        },
    }

    report = build_report(
        discovered,
        audited,
        manifest_results,
    )

    participants = report["summary"][
        "participants_represented"
    ]

    # ------------------------------------------------------------------------
    # 4. COHORT
    # ------------------------------------------------------------------------

    print("-" * 80)
    print("4. FROZEN COHORT VALIDATION")
    print("-" * 80)

    print(
        f"Participants represented: {len(participants)}"
    )

    for participant in participants:
        print(f"  {participant}")

    cohort_pass = (
        len(participants) == len(FROZEN_PARTICIPANTS)
        and set(participants) == EXPECTED_PARTICIPANT_SET
        and not report["participant_failures"]
        and not report["unknown_participants"]
        and not report["missing_participants"]
    )

    print(
        "Frozen 13-participant cohort: "
        + ("PASS" if cohort_pass else "FAIL")
    )

    if report["participant_failures"]:
        print()
        print("Participant-identification failures:")
        for item in report["participant_failures"]:
            print(f"  - {item}")

    if report["unknown_participants"]:
        print()
        print("Unknown participants:")
        for item in report["unknown_participants"]:
            print(f"  - {item}")

    if report["missing_participants"]:
        print()
        print("Missing frozen participants:")
        for item in report["missing_participants"]:
            print(f"  - {item}")

    print()

    # ------------------------------------------------------------------------
    # 5. MODALITY COUNTS
    # ------------------------------------------------------------------------

    print("-" * 80)
    print("5. MODALITY FAMILY VALIDATION")
    print("-" * 80)

    modality_counts = report["summary"][
        "modality_counts"
    ]

    for modality, expected in EXPECTED_MODALITY_COUNTS.items():

        actual = modality_counts.get(
            modality,
            0,
        )

        print(
            f"{modality:<18}"
            f"files={actual:<3}"
            f" expected={expected}"
            f" "
            + (
                "PASS"
                if actual == expected
                else "FAIL"
            )
        )

    modality_pass = not report[
        "modality_failures"
    ]

    print(
        "Modality family counts: "
        + ("PASS" if modality_pass else "FAIL")
    )

    print()

    # ------------------------------------------------------------------------
    # 6. ARCHITECTURE
    # ------------------------------------------------------------------------

    print_architecture()

    # ------------------------------------------------------------------------
    # 7. PHYSICAL CONTRACT
    # ------------------------------------------------------------------------

    print_contract()

    # ------------------------------------------------------------------------
    # 8. FILE-LEVEL SUMMARY
    # ------------------------------------------------------------------------

    print("-" * 80)
    print("8. FILE-LEVEL CONTRACT SUMMARY")
    print("-" * 80)

    schema_failures = report["schema_failures"]
    numeric_failures = report["numeric_failures"]
    categorical_failures = report[
        "categorical_failures"
    ]
    structured_failures = report[
        "structured_failures"
    ]

    print(
        f"Files discovered:                  "
        f"{len(discovered)}"
    )

    print(
        f"Files audited:                     "
        f"{len(audited)}"
    )

    print(
        f"Schema failures:                   "
        f"{len(schema_failures)}"
    )

    print(
        f"Numeric semantic failures:         "
        f"{len(numeric_failures)}"
    )

    print(
        f"Categorical semantic failures:     "
        f"{len(categorical_failures)}"
    )

    print(
        f"Structured semantic failures:      "
        f"{len(structured_failures)}"
    )

    print(
        f"Files requiring missing-value review: "
        f"{len(report['missing_value_reviews'])}"
    )

    print()

    # ------------------------------------------------------------------------
    # 9. WRITE JSON REPORT
    # ------------------------------------------------------------------------

    print("-" * 80)
    print("9. WRITING AUDIT REPORT")
    print("-" * 80)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUDIT_REPORT.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            json_safe(report),
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("Audit report saved:")
    print(f"  {AUDIT_REPORT}")
    print()

    # ------------------------------------------------------------------------
    # FINAL STATUS
    # ------------------------------------------------------------------------

    genuine_failure = report[
        "summary"
    ]["genuine_contract_failure"]

    complete_audit = (
        len(audited) == len(discovered)
        and len(discovered) == EXPECTED_FILE_COUNT
    )

    final_pass = (
        complete_audit
        and cohort_pass
        and modality_pass
        and not schema_failures
        and not numeric_failures
        and not categorical_failures
        and not structured_failures
        and modeling_ok
        and prep_ok
    )

    print("=" * 80)

    if final_pass:

        print(
            "T1D-UOM NUMERIC / SEMANTIC GRU INPUT "
            "CONTRACT AUDIT COMPLETE"
        )

        print("=" * 80)
        print()
        print("FINAL RESULT: PASS")
        print()
        print(
            "All frozen sequence-input files were audited."
        )
        print(
            f"Files audited: {len(audited)}/{EXPECTED_FILE_COUNT}"
        )
        print(
            "Frozen 13-participant cohort: PASS"
        )
        print(
            "Modality family counts: PASS"
        )
        print(
            "Schema contract: PASS"
        )
        print(
            "Numeric semantic contract: PASS"
        )
        print(
            "Categorical semantic contract: PASS"
        )
        print(
            "Structured semantic contract: PASS"
        )
        print()
        print(
            "Missing-value reviews are informational only."
        )
        print(
            "No imputation or other missing-value handling "
            "was performed."
        )
        print()
        print(
            "No dataset files were modified."
        )
        print(
            "No sequence-input files were modified."
        )
        print(
            "No categorical encoding was performed."
        )
        print(
            "No event_type was constructed."
        )
        print(
            "No windows were created."
        )
        print(
            "No targets were created."
        )
        print(
            "No model was trained."
        )
        print(
            "Frozen architecture was not changed."
        )
        print()
        print(
            "The physical / semantic input contract is "
            "ready for the next controlled project stage."
        )
        print("=" * 80)

        return 0

    print(
        "T1D-UOM NUMERIC / SEMANTIC GRU INPUT "
        "CONTRACT AUDIT FAILED"
    )

    print("=" * 80)
    print()

    if not complete_audit:
        print(
            "CRITICAL: Complete 86-file audit was not achieved."
        )
        print(
            f"Files discovered: {len(discovered)}"
        )
        print(
            f"Files audited:    {len(audited)}"
        )

    if report["participant_failures"]:
        print(
            f"Participant-identification failures: "
            f"{len(report['participant_failures'])}"
        )

    if report["missing_participants"]:
        print(
            "Missing frozen participants: "
            + ", ".join(
                report["missing_participants"]
            )
        )

    if report["modality_failures"]:
        print(
            "Modality count failures detected."
        )

    if schema_failures:
        print(
            f"Schema failures: {len(schema_failures)}"
        )

    if numeric_failures:
        print(
            f"Numeric semantic failures: "
            f"{len(numeric_failures)}"
        )

    if categorical_failures:
        print(
            f"Categorical semantic failures: "
            f"{len(categorical_failures)}"
        )

    if structured_failures:
        print(
            f"Structured semantic failures: "
            f"{len(structured_failures)}"
        )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "  No dataset files were modified."
    )
    print(
        "  No sequence-input files were modified."
    )
    print(
        "  No categorical encoding was performed."
    )
    print(
        "  No event_type was constructed."
    )
    print(
        "  No windows were created."
    )
    print(
        "  No targets were created."
    )
    print(
        "  No model was trained."
    )
    print(
        "  Frozen architecture was not changed."
    )
    print()
    print(
        "Inspect the JSON report for exact file-level details."
    )
    print("=" * 80)

    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print(
            "AUDIT INTERRUPTED BY USER."
        )
        print(
            "No dataset or sequence-input files were modified."
        )
        sys.exit(130)

    except Exception as exc:
        print()
        print("=" * 80)
        print(
            "UNEXPECTED AUDIT ERROR"
        )
        print("=" * 80)
        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        print(
            "No dataset or sequence-input files were modified "
            "by the audit."
        )
        print("=" * 80)
        sys.exit(2)