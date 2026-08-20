"""
T1D-UOM FIVE-GRU RUNTIME INPUT CONTRACT VALIDATION

READ-ONLY runtime-input contract gate for the frozen 86-file sequence-input set.

This validator intentionally does NOT:
- modify raw dataset files
- modify sequence-input CSV files
- add/remove/reorder rows
- impute missing values
- normalize numeric values
- encode categorical values
- construct event_type
- create windows or targets
- train a model
- implement MLP Fusion, Digital Twin, Prediction/What-if, or UI

Important frozen filename rules:
- UoMActivity2301.csv      -> UoM2301 / activity
- UoMGlucose2301.csv       -> UoM2301 / glucose
- UoMBolus2301.csv         -> UoM2301 / bolus_insulin
- UoMBasal2302.csv         -> UoM2302 / basal_insulin
- UoMNutrition2301.csv     -> UoM2301 / nutrition
- UoM2302sleeptime.csv     -> UoM2302 / sleep_summary
- UoMsleep2302.csv         -> UoM2302 / sleep_timeseries

Timestamp policy is aligned with the established sequence-input preparation:
- day-first: DD/MM/YYYY [HH:MM[:SS]]
- standard: YYYY-MM-DD [HH:MM[:SS]] or ISO T form

The validator writes ONLY its JSON report. All source CSVs are read-only.
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
REPORT_PATH = REPORT_DIR / "five_gru_runtime_input_contract_validation.json"

MODELING_MANIFEST = REPORT_DIR / "modeling_dataset_manifest.json"
PREPARATION_MANIFEST = REPORT_DIR / "sequence_input_preparation_manifest.json"

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
# FROZEN FIVE-GRU CONTRACT
# ============================================================================

ARCHITECTURE = {
    "glucose": "GRU -> zG",
    "insulin": "GRU -> zI",
    "nutrition": "GRU -> zN",
    "activity": "GRU -> zA",
    "sleep": "GRU -> zS",
}

MODALITY_CONTRACTS: dict[str, dict[str, Any]] = {
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
        "numeric": ["carbs_g", "prot_g", "fat_g", "fibre_g"],
        "categorical": ["meal_type", "meal_tag"],
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
        "categorical": ["activity_type", "intensity"],
        "structured": [],
        "expected_dim": 12,
    },
    "basal_insulin": {
        "folder": "Insulin Data/Basal Data",
        "timestamp": "basal_ts",
        "numeric": ["basal_dose"],
        "categorical": ["insulin_kind"],
        "structured": [],
        "expected_dim": 2,
    },
    "bolus_insulin": {
        "folder": "Insulin Data/Bolus Data",
        "timestamp": "bolus_ts",
        "numeric": ["bolus_dose"],
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
        "categorical": ["calendar_date", "validation"],
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


# ============================================================================
# HELPERS
# ============================================================================

EMPTY_STRINGS = {"", "nan", "none", "null", "nat"}
MAX_EXAMPLES_PER_ERROR = 10
MIN_REASONABLE_YEAR = 2020
MAX_REASONABLE_YEAR = 2026


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(SEQUENCE_INPUT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return str(value)


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip().lower() in EMPTY_STRINGS


def add_limited(mapping: dict[str, list[Any]], key: str, value: Any) -> None:
    bucket = mapping.setdefault(key, [])
    if len(bucket) < MAX_EXAMPLES_PER_ERROR:
        bucket.append(value)


# ============================================================================
# PARTICIPANT IDENTIFICATION
# ============================================================================

PARTICIPANT_PATTERN = re.compile(
    r"^UoM(?:Activity|Glucose|Bolus|Basal|Nutrition|sleep)?(2[34]\d{2})(?:.*)?$",
    re.IGNORECASE,
)


def identify_participant(path: Path) -> str | None:
    """Identify the canonical participant from the actual frozen filename."""
    stem = path.stem.strip()
    match = PARTICIPANT_PATTERN.match(stem)
    if match:
        return f"UoM{match.group(1)}"

    # Conservative fallback for known naming families.
    fallback = re.search(r"UoM(2[34]\d{2})", stem, re.IGNORECASE)
    if fallback:
        return f"UoM{fallback.group(1)}"
    return None


# ============================================================================
# MODALITY IDENTIFICATION
# ============================================================================


def identify_modality(path: Path) -> str | None:
    rel = relative_path(path).lower()
    name = path.name.lower()

    if rel.startswith("activity data/"):
        return "activity"
    if rel.startswith("glucose data/"):
        return "glucose"
    if rel.startswith("nutrition data/"):
        return "nutrition"
    if rel.startswith("insulin data/basal data/"):
        return "basal_insulin"
    if rel.startswith("insulin data/bolus data/"):
        return "bolus_insulin"

    # CRITICAL FROZEN RULE:
    # sleeptime files are sleep_summary;
    # UoMsleepXXXX files are sleep_timeseries.
    if rel.startswith("sleep data/"):
        if "sleeptime" in name:
            return "sleep_summary"
        if name.startswith("uomsleep"):
            return "sleep_timeseries"

    return None


# ============================================================================
# TIMESTAMP PARSING
# ============================================================================

DAYFIRST_DATETIME = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?$"
)
DAYFIRST_DATE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
STANDARD_DATETIME = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?$"
)
STANDARD_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_T = re.compile(r"^\d{4}-\d{2}-\d{2}T")


def parse_timestamp_value(value: Any) -> tuple[pd.Timestamp | None, str]:
    """Parse one timestamp without mutating the source value."""
    if is_empty(value):
        return None, "missing"

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None, "missing"
        return value.tz_localize(None) if value.tzinfo else value, "pandas_timestamp"

    text = str(value).strip()

    try:
        if DAYFIRST_DATE.fullmatch(text):
            parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
            return (None, "invalid") if pd.isna(parsed) else (parsed, "dayfirst_date")

        if DAYFIRST_DATETIME.fullmatch(text):
            # Try with seconds first, then minute precision.
            for fmt in ("%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
                parsed = pd.to_datetime(text, format=fmt, errors="coerce")
                if not pd.isna(parsed):
                    return parsed, "dayfirst_datetime"
            return None, "invalid"

        if STANDARD_DATE.fullmatch(text):
            parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
            return (None, "invalid") if pd.isna(parsed) else (parsed, "standard_date")

        if STANDARD_DATETIME.fullmatch(text):
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                parsed = pd.to_datetime(text, format=fmt, errors="coerce")
                if not pd.isna(parsed):
                    return parsed, "standard_datetime"
            return None, "invalid"

        if ISO_T.match(text):
            parsed = pd.to_datetime(text, errors="coerce", format="mixed", utc=True)
            if pd.isna(parsed):
                return None, "invalid"
            return parsed.tz_convert(None), "iso"

    except Exception:
        return None, "invalid"

    return None, "invalid"


def audit_timestamp(series: pd.Series) -> dict[str, Any]:
    missing_count = 0
    parseable_count = 0
    invalid_count = 0
    strategy_counts: dict[str, int] = {}
    invalid_examples: list[dict[str, Any]] = []
    years: list[int] = []

    for row_index, value in series.items():
        parsed, strategy = parse_timestamp_value(value)
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        if strategy == "missing":
            missing_count += 1
            continue

        if parsed is None:
            invalid_count += 1
            if len(invalid_examples) < MAX_EXAMPLES_PER_ERROR:
                invalid_examples.append({
                    "row_index": int(row_index),
                    "value": str(value),
                })
            continue

        parseable_count += 1
        try:
            years.append(int(parsed.year))
        except Exception:
            pass

    suspicious_year_count = sum(
        1 for year in years
        if year < MIN_REASONABLE_YEAR or year > MAX_REASONABLE_YEAR
    )

    return {
        "row_count": int(len(series)),
        "missing_count": missing_count,
        "parseable_count": parseable_count,
        "unparseable_non_missing_count": invalid_count,
        "strategy_counts": strategy_counts,
        "invalid_examples": invalid_examples,
        "suspicious_year_count": suspicious_year_count,
        "min_year": min(years) if years else None,
        "max_year": max(years) if years else None,
    }


# ============================================================================
# VALUE-LEVEL AUDITS
# ============================================================================


def audit_numeric_columns(df: pd.DataFrame, columns: list[str]) -> tuple[dict[str, int], dict[str, int], dict[str, list[Any]]]:
    missing: dict[str, int] = {}
    invalid: dict[str, int] = {}
    invalid_examples: dict[str, list[Any]] = {}

    for column in columns:
        series = df[column]
        empty_mask = series.map(is_empty)
        missing_count = int(empty_mask.sum())
        if missing_count:
            missing[column] = missing_count

        nonempty = series.loc[~empty_mask]
        if nonempty.empty:
            continue

        converted = pd.to_numeric(nonempty, errors="coerce")
        invalid_mask = converted.isna() | ~converted.apply(lambda x: math.isfinite(float(x)))
        invalid_count = int(invalid_mask.sum())

        if invalid_count:
            invalid[column] = invalid_count
            examples = nonempty.loc[invalid_mask].head(MAX_EXAMPLES_PER_ERROR).tolist()
            invalid_examples[column] = [str(x) for x in examples]

    return missing, invalid, invalid_examples


def audit_categorical_columns(df: pd.DataFrame, columns: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    missing: dict[str, int] = {}
    cardinality: dict[str, int] = {}

    for column in columns:
        series = df[column]
        missing_count = int(series.map(is_empty).sum())
        if missing_count:
            missing[column] = missing_count

        nonempty = series.loc[~series.map(is_empty)]
        cardinality[column] = int(nonempty.astype(str).nunique(dropna=True))

    return missing, cardinality


def audit_structured_columns(df: pd.DataFrame, columns: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    """
    Structured sleep maps remain raw strings at this stage.
    We do not transform or require full JSON parsing.
    We only flag empty values as REVIEW and obvious null-like non-empty tokens
    as failures, matching the established physical/semantic audit policy.
    """
    empty_values: dict[str, int] = {}
    invalid_values: dict[str, int] = {}

    for column in columns:
        series = df[column]
        empty_mask = series.map(is_empty)
        empty_count = int(empty_mask.sum())
        if empty_count:
            empty_values[column] = empty_count

        nonempty = series.loc[~empty_mask]
        invalid_count = 0
        for value in nonempty:
            text = str(value).strip().lower()
            if text in {"nan", "none", "null"}:
                invalid_count += 1
        if invalid_count:
            invalid_values[column] = invalid_count

    return empty_values, invalid_values


# ============================================================================
# FILE AUDIT
# ============================================================================


def audit_file(path: Path) -> dict[str, Any]:
    participant = identify_participant(path)
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
        "timestamp_status": "PASS",
        "timestamp_column": None,
        "timestamp_missing_count": 0,
        "timestamp_parseable_count": 0,
        "timestamp_unparseable_non_missing_count": 0,
        "timestamp_strategy_counts": {},
        "timestamp_invalid_examples": [],
        "timestamp_suspicious_year_count": 0,
        "expected_numeric_columns": [],
        "expected_categorical_columns": [],
        "expected_structured_columns": [],
        "actual_columns": [],
        "missing_columns": [],
        "unexpected_columns": [],
        "missing_numeric_values": {},
        "invalid_numeric_values": {},
        "invalid_numeric_examples": {},
        "categorical_missing_values": {},
        "categorical_cardinality": {},
        "structured_empty_values": {},
        "invalid_structured_values": {},
        "row_count": None,
        "read_error": None,
        "errors": [],
        "warnings": [],
    }

    if participant is None:
        result["schema_status"] = "FAIL"
        result["status"] = "FAIL"
        result["errors"].append("Participant could not be identified from filename.")

    elif participant not in EXPECTED_PARTICIPANT_SET:
        result["schema_status"] = "FAIL"
        result["status"] = "FAIL"
        result["errors"].append(
            f"Participant '{participant}' is outside the frozen cohort."
        )

    if modality is None:
        result["schema_status"] = "FAIL"
        result["status"] = "FAIL"
        result["errors"].append("Frozen modality could not be identified from path/filename.")
        return result

    contract = MODALITY_CONTRACTS[modality]
    result["timestamp_column"] = contract["timestamp"]
    result["expected_numeric_columns"] = list(contract["numeric"])
    result["expected_categorical_columns"] = list(contract["categorical"])
    result["expected_structured_columns"] = list(contract["structured"])

    try:
        # utf-8-sig safely handles the existing UTF-8/UTF-8-BOM CSVs without
        # writing anything back to the source.
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        except Exception as exc:
            result["schema_status"] = "FAIL"
            result["status"] = "FAIL"
            result["read_error"] = repr(exc)
            return result
    except Exception as exc:
        result["schema_status"] = "FAIL"
        result["status"] = "FAIL"
        result["read_error"] = repr(exc)
        return result

    result["actual_columns"] = [str(c) for c in df.columns]
    result["row_count"] = int(len(df))

    expected_columns = [
        contract["timestamp"],
        *contract["numeric"],
        *contract["categorical"],
        *contract["structured"],
    ]
    expected_set = set(expected_columns)
    actual_set = set(df.columns)

    missing_columns = [c for c in expected_columns if c not in actual_set]
    unexpected_columns = [str(c) for c in df.columns if c not in expected_set]

    result["missing_columns"] = missing_columns
    result["unexpected_columns"] = unexpected_columns

    if missing_columns:
        result["schema_status"] = "FAIL"
        result["status"] = "FAIL"
        result["errors"].append(
            "Required contract columns are missing: " + ", ".join(missing_columns)
        )
        # No value-level validation against missing fields.
        return result

    if unexpected_columns:
        result["schema_status"] = "FAIL"
        result["status"] = "FAIL"
        result["errors"].append(
            "Unexpected columns are present: " + ", ".join(unexpected_columns)
        )

    # Timestamp validation.
    timestamp_result = audit_timestamp(df[contract["timestamp"]])
    result["timestamp_missing_count"] = timestamp_result["missing_count"]
    result["timestamp_parseable_count"] = timestamp_result["parseable_count"]
    result["timestamp_unparseable_non_missing_count"] = timestamp_result["unparseable_non_missing_count"]
    result["timestamp_strategy_counts"] = timestamp_result["strategy_counts"]
    result["timestamp_invalid_examples"] = timestamp_result["invalid_examples"]
    result["timestamp_suspicious_year_count"] = timestamp_result["suspicious_year_count"]

    if timestamp_result["unparseable_non_missing_count"]:
        result["timestamp_status"] = "FAIL"
        result["schema_status"] = "FAIL"
        result["status"] = "FAIL"
        result["errors"].append(
            f"Timestamp column '{contract['timestamp']}' contains "
            f"{timestamp_result['unparseable_non_missing_count']} unparseable non-missing value(s)."
        )

    if timestamp_result["missing_count"]:
        # Missing timestamps are a review-worthy runtime issue because the
        # sequence order cannot be determined for those rows. The existing
        # frozen 86-file set is expected to have zero timestamp missingness.
        result["timestamp_status"] = "FAIL"
        result["schema_status"] = "FAIL"
        result["status"] = "FAIL"
        result["errors"].append(
            f"Required timestamp column '{contract['timestamp']}' contains "
            f"{timestamp_result['missing_count']} missing value(s)."
        )

    if timestamp_result["suspicious_year_count"]:
        result["warnings"].append(
            f"{timestamp_result['suspicious_year_count']} timestamp value(s) "
            f"fall outside the broad {MIN_REASONABLE_YEAR}-{MAX_REASONABLE_YEAR} year range."
        )

    # Numeric values.
    missing_numeric, invalid_numeric, invalid_numeric_examples = audit_numeric_columns(
        df, contract["numeric"]
    )
    result["missing_numeric_values"] = missing_numeric
    result["invalid_numeric_values"] = invalid_numeric
    result["invalid_numeric_examples"] = invalid_numeric_examples

    if invalid_numeric:
        result["numeric_status"] = "FAIL"
        result["status"] = "FAIL"
        result["errors"].append(
            "One or more numeric contract columns contain non-numeric/non-finite values."
        )

    # Categorical values: missingness is REVIEW only; no encoding occurs.
    categorical_missing, cardinality = audit_categorical_columns(
        df, contract["categorical"]
    )
    result["categorical_missing_values"] = categorical_missing
    result["categorical_cardinality"] = cardinality

    # Structured values: empty is REVIEW only; obvious null-like tokens fail.
    structured_empty, structured_invalid = audit_structured_columns(
        df, contract["structured"]
    )
    result["structured_empty_values"] = structured_empty
    result["invalid_structured_values"] = structured_invalid

    if structured_invalid:
        result["structured_status"] = "FAIL"
        result["status"] = "FAIL"
        result["errors"].append(
            "One or more structured contract columns contain obvious null-like values."
        )

    # Frozen feature dimension is a contract-level invariant.
    calculated_dim = (
        len(contract["numeric"])
        + len(contract["categorical"])
        + len(contract["structured"])
    )
    if calculated_dim != contract["expected_dim"]:
        result["schema_status"] = "FAIL"
        result["status"] = "FAIL"
        result["dimension_error"] = {
            "expected": contract["expected_dim"],
            "calculated": calculated_dim,
        }
        result["errors"].append(
            f"Frozen feature dimension mismatch: expected {contract['expected_dim']}, "
            f"calculated {calculated_dim}."
        )

    return result


# ============================================================================
# INVENTORY / MANIFESTS
# ============================================================================


def discover_csv_files() -> list[Path]:
    if not SEQUENCE_INPUT_DIR.is_dir():
        return []
    return sorted(
        [p for p in SEQUENCE_INPUT_DIR.rglob("*.csv") if p.is_file()],
        key=lambda p: relative_path(p).lower(),
    )


def check_json_manifest(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "NOT FOUND"
    if not path.is_file():
        return False, "NOT A FILE"
    try:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
        return True, "PASS"
    except Exception as exc:
        return False, f"UNREADABLE: {exc}"


# ============================================================================
# REPORT
# ============================================================================


def build_report(
    discovered: list[Path],
    audited: list[dict[str, Any]],
    modeling_manifest_status: tuple[bool, str],
    preparation_manifest_status: tuple[bool, str],
) -> dict[str, Any]:
    participants = sorted(
        {
            item["participant"]
            for item in audited
            if item.get("participant") in EXPECTED_PARTICIPANT_SET
        }
    )

    unknown_participants = sorted(
        {
            item["participant"]
            for item in audited
            if item.get("participant")
            and item.get("participant") not in EXPECTED_PARTICIPANT_SET
        }
    )

    participant_failures = sorted(
        item["relative_path"]
        for item in audited
        if item.get("participant") is None
    )

    missing_participants = sorted(EXPECTED_PARTICIPANT_SET - set(participants))

    modality_counts = {modality: 0 for modality in EXPECTED_MODALITY_COUNTS}
    for item in audited:
        modality = item.get("modality")
        if modality in modality_counts:
            modality_counts[modality] += 1

    modality_failures = {
        modality: {
            "observed": modality_counts.get(modality, 0),
            "expected": expected,
        }
        for modality, expected in EXPECTED_MODALITY_COUNTS.items()
        if modality_counts.get(modality, 0) != expected
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
    timestamp_failures = [
        item["relative_path"]
        for item in audited
        if item["timestamp_status"] == "FAIL"
    ]

    missing_value_reviews = []
    for item in audited:
        if (
            item["missing_numeric_values"]
            or item["categorical_missing_values"]
            or item["structured_empty_values"]
        ):
            missing_value_reviews.append({
                "relative_path": item["relative_path"],
                "missing_numeric_values": item["missing_numeric_values"],
                "categorical_missing_values": item["categorical_missing_values"],
                "structured_empty_values": item["structured_empty_values"],
            })

    timestamp_strategy_counts: dict[str, int] = {}
    for item in audited:
        for strategy, count in item.get("timestamp_strategy_counts", {}).items():
            timestamp_strategy_counts[strategy] = timestamp_strategy_counts.get(strategy, 0) + count

    genuine_contract_failure = any(
        [
            len(discovered) != EXPECTED_FILE_COUNT,
            len(audited) != len(discovered),
            bool(participant_failures),
            bool(unknown_participants),
            bool(missing_participants),
            bool(modality_failures),
            bool(schema_failures),
            bool(numeric_failures),
            bool(categorical_failures),
            bool(structured_failures),
            bool(timestamp_failures),
            not modeling_manifest_status[0],
            not preparation_manifest_status[0],
        ]
    )

    return {
        "audit_name": "T1D-UOM FIVE-GRU RUNTIME INPUT CONTRACT VALIDATION",
        "audit_version": "2.0.0",
        "read_only": True,
        "source_files_modified": False,
        "sequence_input_files_modified": False,
        "windows_created": False,
        "targets_created": False,
        "imputation_performed": False,
        "normalization_performed": False,
        "categorical_encoding_written_to_source": False,
        "model_training_performed": False,
        "project_root": str(PROJECT_ROOT),
        "sequence_input_directory": str(SEQUENCE_INPUT_DIR),
        "report_path": str(REPORT_PATH),
        "frozen_architecture": {
            "branches": ARCHITECTURE,
            "branch_count": 5,
            "mlp_fusion_implemented": False,
            "digital_twin_implemented": False,
            "prediction_implemented": False,
            "what_if_implemented": False,
            "interactive_ui_implemented": False,
        },
        "physical_contract": MODALITY_CONTRACTS,
        "runtime_representation_policy": {
            "numeric_features": "remain numeric; this validator does not transform them",
            "categorical_features": "remain raw strings; this validator does not encode them",
            "categorical_vocabulary": "training-only fitting in the controlled representation stage",
            "unknown_category_policy": "UNK",
            "missing_category_policy": "MISSING",
            "padding_policy": "PAD",
            "embedding_dimensions": "NOT FROZEN BY THIS AUDIT",
        },
        "inventory": {
            "files_discovered": len(discovered),
            "files_audited": len(audited),
            "expected_files": EXPECTED_FILE_COUNT,
            "file_count_status": "PASS" if len(discovered) == EXPECTED_FILE_COUNT else "FAIL",
        },
        "cohort": {
            "participants": participants,
            "participant_count": len(participants),
            "expected_participant_count": len(FROZEN_PARTICIPANTS),
            "status": "PASS"
            if set(participants) == EXPECTED_PARTICIPANT_SET and not participant_failures and not unknown_participants
            else "FAIL",
            "missing_participants": missing_participants,
            "unexpected_participants": unknown_participants,
        },
        "modality_counts": {
            modality: {
                "observed": modality_counts.get(modality, 0),
                "expected": expected,
                "status": "PASS" if modality_counts.get(modality, 0) == expected else "FAIL",
            }
            for modality, expected in EXPECTED_MODALITY_COUNTS.items()
        },
        "timestamp_policy": {
            "accepted_dayfirst": [
                "%d/%m/%Y",
                "%d/%m/%Y %H:%M",
                "%d/%m/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M:%S.%f",
            ],
            "accepted_standard": [
                "%Y-%m-%d",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "ISO-8601 T form",
            ],
            "strategy_counts": timestamp_strategy_counts,
            "timestamp_failures": timestamp_failures,
        },
        "summary": {
            "schema_failures": len(schema_failures),
            "numeric_failures": len(numeric_failures),
            "categorical_failures": len(categorical_failures),
            "structured_failures": len(structured_failures),
            "timestamp_failures": len(timestamp_failures),
            "files_requiring_missing_value_review": len(missing_value_reviews),
            "overall_status": "PASS" if not genuine_contract_failure else "FAIL",
        },
        "failure_lists": {
            "participant_failures": participant_failures,
            "unknown_participants": unknown_participants,
            "missing_participants": missing_participants,
            "modality_failures": modality_failures,
            "schema_failures": schema_failures,
            "numeric_failures": numeric_failures,
            "categorical_failures": categorical_failures,
            "structured_failures": structured_failures,
            "timestamp_failures": timestamp_failures,
        },
        "missing_value_reviews": missing_value_reviews,
        "manifests": {
            "modeling_dataset_manifest": {
                "path": relative_project_path(MODELING_MANIFEST),
                "exists": MODELING_MANIFEST.exists(),
                "readable": modeling_manifest_status[0],
                "status": modeling_manifest_status[1],
            },
            "sequence_input_preparation_manifest": {
                "path": relative_project_path(PREPARATION_MANIFEST),
                "exists": PREPARATION_MANIFEST.exists(),
                "readable": preparation_manifest_status[0],
                "status": preparation_manifest_status[1],
            },
        },
        "files": audited,
    }


def relative_project_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


# ============================================================================
# MAIN
# ============================================================================


def print_contract() -> None:
    print("-" * 80)
    print("5A. FROZEN PHYSICAL / SEMANTIC CONTRACT")
    print("-" * 80)
    for modality, contract in MODALITY_CONTRACTS.items():
        print(f"\n{modality.upper()}")
        print(f"  Timestamp:          {contract['timestamp']}")
        print(f"  Expected feature dim: {contract['expected_dim']}")
        if contract["numeric"]:
            print("  Numeric:")
            for col in contract["numeric"]:
                print(f"    - {col}")
        if contract["categorical"]:
            print("  Categorical/raw:")
            for col in contract["categorical"]:
                print(f"    - {col}")
        if contract["structured"]:
            print("  Structured/raw:")
            for col in contract["structured"]:
                print(f"    - {col}")
    print()


def main() -> int:
    print("=" * 80)
    print("T1D-UOM FIVE-GRU RUNTIME INPUT CONTRACT VALIDATION")
    print("=" * 80)
    print("IMPORTANT: READ-ONLY.")
    print("No dataset files will be modified.")
    print("No sequence-input files will be modified.")
    print("No values will be transformed.")
    print("No rows will be deleted.")
    print("No imputation will be performed.")
    print("No normalization will be performed.")
    print("No categorical encoding will be performed.")
    print("No event_type will be constructed.")
    print("No windows or targets will be created.")
    print("No model will be trained.")
    print("No MLP Fusion / Digital Twin / Prediction / UI will be implemented.")
    print()
    print(f"Project root:       {PROJECT_ROOT}")
    print(f"Sequence inputs:    {SEQUENCE_INPUT_DIR}")
    print(f"Audit report:       {REPORT_PATH}")
    print()

    print("-" * 80)
    print("1. DIRECTORY AND MANIFEST VALIDATION")
    print("-" * 80)
    directory_pass = SEQUENCE_INPUT_DIR.is_dir()
    print("Sequence-input directory: " + ("PASS" if directory_pass else "FAIL"))

    modeling_status = check_json_manifest(MODELING_MANIFEST)
    preparation_status = check_json_manifest(PREPARATION_MANIFEST)
    print("modeling_dataset_manifest.json: " + modeling_status[1])
    print("sequence_input_preparation_manifest.json: " + preparation_status[1])
    print()

    if not directory_pass:
        print("=" * 80)
        print("FINAL RESULT: FAIL")
        print("Sequence-input directory does not exist.")
        print("=" * 80)
        return 1

    print("-" * 80)
    print("2. FROZEN SEQUENCE-INPUT INVENTORY")
    print("-" * 80)
    discovered = discover_csv_files()
    print(f"Sequence-input CSV files discovered: {len(discovered)}")
    inventory_pass = len(discovered) == EXPECTED_FILE_COUNT
    print("Frozen sequence-input file count: " + ("PASS" if inventory_pass else "FAIL"))
    print()

    print("-" * 80)
    print("3. COMPLETE 86-FILE RUNTIME CONTRACT AUDIT")
    print("-" * 80)
    audited: list[dict[str, Any]] = []

    for index, path in enumerate(discovered, start=1):
        result = audit_file(path)
        audited.append(result)
        status = result["status"]
        print(
            f"[{index:03d}/{len(discovered):03d}] "
            f"{relative_path(path)} "
            f"| participant={result.get('participant')} "
            f"| modality={result.get('modality')} "
            f"| {status}"
        )
        for error in result["errors"]:
            print(f"    error: {error}")
        for warning in result["warnings"]:
            print(f"    warning: {warning}")

    print()
    print("-" * 80)
    print("4. FROZEN COHORT VALIDATION")
    print("-" * 80)
    participants = sorted(
        {x["participant"] for x in audited if x.get("participant") in EXPECTED_PARTICIPANT_SET}
    )
    print(f"Participants represented: {len(participants)}")
    for participant in participants:
        print(f"  {participant}")
    print(
        "Frozen 13-participant cohort: "
        + (
            "PASS"
            if set(participants) == EXPECTED_PARTICIPANT_SET
            and all(x.get("participant") is not None for x in audited)
            else "FAIL"
        )
    )
    print()

    print("-" * 80)
    print("5. MODALITY FAMILY VALIDATION")
    print("-" * 80)
    modality_counts = {m: 0 for m in EXPECTED_MODALITY_COUNTS}
    for item in audited:
        if item.get("modality") in modality_counts:
            modality_counts[item["modality"]] += 1
    for modality, expected in EXPECTED_MODALITY_COUNTS.items():
        actual = modality_counts[modality]
        print(f"{modality:<18} files={actual:<3} expected={expected:<3} " + ("PASS" if actual == expected else "FAIL"))
    modality_pass = all(modality_counts[m] == expected for m, expected in EXPECTED_MODALITY_COUNTS.items())
    print("Modality family counts: " + ("PASS" if modality_pass else "FAIL"))
    print()

    print("-" * 80)
    print("6. FROZEN FIVE-GRU ARCHITECTURE")
    print("-" * 80)
    for branch, mapping in ARCHITECTURE.items():
        print(f"{branch.capitalize():<10} -> {mapping}")
    print("Exactly five modality-specific GRU branches.")
    print()

    print_contract()

    report = build_report(
        discovered,
        audited,
        modeling_status,
        preparation_status,
    )

    print("-" * 80)
    print("7. RUNTIME CONTRACT SUMMARY")
    print("-" * 80)
    summary = report["summary"]
    print(f"Files discovered:                  {len(discovered)}")
    print(f"Files audited:                     {len(audited)}")
    print(f"Schema failures:                   {summary['schema_failures']}")
    print(f"Numeric semantic failures:         {summary['numeric_failures']}")
    print(f"Categorical semantic failures:     {summary['categorical_failures']}")
    print(f"Structured semantic failures:      {summary['structured_failures']}")
    print(f"Timestamp failures:                {summary['timestamp_failures']}")
    print(f"Files requiring missing-value review: {summary['files_requiring_missing_value_review']}")
    print()

    print("-" * 80)
    print("8. WRITING AUDIT REPORT")
    print("-" * 80)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(json_safe(report), handle, indent=2, ensure_ascii=False)
    print(f"Audit report saved:\n  {REPORT_PATH}")
    print()

    print("=" * 80)
    if report["summary"]["overall_status"] == "PASS":
        print("T1D-UOM FIVE-GRU RUNTIME INPUT CONTRACT VALIDATION PASSED")
        print("=" * 80)
        print()
        print("FINAL RESULT: PASS")
        print(f"Files audited: {len(audited)}/{EXPECTED_FILE_COUNT}")
        print("Frozen 13-participant cohort: PASS")
        print("Modality family counts: PASS")
        print("Schema contract: PASS")
        print("Numeric semantic contract: PASS")
        print("Categorical semantic contract: PASS")
        print("Structured semantic contract: PASS")
        print("Timestamp contract: PASS")
        print()
        print("Missing-value reviews are informational only.")
        print("No imputation or representation transformation was performed.")
        print("No dataset or sequence-input CSV was modified.")
        print("No model was trained.")
        print("Embedding dimensions remain intentionally unfrozen.")
        print()
        print("The frozen physical/semantic runtime-input contract is ready for the next controlled stage.")
        print("=" * 80)
        return 0

    print("T1D-UOM FIVE-GRU RUNTIME INPUT CONTRACT VALIDATION FAILED")
    print("=" * 80)
    print()
    print("FINAL RESULT: FAIL")
    print("Inspect the JSON report for exact file-level failures.")
    print("No dataset or sequence-input CSV was modified.")
    print("=" * 80)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAUDIT INTERRUPTED BY USER.")
        print("No dataset or sequence-input files were modified.")
        sys.exit(130)
    except Exception as exc:
        print("\n" + "=" * 80)
        print("UNEXPECTED AUDIT ERROR")
        print("=" * 80)
        print(f"{type(exc).__name__}: {exc}")
        print("No dataset or sequence-input files were modified by the audit.")
        print("=" * 80)
        sys.exit(2)
