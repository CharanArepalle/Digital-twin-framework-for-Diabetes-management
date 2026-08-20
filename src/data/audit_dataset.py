from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================================
# T1D-UOM DATASET AUDIT
# ============================================================================
#
# READ-ONLY AUDIT
#
# This script:
#   - discovers all CSV files
#   - identifies participants
#   - identifies modalities
#   - validates schemas
#   - recognizes optional columns
#   - reports missingness
#   - reports empty rows/columns
#   - reports duplicate rows
#   - parses timestamps deterministically
#   - reports timestamp ranges
#   - reports duplicate timestamps
#   - reports suspicious years
#   - reports timestamp issue details
#   - reports numeric summaries
#   - reports participant/modality coverage
#   - determines the five-project-modality cohort
#   - determines the full-core cohort
#   - compares raw and derived datasets when available
#   - writes dataset_audit.json
#
# IMPORTANT:
#   RAW DATA ARE NEVER MODIFIED.
#   DERIVED DATA ARE NEVER MODIFIED.
#
# Run from anywhere:
#
#   python src\data\audit_dataset.py
#
# ============================================================================


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "t1d_uom_v1.0.3"
)

DERIVED_DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "t1d_uom_v1.0.3_timestamp_corrected"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_audit"
)

REPORT_PATH = (
    REPORT_DIR
    / "dataset_audit.json"
)


# ============================================================================
# DATASET EXPECTATIONS
# ============================================================================

# The current T1D-UOM project contains 112 expected CSV files.
# This is an audit expectation only. It does NOT cause files to be created,
# deleted, renamed, or modified.
EXPECTED_CSV_COUNT = 112


# ============================================================================
# MODALITIES
# ============================================================================

MODALITY_ORDER = [
    "activity",
    "basal_insulin",
    "bolus_insulin",
    "glucose",
    "nutrition",
    "sleep_summary",
    "sleep_timeseries",
]


# ============================================================================
# REQUIRED SCHEMA
# ============================================================================

EXPECTED_COLUMNS: dict[str, list[str]] = {
    "activity": [
        "activity_ts",
        "activity_type",
        "active_Kcal",
        "step_count",
        "distance_m",
        "duration_s",
        "active_time_s",
        "start_time_s",
        "start_time_offset_s",
        "met",
        "intensity",
        "motion_intensity_mean",
        "motion_intensity_max",
    ],
    "basal_insulin": [
        "basal_ts",
        "basal_dose",
        "insulin_kind",
    ],
    "bolus_insulin": [
        "bolus_ts",
        "bolus_dose",
    ],
    "glucose": [
        "bg_ts",
        "value",
    ],
    "nutrition": [
        "meal_ts",
        "meal_type",
        "meal_tag",
        "carbs_g",
        "prot_g",
        "fat_g",
        "fibre_g",
    ],
    "sleep_summary": [
        "calendar_date",
        "duration_in_sec",
        "start_date_ts",
        "start_time_offset_s",
        "unmeasurable_sleep_s",
        "deep_sleep_s",
        "light_sleep_s",
        "rem_sleep_s",
        "awake_s",
    ],
    "sleep_timeseries": [
        "sleep_ts",
        "step_count",
        "heart_rate",
        "current_activity_type_intensity",
        "stress_level_value",
        "sleep_level",
        "resting_heart_rate",
    ],
}


# ============================================================================
# RECOGNIZED OPTIONAL COLUMNS
# ============================================================================
#
# These columns may appear in the supplied T1D-UOM files and are intentionally
# NOT considered schema failures.
#
# Evidence from the existing audit shows that sleep_summary files can contain:
#   sleep_levels_map_deep
#   sleep_levels_map_light
#   sleep_levels_map_awake
#   sleep_levels_map_rem
#   sleep_levels_map_unmeasurable
#   validation
#
# There are also a small number of "Unnamed:" columns in supplied files.
# Those are recognized as optional audit artifacts rather than required schema.
# ============================================================================

OPTIONAL_COLUMNS: dict[str, set[str]] = {
    "activity": set(),
    "basal_insulin": {
        "Unnamed: 3",
        "Unnamed: 4",
    },
    "bolus_insulin": set(),
    "glucose": set(),
    "nutrition": set(),
    "sleep_summary": {
        "sleep_levels_map_deep",
        "sleep_levels_map.deep",
        "sleep_levels_map_light",
        "sleep_levels_map_awake",
        "sleep_levels_map_rem",
        "sleep_levels_map_unmeasurable",
        "validation",
        "Unnamed: 13",
    },
    "sleep_timeseries": set(),
}


# ============================================================================
# TIMESTAMP COLUMNS
# ============================================================================

TIMESTAMP_COLUMNS: dict[str, list[str]] = {
    "activity": [
        "activity_ts",
    ],
    "basal_insulin": [
        "basal_ts",
    ],
    "bolus_insulin": [
        "bolus_ts",
    ],
    "glucose": [
        "bg_ts",
    ],
    "nutrition": [
        "meal_ts",
    ],
    "sleep_summary": [
        "calendar_date",
        "start_date_ts",
    ],
    "sleep_timeseries": [
        "sleep_ts",
    ],
}


# ============================================================================
# NON-NUMERIC COLUMNS
# ============================================================================
#
# All other columns are examined with pd.to_numeric(errors="coerce") for
# numeric summaries.
# ============================================================================

NON_NUMERIC_COLUMNS = {
    "activity": {
        "activity_ts",
        "activity_type",
    },
    "basal_insulin": {
        "basal_ts",
        "insulin_kind",
    },
    "bolus_insulin": {
        "bolus_ts",
    },
    "glucose": {
        "bg_ts",
    },
    "nutrition": {
        "meal_ts",
        "meal_type",
        "meal_tag",
    },
    "sleep_summary": {
        "calendar_date",
        "start_date_ts",
        "validation",
    },
    "sleep_timeseries": {
        "sleep_ts",
    },
}


# ============================================================================
# TIMESTAMP QUALITY THRESHOLD
# ============================================================================
#
# These values ONLY determine whether a timestamp is flagged as suspicious.
# They DO NOT modify timestamps.
#
# The supplied project data span approximately 2023-2024, so this deliberately
# broad range catches obvious anomalies such as years 2033 and 2204.
# ============================================================================

MIN_REASONABLE_YEAR = 2020
MAX_REASONABLE_YEAR = 2026


# ============================================================================
# PARTICIPANT IDENTIFICATION
# ============================================================================

def extract_participant_id(
    file_path: Path,
) -> str | None:
    """
    Extract participant ID from all known T1D-UOM filename forms.

    Supported examples:

        UoM2301.csv
        UoMActivity2301.csv
        UoMGlucose2301.csv
        UoMBasal2301.csv
        UoMBolus2301.csv
        UoMNutrition2301.csv
        UoMsleep2301.csv
        UoM2301sleeptime.csv

    Returns canonical form:

        UoM2301
    """

    filename = file_path.name

    # IMPORTANT FIX:
    #
    # The previous broken audit used a pattern that expected "UoM" to be
    # immediately followed by the participant number. That works for
    # UoM2301sleeptime.csv but NOT for:
    #
    #   UoMActivity2301.csv
    #   UoMGlucose2301.csv
    #   UoMBasal2301.csv
    #   UoMBolus2301.csv
    #   UoMNutrition2301.csv
    #   UoMsleep2301.csv
    #
    # This pattern explicitly permits the modality text between UoM and
    # the four-digit participant code.
    #
    # 23xx and 24xx are the participant code families present in this
    # dataset.
    match = re.search(
        r"UoM(?:[A-Za-z]+)?(2[34]\d{2})",
        filename,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return f"UoM{match.group(1)}"


# ============================================================================
# MODALITY IDENTIFICATION
# ============================================================================

def identify_modality(
    file_path: Path,
) -> str:
    """
    Identify modality from directory and filename.

    Important:
        * sleeptime files = sleep_summary
        * sleepXXXX files = sleep_timeseries
    """

    path_string = str(file_path).lower()
    filename = file_path.name.lower()

    if "activity data" in path_string:
        return "activity"

    if "glucose data" in path_string:
        return "glucose"

    if "basal data" in path_string:
        return "basal_insulin"

    if "bolus data" in path_string:
        return "bolus_insulin"

    if "nutrition data" in path_string:
        return "nutrition"

    # Must be checked BEFORE generic "sleep data".
    if "sleeptime" in filename:
        return "sleep_summary"

    if "sleep data" in path_string:
        return "sleep_timeseries"

    return "unknown"


# ============================================================================
# DETERMINISTIC TIMESTAMP PARSING
# ============================================================================

def _parse_one_timestamp(
    value: Any,
) -> pd.Timestamp | Any:
    """
    Parse one timestamp deterministically.

    Supported families include:

        DD/MM/YYYY
        DD/MM/YYYY HH:MM
        DD/MM/YYYY HH:MM:SS
        YYYY-MM-DD
        YYYY-MM-DD HH:MM
        YYYY-MM-DD HH:MM:SS
        ISO-like YYYY-MM-DDTHH:MM...

    Day/month/year values are always interpreted with day-first semantics.
    """

    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()

    if not text:
        return pd.NaT

    # ------------------------------------------------------------------------
    # Day-first formats.
    # ------------------------------------------------------------------------

    if re.fullmatch(
        r"\d{1,2}/\d{1,2}/\d{4}"
        r"(?: \d{2}:\d{2}"
        r"(?::\d{2})?)?",
        text,
    ):
        parsed = pd.to_datetime(
            text,
            errors="coerce",
            dayfirst=True,
        )

        if pd.isna(parsed):
            return pd.NaT

        return parsed

    # ------------------------------------------------------------------------
    # ISO / standard formats.
    # ------------------------------------------------------------------------

    if re.match(
        r"^\d{4}-\d{2}-\d{2}",
        text,
    ):
        parsed = pd.to_datetime(
            text,
            errors="coerce",
            format="mixed",
        )

        if pd.isna(parsed):
            return pd.NaT

        # Remove timezone information if present so that the final Series
        # remains consistently datetime64[ns].
        if getattr(parsed, "tzinfo", None) is not None:

            try:
                parsed = parsed.tz_convert(None)
            except Exception:

                try:
                    parsed = parsed.tz_localize(None)
                except Exception:
                    return pd.NaT

        return parsed

    return pd.NaT


def parse_timestamp_series(
    series: pd.Series,
) -> pd.Series:
    """
    Parse a timestamp Series deterministically.
    """

    parsed = [
        _parse_one_timestamp(value)
        for value in series
    ]

    return pd.Series(
        parsed,
        index=series.index,
        dtype="datetime64[ns]",
    )


def detect_timestamp_strategy(
    series: pd.Series,
) -> str:
    """
    Identify the format family found in a timestamp column.
    """

    formats_found: set[str] = set()

    for value in series.dropna():

        text = str(value).strip()

        if not text:
            continue

        if re.fullmatch(
            r"\d{4}-\d{2}-\d{2}"
            r"(?: \d{2}:\d{2}"
            r"(?::\d{2})?)?",
            text,
        ):
            formats_found.add("standard")

        elif re.fullmatch(
            r"\d{1,2}/\d{1,2}/\d{4}"
            r"(?: \d{2}:\d{2}"
            r"(?::\d{2})?)?",
            text,
        ):
            formats_found.add("dayfirst")

        elif re.match(
            r"^\d{4}-\d{2}-\d{2}T",
            text,
        ):
            formats_found.add("standard")

        else:
            formats_found.add("unknown")

    if not formats_found:
        return "unknown"

    if formats_found == {"standard"}:
        return "standard"

    if formats_found == {"dayfirst"}:
        return "dayfirst"

    if formats_found == {"unknown"}:
        return "unknown"

    return "mixed"


# ============================================================================
# EMPTY DATA
# ============================================================================

def calculate_empty_row_count(
    df: pd.DataFrame,
) -> int:
    """
    Count rows where every field is empty/NaN.
    """

    if df.empty:
        return 0

    normalized = df.copy()

    for column in normalized.columns:

        normalized[column] = (
            normalized[column]
            .astype("string")
            .str.strip()
        )

    return int(
        normalized.isna()
        .all(axis=1)
        .sum()
    )


def calculate_empty_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Identify columns that contain no usable values.
    """

    empty_columns: list[str] = []

    for column in df.columns:

        text = (
            df[column]
            .astype("string")
            .str.strip()
        )

        if (
            text.isna().all()
            or text.eq("").all()
        ):
            empty_columns.append(
                str(column)
            )

    return empty_columns


# ============================================================================
# MISSINGNESS
# ============================================================================

def calculate_missingness(
    df: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate missing fraction per column.

    Missing means:
        * NaN
        * empty string
        * whitespace-only string
    """

    if len(df) == 0:

        return {
            str(column): 1.0
            for column in df.columns
        }

    result: dict[str, float] = {}

    for column in df.columns:

        series = df[column]

        missing_mask = (
            series.isna()
            | series.astype("string")
            .str.strip()
            .eq("")
        )

        result[str(column)] = round(
            float(
                missing_mask.mean()
            ),
            6,
        )

    return result


# ============================================================================
# NUMERIC ANALYSIS
# ============================================================================

def calculate_numeric_analysis(
    df: pd.DataFrame,
    modality: str,
) -> dict[str, dict[str, Any]]:
    """
    Calculate numeric missingness, min, max and mean.

    Timestamp and known categorical columns are excluded.
    Unnamed audit artifacts are excluded.
    """

    results: dict[
        str,
        dict[str, Any],
    ] = {}

    excluded = NON_NUMERIC_COLUMNS.get(
        modality,
        set(),
    )

    for column in df.columns:

        column = str(column)

        if column in excluded:
            continue

        if column.startswith("Unnamed:"):
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        valid = numeric.dropna()

        missing_count = int(
            numeric.isna().sum()
        )

        valid_count = int(
            valid.count()
        )

        result: dict[str, Any] = {
            "missing_count": missing_count,
            "missing_fraction": round(
                (
                    missing_count / len(df)
                    if len(df) > 0
                    else 1.0
                ),
                6,
            ),
            "valid_count": valid_count,
        }

        if valid_count > 0:

            result.update(
                {
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                    "mean": float(valid.mean()),
                }
            )

        else:

            result.update(
                {
                    "min": None,
                    "max": None,
                    "mean": None,
                }
            )

        results[column] = result

    return results


# ============================================================================
# TIMESTAMP ANALYSIS
# ============================================================================

def calculate_timestamp_analysis(
    df: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """
    Analyze one timestamp column.
    """

    original = df[column]

    parsed = parse_timestamp_series(
        original
    )

    strategy = detect_timestamp_strategy(
        original
    )

    valid = parsed.dropna()

    result: dict[str, Any] = {
        "selected_strategy": strategy,
        "row_count": int(len(df)),
        "valid_count": int(len(valid)),
        "missing_or_unparseable_count": int(
            parsed.isna().sum()
        ),
        "start": None,
        "end": None,
        "duplicate_timestamp_count": 0,
        "suspicious_year_count": 0,
        "suspicious_years": [],
    }

    if len(valid) == 0:
        return result

    result["start"] = str(valid.min())
    result["end"] = str(valid.max())

    result["duplicate_timestamp_count"] = int(
        valid.duplicated(
            keep=False
        ).sum()
    )

    suspicious_mask = (
        (valid.dt.year < MIN_REASONABLE_YEAR)
        | (valid.dt.year > MAX_REASONABLE_YEAR)
    )

    suspicious = valid[
        suspicious_mask
    ]

    result["suspicious_year_count"] = int(
        len(suspicious)
    )

    result["suspicious_years"] = [
        int(year)
        for year in sorted(
            suspicious.dt.year
            .unique()
            .tolist()
        )
    ]

    return result


def get_timestamp_issue_details(
    df: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """
    Return row-level details for timestamp problems.
    """

    original = df[column]

    parsed = parse_timestamp_series(
        original
    )

    invalid_mask = (
        original.isna()
        | original.astype("string")
        .str.strip()
        .eq("")
        | parsed.isna()
    )

    invalid_values = []

    for index in original.index[
        invalid_mask
    ]:

        value = original.loc[index]

        invalid_values.append(
            {
                "row_index": int(index),
                "original_value": (
                    None
                    if pd.isna(value)
                    else str(value)
                ),
            }
        )

    valid = parsed.dropna()

    suspicious_mask = (
        (valid.dt.year < MIN_REASONABLE_YEAR)
        | (valid.dt.year > MAX_REASONABLE_YEAR)
    )

    suspicious_values = []

    for index in valid.index[
        suspicious_mask
    ]:

        original_value = (
            original.loc[index]
        )

        parsed_value = (
            parsed.loc[index]
        )

        suspicious_values.append(
            {
                "row_index": int(index),
                "original_value": (
                    None
                    if pd.isna(original_value)
                    else str(original_value)
                ),
                "parsed_value": str(
                    parsed_value
                ),
            }
        )

    return {
        "invalid_or_unparseable_values": (
            invalid_values
        ),
        "suspicious_timestamp_values": (
            suspicious_values
        ),
    }


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

def validate_schema(
    columns: list[str],
    modality: str,
) -> dict[str, Any]:
    """
    Validate required columns while recognizing approved optional columns.

    A file is schema-valid when ALL required columns are present.

    Approved optional columns do not make a file invalid.
    """

    expected = EXPECTED_COLUMNS[
        modality
    ]

    optional = OPTIONAL_COLUMNS.get(
        modality,
        set(),
    )

    actual_set = set(columns)
    expected_set = set(expected)

    missing_required = [
        column
        for column in expected
        if column not in actual_set
    ]

    recognized_optional_present = [
        column
        for column in columns
        if column in optional
    ]

    empty_optional_columns = []

    nonempty_optional_columns = []

    # Empty/nonempty optional-column classification is performed by
    # audit_file() after the DataFrame has been loaded. These lists are
    # initially populated there.
    #
    # Here we identify only truly unexpected columns.
    truly_unexpected_columns = [
        column
        for column in columns
        if (
            column not in expected_set
            and column not in optional
            and not column.startswith("Unnamed:")
        )
    ]

    schema_valid = (
        len(missing_required) == 0
        and len(truly_unexpected_columns) == 0
    )

    return {
        "required_columns": expected,
        "recognized_optional_columns": sorted(
            optional
        ),
        "present_columns": columns,
        "missing_required_columns": (
            missing_required
        ),
        "recognized_optional_present": (
            recognized_optional_present
        ),
        "empty_optional_columns": (
            empty_optional_columns
        ),
        "nonempty_optional_columns": (
            nonempty_optional_columns
        ),
        "truly_unexpected_columns": (
            truly_unexpected_columns
        ),
        "schema_valid": schema_valid,
    }


# ============================================================================
# FILE AUDIT
# ============================================================================

def audit_file(
    file_path: Path,
) -> dict[str, Any]:
    """
    Audit a single CSV file.
    """

    participant_id = (
        extract_participant_id(
            file_path
        )
    )

    modality = identify_modality(
        file_path
    )

    result: dict[str, Any] = {
        "file": str(file_path),
        "status": "ok",
        "filename": file_path.name,
        "participant_id": participant_id,
        "modality": modality,
    }

    if modality == "unknown":

        result["status"] = "error"
        result["error"] = (
            "Unable to identify modality."
        )

        return result

    try:

        df = pd.read_csv(
            file_path,
            low_memory=False,
        )

    except Exception as exc:

        result["status"] = "error"
        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

        return result

    # ------------------------------------------------------------------------
    # Basic structure
    # ------------------------------------------------------------------------

    columns = [
        str(column)
        for column in df.columns
    ]

    result["rows"] = int(len(df))
    result["columns"] = columns
    result["column_count"] = int(
        len(columns)
    )

    result["empty_row_count"] = (
        calculate_empty_row_count(df)
    )

    empty_columns = (
        calculate_empty_columns(df)
    )

    result["empty_column_count"] = (
        len(empty_columns)
    )

    result["empty_columns"] = (
        empty_columns
    )

    result["duplicate_row_count"] = int(
        df.duplicated().sum()
    )

    result["missingness"] = (
        calculate_missingness(df)
    )

    # ------------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------------

    schema = validate_schema(
        columns,
        modality,
    )

    optional_columns = OPTIONAL_COLUMNS.get(
        modality,
        set(),
    )

    empty_optional_columns = []
    nonempty_optional_columns = []

    for column in columns:

        if column not in optional_columns:
            continue

        text = (
            df[column]
            .astype("string")
            .str.strip()
        )

        if (
            text.isna().all()
            or text.eq("").all()
        ):
            empty_optional_columns.append(
                column
            )
        else:
            nonempty_optional_columns.append(
                column
            )

    schema[
        "empty_optional_columns"
    ] = sorted(
        empty_optional_columns
    )

    schema[
        "nonempty_optional_columns"
    ] = sorted(
        nonempty_optional_columns
    )

    result["schema_validation"] = schema

    # ------------------------------------------------------------------------
    # Timestamp analysis
    # ------------------------------------------------------------------------

    timestamp_analysis = {}
    timestamp_issue_details = {}

    for timestamp_column in TIMESTAMP_COLUMNS[
        modality
    ]:

        if timestamp_column not in df.columns:

            timestamp_analysis[
                timestamp_column
            ] = {
                "selected_strategy": (
                    "not_available"
                ),
                "row_count": int(len(df)),
                "valid_count": 0,
                "missing_or_unparseable_count": int(
                    len(df)
                ),
                "start": None,
                "end": None,
                "duplicate_timestamp_count": 0,
                "suspicious_year_count": 0,
                "suspicious_years": [],
            }

            timestamp_issue_details[
                timestamp_column
            ] = {
                "missing_column": True
            }

            continue

        timestamp_analysis[
            timestamp_column
        ] = calculate_timestamp_analysis(
            df,
            timestamp_column,
        )

        timestamp_issue_details[
            timestamp_column
        ] = get_timestamp_issue_details(
            df,
            timestamp_column,
        )

    result["timestamp_analysis"] = (
        timestamp_analysis
    )

    # Only include detailed timestamp issue information when there is
    # something to report. This keeps the JSON cleaner.
    useful_timestamp_details = {}

    for column, details in (
        timestamp_issue_details.items()
    ):

        if details.get(
            "missing_column",
            False,
        ):

            useful_timestamp_details[
                column
            ] = details

            continue

        if (
            details.get(
                "invalid_or_unparseable_values",
                [],
            )
            or details.get(
                "suspicious_timestamp_values",
                [],
            )
        ):

            useful_timestamp_details[
                column
            ] = details

    if useful_timestamp_details:

        result[
            "timestamp_issue_details"
        ] = useful_timestamp_details

    # ------------------------------------------------------------------------
    # Numeric analysis
    # ------------------------------------------------------------------------

    result["numeric_analysis"] = (
        calculate_numeric_analysis(
            df,
            modality,
        )
    )

    result["errors"] = []

    return result


# ============================================================================
# MODALITY SUMMARY
# ============================================================================

def build_modality_summary(
    reports: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Build per-modality aggregate statistics.
    """

    summary: dict[str, dict[str, Any]] = {}

    for modality in MODALITY_ORDER:

        modality_reports = [
            report
            for report in reports
            if (
                report.get("status") == "ok"
                and report.get("modality")
                == modality
            )
        ]

        participants = sorted(
            {
                report["participant_id"]
                for report in modality_reports
                if report.get(
                    "participant_id"
                ) is not None
            }
        )

        total_rows = sum(
            int(
                report.get(
                    "rows",
                    0,
                )
            )
            for report in modality_reports
        )

        total_empty_rows = sum(
            int(
                report.get(
                    "empty_row_count",
                    0,
                )
            )
            for report in modality_reports
        )

        total_duplicate_rows = sum(
            int(
                report.get(
                    "duplicate_row_count",
                    0,
                )
            )
            for report in modality_reports
        )

        # IMPORTANT FIX:
        #
        # Count schema validity directly from each file's schema_validation
        # dictionary. Do NOT depend on participant extraction or another
        # aggregate condition.
        schema_valid_files = sum(
            1
            for report in modality_reports
            if report.get(
                "schema_validation",
                {},
            ).get(
                "schema_valid",
                False,
            )
        )

        schema_invalid_files = (
            len(modality_reports)
            - schema_valid_files
        )

        summary[modality] = {
            "file_count": len(
                modality_reports
            ),
            "participants": participants,
            "total_rows": int(
                total_rows
            ),
            "empty_rows": int(
                total_empty_rows
            ),
            "duplicate_rows": int(
                total_duplicate_rows
            ),
            "schema_valid_files": int(
                schema_valid_files
            ),
            "schema_invalid_files": int(
                schema_invalid_files
            ),
            "participant_count": len(
                participants
            ),
        }

    return summary


# ============================================================================
# PARTICIPANT COVERAGE
# ============================================================================

def build_participant_modality_coverage(
    reports: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """
    Build participant -> modalities mapping.
    """

    coverage: dict[
        str,
        set[str],
    ] = {}

    for report in reports:

        if report.get("status") != "ok":
            continue

        participant = report.get(
            "participant_id"
        )

        modality = report.get(
            "modality"
        )

        if participant is None:
            continue

        if modality not in MODALITY_ORDER:
            continue

        coverage.setdefault(
            participant,
            set(),
        ).add(modality)

    return {
        participant: [
            modality
            for modality in MODALITY_ORDER
            if modality in modalities
        ]
        for participant, modalities
        in sorted(
            coverage.items()
        )
    }


# ============================================================================
# COHORTS
# ============================================================================

def build_cohorts(
    coverage: dict[str, list[str]],
) -> tuple[
    list[str],
    list[str],
]:
    """
    Determine:

        five_project:
            activity
            basal_insulin
            bolus_insulin
            glucose
            nutrition

        full_core:
            all five above
            +
            sleep_timeseries
    """

    five_project_modalities = {
        "activity",
        "basal_insulin",
        "bolus_insulin",
        "glucose",
        "nutrition",
    }

    full_core_modalities = (
        five_project_modalities
        | {
            "sleep_timeseries",
        }
    )

    five_project = []
    full_core = []

    for participant, modalities in (
        coverage.items()
    ):

        modality_set = set(
            modalities
        )

        if five_project_modalities.issubset(
            modality_set
        ):

            five_project.append(
                participant
            )

        if full_core_modalities.issubset(
            modality_set
        ):

            full_core.append(
                participant
            )

    return (
        sorted(five_project),
        sorted(full_core),
    )


# ============================================================================
# TIMESTAMP QUALITY
# ============================================================================

def build_timestamp_quality_summary(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build dataset-wide timestamp quality summary.
    """

    files_with_timestamp_issues = []
    files_with_suspicious_timestamps = []

    for report in reports:

        if report.get("status") != "ok":
            continue

        analyses = report.get(
            "timestamp_analysis",
            {},
        )

        has_issue = False
        has_suspicious = False

        for analysis in analyses.values():

            if (
                analysis.get(
                    "missing_or_unparseable_count",
                    0,
                )
                > 0
            ):

                has_issue = True

            if (
                analysis.get(
                    "suspicious_year_count",
                    0,
                )
                > 0
            ):

                has_suspicious = True

        if has_issue:

            files_with_timestamp_issues.append(
                report["filename"]
            )

        if has_suspicious:

            files_with_suspicious_timestamps.append(
                report["filename"]
            )

    return {
        "files_with_timestamp_issues": (
            files_with_timestamp_issues
        ),
        "files_with_suspicious_timestamps": (
            files_with_suspicious_timestamps
        ),
        "timestamp_issue_file_count": len(
            files_with_timestamp_issues
        ),
        "suspicious_timestamp_file_count": len(
            files_with_suspicious_timestamps
        ),
    }


# ============================================================================
# RAW / DERIVED COMPARISON
# ============================================================================

def compare_raw_and_derived() -> dict[str, Any]:
    """
    Compare raw and derived datasets without modifying either one.

    Comparison is byte-level for files with matching relative paths.
    """

    result: dict[str, Any] = {
        "available": False,
        "raw_exists": DATASET_ROOT.exists(),
        "derived_exists": (
            DERIVED_DATASET_ROOT.exists()
        ),
        "raw_csv_count": 0,
        "derived_csv_count": 0,
        "file_list_identical": False,
        "byte_identical_files": 0,
        "changed_files": [],
        "missing_from_derived": [],
        "missing_from_raw": [],
        "errors": [],
    }

    if not DATASET_ROOT.exists():
        result["errors"].append(
            "Raw dataset directory does not exist."
        )
        return result

    if not DERIVED_DATASET_ROOT.exists():
        result["errors"].append(
            "Derived dataset directory does not exist."
        )
        return result

    try:

        raw_files = {
            path.relative_to(DATASET_ROOT)
            for path in DATASET_ROOT.rglob("*.csv")
        }

        derived_files = {
            path.relative_to(
                DERIVED_DATASET_ROOT
            )
            for path in DERIVED_DATASET_ROOT.rglob(
                "*.csv"
            )
        }

        result["available"] = True

        result["raw_csv_count"] = len(
            raw_files
        )

        result["derived_csv_count"] = len(
            derived_files
        )

        missing_from_derived = sorted(
            raw_files - derived_files,
            key=lambda path: str(path).lower(),
        )

        missing_from_raw = sorted(
            derived_files - raw_files,
            key=lambda path: str(path).lower(),
        )

        result[
            "missing_from_derived"
        ] = [
            str(path)
            for path in missing_from_derived
        ]

        result[
            "missing_from_raw"
        ] = [
            str(path)
            for path in missing_from_raw
        ]

        result[
            "file_list_identical"
        ] = (
            raw_files == derived_files
        )

        byte_identical = 0
        changed_files = []

        for relative_path in sorted(
            raw_files & derived_files,
            key=lambda path: str(path).lower(),
        ):

            raw_path = (
                DATASET_ROOT
                / relative_path
            )

            derived_path = (
                DERIVED_DATASET_ROOT
                / relative_path
            )

            try:

                if (
                    raw_path.read_bytes()
                    == derived_path.read_bytes()
                ):

                    byte_identical += 1

                else:

                    changed_files.append(
                        str(relative_path)
                    )

            except Exception as exc:

                result["errors"].append(
                    (
                        f"{relative_path}: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )
                )

        result[
            "byte_identical_files"
        ] = byte_identical

        result[
            "changed_files"
        ] = changed_files

    except Exception as exc:

        result["errors"].append(
            f"{type(exc).__name__}: {exc}"
        )

    return result


# ============================================================================
# GLOBAL REPORT
# ============================================================================

def build_report(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build complete JSON audit report.
    """

    successful_reports = [
        report
        for report in reports
        if report.get("status") == "ok"
    ]

    failed_reports = [
        report
        for report in reports
        if report.get("status") != "ok"
    ]

    participants = sorted(
        {
            report["participant_id"]
            for report in successful_reports
            if report.get(
                "participant_id"
            ) is not None
        }
    )

    modality_summary = (
        build_modality_summary(
            reports
        )
    )

    coverage = (
        build_participant_modality_coverage(
            reports
        )
    )

    (
        five_project,
        full_core,
    ) = build_cohorts(
        coverage
    )

    timestamp_quality = (
        build_timestamp_quality_summary(
            reports
        )
    )

    schema_invalid_files = [
        report["filename"]
        for report in successful_reports
        if not report.get(
            "schema_validation",
            {},
        ).get(
            "schema_valid",
            False,
        )
    ]

    unexpected_columns_by_file = {}

    for report in successful_reports:

        schema = report.get(
            "schema_validation",
            {},
        )

        unexpected = schema.get(
            "truly_unexpected_columns",
            [],
        )

        if unexpected:

            unexpected_columns_by_file[
                report["filename"]
            ] = unexpected

    raw_derived_comparison = (
        compare_raw_and_derived()
    )

    return {
        "dataset_root": str(
            DATASET_ROOT
        ),
        "derived_dataset_root": str(
            DERIVED_DATASET_ROOT
        ),
        "expected_csv_count": (
            EXPECTED_CSV_COUNT
        ),
        "csv_file_count": len(
            reports
        ),
        "successful_file_count": len(
            successful_reports
        ),
        "failed_file_count": len(
            failed_reports
        ),
        "participants": participants,
        "participant_count": len(
            participants
        ),
        "participants_with_five_project_modalities": (
            five_project
        ),
        "participants_with_all_core_modalities": (
            full_core
        ),
        "participant_count_with_five_project_modalities": (
            len(five_project)
        ),
        "participant_count_with_all_core_modalities": (
            len(full_core)
        ),
        "modality_summary": (
            modality_summary
        ),
        "participant_modality_coverage": (
            coverage
        ),
        "timestamp_quality": (
            timestamp_quality
        ),
        "schema_invalid_files": (
            schema_invalid_files
        ),
        "unexpected_columns_by_file": (
            unexpected_columns_by_file
        ),
        "raw_derived_comparison": (
            raw_derived_comparison
        ),
        "files": reports,
    }


# ============================================================================
# CONSOLE OUTPUT
# ============================================================================

def print_modality_summary(
    summary: dict[str, Any],
) -> None:

    print()
    print("Modality summary:")
    print("-" * 80)

    for modality in MODALITY_ORDER:

        item = summary[
            modality
        ]

        print(
            f"{modality:<20} | "
            f"files={item['file_count']} | "
            f"participants={item['participant_count']} | "
            f"rows={item['total_rows']:,} | "
            f"schema-valid={item['schema_valid_files']}"
        )


def print_participant_coverage(
    coverage: dict[str, list[str]],
) -> None:

    print()
    print(
        "Participant modality coverage:"
    )
    print("-" * 80)

    for participant, modalities in (
        coverage.items()
    ):

        print(
            f"{participant}: "
            f"{', '.join(modalities)}"
        )


def print_timestamp_quality(
    quality: dict[str, Any],
) -> None:

    print()
    print("Timestamp quality:")
    print("-" * 80)

    print(
        "Files with timestamp issues: "
        f"{quality['timestamp_issue_file_count']}"
    )

    print(
        "Files with suspicious timestamps: "
        f"{quality['suspicious_timestamp_file_count']}"
    )

    if quality[
        "files_with_timestamp_issues"
    ]:

        print()
        print(
            "Timestamp-issue files:"
        )

        for filename in quality[
            "files_with_timestamp_issues"
        ]:

            print(
                f"  - {filename}"
            )

    if quality[
        "files_with_suspicious_timestamps"
    ]:

        print()
        print(
            "Suspicious-timestamp files:"
        )

        for filename in quality[
            "files_with_suspicious_timestamps"
        ]:

            print(
                f"  - {filename}"
            )


def print_raw_derived_comparison(
    comparison: dict[str, Any],
) -> None:

    print()
    print(
        "Raw / derived dataset comparison:"
    )
    print("-" * 80)

    if not comparison.get(
        "available",
        False,
    ):

        print(
            "Comparison unavailable."
        )

        for error in comparison.get(
            "errors",
            [],
        ):

            print(
                f"  - {error}"
            )

        return

    print(
        f"Raw CSV files:       "
        f"{comparison['raw_csv_count']}"
    )

    print(
        f"Derived CSV files:   "
        f"{comparison['derived_csv_count']}"
    )

    print(
        "CSV file-list preservation: "
        + (
            "PASS"
            if comparison[
                "file_list_identical"
            ]
            else "FAIL"
        )
    )

    print(
        f"Byte-identical CSV files: "
        f"{comparison['byte_identical_files']}"
    )

    print(
        f"Changed CSV files: "
        f"{len(comparison['changed_files'])}"
    )

    for filename in comparison[
        "changed_files"
    ]:

        print(
            f"  - {filename}"
        )

    if comparison[
        "missing_from_derived"
    ]:

        print()
        print(
            "Missing from derived:"
        )

        for filename in comparison[
            "missing_from_derived"
        ]:

            print(
                f"  - {filename}"
            )

    if comparison[
        "missing_from_raw"
    ]:

        print()
        print(
            "Present only in derived:"
        )

        for filename in comparison[
            "missing_from_raw"
        ]:

            print(
                f"  - {filename}"
            )

    if comparison[
        "errors"
    ]:

        print()
        print(
            "Comparison errors:"
        )

        for error in comparison[
            "errors"
        ]:

            print(
                f"  - {error}"
            )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 80)
    print("T1D-UOM DATASET AUDIT")
    print("=" * 80)

    print()
    print(
        "IMPORTANT: READ-ONLY."
    )
    print(
        "Raw and working datasets will NOT be modified."
    )

    print()
    print(
        f"Raw dataset:     {DATASET_ROOT}"
    )

    print(
        f"Working dataset: {DERIVED_DATASET_ROOT}"
    )

    # ------------------------------------------------------------------------
    # Validate raw dataset root.
    # ------------------------------------------------------------------------

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            "Raw dataset directory does not exist:\n"
            f"{DATASET_ROOT}"
        )

    if not DATASET_ROOT.is_dir():

        raise NotADirectoryError(
            "Raw dataset root is not a directory:\n"
            f"{DATASET_ROOT}"
        )

    # ------------------------------------------------------------------------
    # Discover CSV files.
    # ------------------------------------------------------------------------

    csv_files = sorted(
        DATASET_ROOT.rglob("*.csv"),
        key=lambda path: str(
            path
        ).lower(),
    )

    if not csv_files:

        raise FileNotFoundError(
            "No CSV files found under:\n"
            f"{DATASET_ROOT}"
        )

    print()
    print(
        f"CSV files discovered: "
        f"{len(csv_files)}"
    )

    print(
        f"Expected CSV files: "
        f"{EXPECTED_CSV_COUNT}"
    )

    # ------------------------------------------------------------------------
    # Audit every file.
    # ------------------------------------------------------------------------

    reports = []

    for index, file_path in enumerate(
        csv_files,
        start=1,
    ):

        print(
            f"[{index:03d}/{len(csv_files):03d}] "
            f"Inspecting: {file_path}"
        )

        reports.append(
            audit_file(
                file_path
            )
        )

    # ------------------------------------------------------------------------
    # Build final report.
    # ------------------------------------------------------------------------

    final_report = build_report(
        reports
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            final_report,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    # ------------------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "T1D-UOM DATASET AUDIT COMPLETE"
    )
    print("=" * 80)

    print(
        f"CSV files found: "
        f"{final_report['csv_file_count']}"
    )

    print(
        f"Successfully inspected: "
        f"{final_report['successful_file_count']}"
    )

    print(
        f"Failed inspections: "
        f"{final_report['failed_file_count']}"
    )

    print(
        f"Participants identified: "
        f"{final_report['participant_count']}"
    )

    print(
        "Participants with all five "
        "non-sleep project modalities: "
        f"{final_report['participant_count_with_five_project_modalities']}"
    )

    print(
        "Participants with all core modalities "
        "(including sleep time-series): "
        f"{final_report['participant_count_with_all_core_modalities']}"
    )

    print_modality_summary(
        final_report[
            "modality_summary"
        ]
    )

    print_participant_coverage(
        final_report[
            "participant_modality_coverage"
        ]
    )

    print_timestamp_quality(
        final_report[
            "timestamp_quality"
        ]
    )

    # ------------------------------------------------------------------------
    # Schema issues.
    # ------------------------------------------------------------------------

    schema_invalid_files = final_report[
        "schema_invalid_files"
    ]

    unexpected_columns = final_report[
        "unexpected_columns_by_file"
    ]

    print()
    print(
        "Schema validation summary:"
    )
    print("-" * 80)

    print(
        f"Expected CSV files:      "
        f"{EXPECTED_CSV_COUNT}"
    )

    print(
        f"Discovered CSV files:    "
        f"{final_report['csv_file_count']}"
    )

    print(
        f"Successfully inspected:  "
        f"{final_report['successful_file_count']}"
    )

    print(
        f"Failed inspections:      "
        f"{final_report['failed_file_count']}"
    )

    print(
        f"Schema-invalid files:    "
        f"{len(schema_invalid_files)}"
    )

    print(
        f"Timestamp issues:        "
        f"{final_report['timestamp_quality']['timestamp_issue_file_count']}"
    )

    print(
        f"Suspicious timestamps:   "
        f"{final_report['timestamp_quality']['suspicious_timestamp_file_count']}"
    )

    if schema_invalid_files:

        print()
        print(
            "Schema-invalid files:"
        )

        for filename in schema_invalid_files:

            print(
                f"  - {filename}"
            )

    if unexpected_columns:

        print()
        print(
            "Files containing truly unexpected columns:"
        )

        for filename, columns in (
            unexpected_columns.items()
        ):

            print(
                f"  - {filename}: "
                f"{', '.join(columns)}"
            )

    # ------------------------------------------------------------------------
    # Cohorts.
    # ------------------------------------------------------------------------

    print()
    print(
        "Five-project-modality participants:"
    )
    print("-" * 80)

    for participant in final_report[
        "participants_with_five_project_modalities"
    ]:

        print(
            f"  - {participant}"
        )

    print()
    print(
        "Full-core participants:"
    )
    print("-" * 80)

    for participant in final_report[
        "participants_with_all_core_modalities"
    ]:

        print(
            f"  - {participant}"
        )

    # ------------------------------------------------------------------------
    # Raw / derived comparison.
    # ------------------------------------------------------------------------

    print_raw_derived_comparison(
        final_report[
            "raw_derived_comparison"
        ]
    )

    # ------------------------------------------------------------------------
    # Failed inspections.
    # ------------------------------------------------------------------------

    failed_reports = [
        report
        for report in reports
        if report.get(
            "status"
        ) != "ok"
    ]

    if failed_reports:

        print()
        print(
            "Failed inspections:"
        )
        print("-" * 80)

        for report in failed_reports:

            print(
                f"{report.get('filename', '<unknown>')}: "
                f"{report.get('error', 'Unknown error')}"
            )

    # ------------------------------------------------------------------------
    # Final status.
    # ------------------------------------------------------------------------

    print()
    print(
        "Audit report saved to:"
    )

    print(
        REPORT_PATH
    )

    print()
    print(
        "Raw dataset was not modified."
    )

    print()
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Determine audit status.
    #
    # A suspicious timestamp is a REVIEW condition, not automatically a
    # schema failure. This is important for the two known nutrition anomalies.
    # ------------------------------------------------------------------------

    audit_ok = True
    review_reasons = []

    if final_report[
        "csv_file_count"
    ] != EXPECTED_CSV_COUNT:

        audit_ok = False

        review_reasons.append(
            (
                f"Expected {EXPECTED_CSV_COUNT} "
                f"CSV files but found "
                f"{final_report['csv_file_count']}."
            )
        )

    if final_report[
        "failed_file_count"
    ] != 0:

        audit_ok = False

        review_reasons.append(
            (
                f"{final_report['failed_file_count']} "
                "files failed inspection."
            )
        )

    if schema_invalid_files:

        audit_ok = False

        review_reasons.append(
            (
                f"{len(schema_invalid_files)} "
                "files failed schema validation."
            )
        )

    timestamp_issue_count = final_report[
        "timestamp_quality"
    ][
        "timestamp_issue_file_count"
    ]

    suspicious_timestamp_count = final_report[
        "timestamp_quality"
    ][
        "suspicious_timestamp_file_count"
    ]

    if timestamp_issue_count > 0:

        audit_ok = False

        review_reasons.append(
            (
                f"{timestamp_issue_count} "
                "files contain timestamp parsing issues."
            )
        )

    print()

    if audit_ok and suspicious_timestamp_count == 0:

        print(
            "AUDIT STATUS: PASS"
        )

    elif audit_ok and suspicious_timestamp_count > 0:

        print(
            "AUDIT STATUS: REVIEW REQUIRED"
        )

        print()

        print(
            f"  - {suspicious_timestamp_count} "
            "files contain suspicious timestamps."
        )

        print(
            "  - No automatic timestamp correction was performed."
        )

    else:

        print(
            "AUDIT STATUS: REVIEW REQUIRED"
        )

        for reason in review_reasons:

            print(
                f"  - {reason}"
            )

        if suspicious_timestamp_count > 0:

            print(
                f"  - {suspicious_timestamp_count} "
                "files contain suspicious timestamps."
            )

    print()


if __name__ == "__main__":
    main()