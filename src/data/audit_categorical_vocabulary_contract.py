from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# T1D-UOM CATEGORICAL VOCABULARY / REPRESENTATION CONTRACT AUDIT
# =============================================================================
#
# PURPOSE
# -------
# Read-only audit of categorical/raw fields present in the frozen sequence
# inputs. This stage DISCOVERS and DOCUMENTS the actual categorical domains.
#
# THIS SCRIPT DOES NOT:
#   - modify dataset files
#   - modify sequence-input files
#   - transform values
#   - normalize values
#   - strip whitespace in source data
#   - merge categories
#   - encode categories
#   - create embeddings
#   - create one-hot vectors
#   - create event_type
#   - create windows
#   - create targets
#   - impute values
#   - resample data
#   - train a model
#   - implement any GRU
#   - implement MLP Fusion
#   - implement Digital Twin
#   - implement Prediction / What-if
#   - implement Interactive UI
#
# IMPORTANT:
#   The audit report is the ONLY file written by this script.
#
# ARCHITECTURE BOUNDARY
# ---------------------
#   Glucose    -> GRU -> zG
#   Insulin    -> GRU -> zI
#   Nutrition  -> GRU -> zN
#   Activity   -> GRU -> zA
#   Sleep      -> GRU -> zS
#
# No additional GRU branch is created by this audit.
# =============================================================================


SCRIPT_VERSION = "1.0.0"

PROJECT_NAME = "T1D-UOM"

EXPECTED_FILE_COUNT = 86

EXPECTED_PARTICIPANTS = [
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

EXPECTED_MODALITY_COUNTS = {
    "activity": 13,
    "glucose": 13,
    "basal_insulin": 12,
    "bolus_insulin": 13,
    "nutrition": 13,
    "sleep_summary": 11,
    "sleep_timeseries": 11,
}

# -------------------------------------------------------------------------
# Physical / semantic contracts already established by previous audits.
# This script DOES NOT change them.
# -------------------------------------------------------------------------

FILE_CONTRACTS = {
    "activity": {
        "timestamp": "activity_ts",
        "columns": [
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
        "categorical": [
            "activity_type",
            "intensity",
        ],
        "selected_gru_branch": True,
    },
    "glucose": {
        "timestamp": "bg_ts",
        "columns": [
            "bg_ts",
            "value",
        ],
        "categorical": [],
        "selected_gru_branch": True,
    },
    "basal_insulin": {
        "timestamp": "basal_ts",
        "columns": [
            "basal_ts",
            "basal_dose",
            "insulin_kind",
        ],
        "categorical": [
            "insulin_kind",
        ],
        "selected_gru_branch": True,
    },
    "bolus_insulin": {
        "timestamp": "bolus_ts",
        "columns": [
            "bolus_ts",
            "bolus_dose",
        ],
        "categorical": [],
        "selected_gru_branch": True,
    },
    "nutrition": {
        "timestamp": "meal_ts",
        "columns": [
            "meal_ts",
            "meal_type",
            "meal_tag",
            "carbs_g",
            "prot_g",
            "fat_g",
            "fibre_g",
        ],
        "categorical": [
            "meal_type",
            "meal_tag",
        ],
        "selected_gru_branch": True,
    },
    "sleep_summary": {
        "timestamp": "start_date_ts",
        "columns": [
            "start_date_ts",
            "duration_in_sec",
            "start_time_offset_s",
            "unmeasurable_sleep_s",
            "deep_sleep_s",
            "light_sleep_s",
            "rem_sleep_s",
            "awake_s",
            "calendar_date",
            "validation",
            "sleep_levels_map_deep",
            "sleep_levels_map_light",
            "sleep_levels_map_awake",
            "sleep_levels_map_rem",
            "sleep_levels_map_unmeasurable",
        ],
        "categorical": [
            "calendar_date",
            "validation",
        ],
        "selected_gru_branch": False,
    },
    "sleep_timeseries": {
        "timestamp": "sleep_ts",
        "columns": [
            "sleep_ts",
            "step_count",
            "heart_rate",
            "current_activity_type_intensity",
            "stress_level_value",
            "sleep_level",
            "resting_heart_rate",
        ],
        "categorical": [],
        "selected_gru_branch": True,
    },
}

# Filename patterns are deliberately explicit.
FILE_PATTERNS = [
    ("activity", re.compile(r"^UoMActivity(?P<pid>\d+)\.csv$", re.IGNORECASE)),
    ("glucose", re.compile(r"^UoMGlucose(?P<pid>\d+)\.csv$", re.IGNORECASE)),
    ("basal_insulin", re.compile(r"^UoMBasal(?P<pid>\d+)\.csv$", re.IGNORECASE)),
    ("bolus_insulin", re.compile(r"^UoMBolus(?P<pid>\d+)\.csv$", re.IGNORECASE)),
    ("nutrition", re.compile(r"^UoMNutrition(?P<pid>\d+)\.csv$", re.IGNORECASE)),
    (
        "sleep_summary",
        re.compile(r"^UoM(?P<pid>\d+)sleeptime\.csv$", re.IGNORECASE),
    ),
    (
        "sleep_timeseries",
        re.compile(r"^UoMsleep(?P<pid>\d+)\.csv$", re.IGNORECASE),
    ),
]

NULL_LIKE_VALUES = {
    "",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "missing",
    "unknown",
    "unk",
    "?",
    "-",
}

# Values that should not automatically be treated as errors.
# The audit reports them explicitly so that a later representation decision
# can be made deliberately.
NULL_LIKE_DISPLAY = {
    "": "<EMPTY>",
    "na": "<NA>",
    "n/a": "<N/A>",
    "nan": "<NAN>",
    "null": "<NULL>",
    "none": "<NONE>",
    "missing": "<MISSING>",
    "unknown": "<UNKNOWN>",
    "unk": "<UNK>",
    "?": "<??>",
    "-": "<DASH>",
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def project_root_from_script() -> Path:
    """
    Resolve project root from:
        <project_root>/src/data/audit_*.py
    """
    script_path = Path(__file__).resolve()

    try:
        root = script_path.parents[2]
    except IndexError:
        root = Path.cwd().resolve()

    return root


def normalize_path_for_display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def print_separator(char: str = "-", width: int = 80) -> None:
    print(char * width)


def print_section(title: str) -> None:
    print()
    print_separator("-")
    print(title)
    print_separator("-")


def fail(message: str, report_path: Path | None = None) -> None:
    print()
    print("=" * 80)
    print("T1D-UOM CATEGORICAL VOCABULARY CONTRACT AUDIT FAILED")
    print("=" * 80)
    print()
    print(message)

    print()
    print("IMPORTANT:")
    print("  No dataset files were modified.")
    print("  No sequence-input files were modified.")
    print("  No categorical encoding was performed.")
    print("  No representation was constructed.")
    print("  No windows were created.")
    print("  No targets were created.")
    print("  No model was trained.")
    print("  Frozen architecture was not changed.")

    if report_path is not None:
        print()
        print("If a report was written, it is located at:")
        print(f"  {report_path}")

    print("=" * 80)
    raise SystemExit(1)


def load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    """
    Read CSV without modifying it.

    Encoding is detected conservatively:
      1. utf-8-sig
      2. utf-8
      3. cp1252

    No values are altered.
    """
    encodings = ["utf-8-sig", "utf-8", "cp1252"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            with path.open(
                "r",
                encoding=encoding,
                newline="",
            ) as f:
                reader = csv.DictReader(f)

                if reader.fieldnames is None:
                    raise ValueError("CSV has no header row.")

                fieldnames = list(reader.fieldnames)
                rows = list(reader)

                return fieldnames, rows, encoding

        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"Unable to read CSV: {path}")


def identify_file_contract(
    path: Path,
) -> tuple[str | None, str | None]:
    """
    Identify modality and participant strictly from the known filename
    conventions.
    """
    name = path.name

    for modality, pattern in FILE_PATTERNS:
        match = pattern.match(name)
        if match:
            pid_digits = match.group("pid")
            return f"UoM{pid_digits}", modality

    return None, None


def discover_sequence_files(sequence_root: Path) -> list[Path]:
    files = sorted(
        [
            p
            for p in sequence_root.rglob("*.csv")
            if p.is_file()
        ],
        key=lambda p: p.relative_to(sequence_root).as_posix().lower(),
    )
    return files


def canonical_value_display(value: str) -> str:
    """
    Display value without changing the source value.

    This function is for REPORTING ONLY.
    """
    lowered = value.strip().lower()

    if lowered in NULL_LIKE_DISPLAY:
        return NULL_LIKE_DISPLAY[lowered]

    return value


def raw_value_flags(value: str) -> dict[str, bool]:
    """
    Analyze a raw categorical value without modifying it.
    """
    stripped = value.strip()
    lowered = stripped.lower()

    return {
        "empty": stripped == "",
        "null_like": lowered in NULL_LIKE_VALUES,
        "leading_or_trailing_whitespace": value != stripped,
        "contains_internal_whitespace": any(ch.isspace() for ch in stripped),
    }


def json_safe_counter(counter: Counter[str]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(
            counter.items(),
            key=lambda item: (-item[1], str(item[0]).lower()),
        )
    }


# =============================================================================
# FILE / INVENTORY AUDIT
# =============================================================================

def audit_inventory(
    root: Path,
    sequence_root: Path,
) -> dict[str, Any]:
    files = discover_sequence_files(sequence_root)

    result: dict[str, Any] = {
        "discovered_file_count": len(files),
        "expected_file_count": EXPECTED_FILE_COUNT,
        "file_count_pass": len(files) == EXPECTED_FILE_COUNT,
        "files": [],
        "unknown_files": [],
        "participants": [],
        "modality_counts": {},
    }

    participant_set: set[str] = set()
    modality_counts: Counter[str] = Counter()

    for path in files:
        relative = normalize_path_for_display(path, sequence_root)
        participant, modality = identify_file_contract(path)

        entry = {
            "relative_path": relative,
            "filename": path.name,
            "participant": participant,
            "modality": modality,
            "identity_status": "PASS"
            if participant is not None and modality is not None
            else "FAIL",
        }

        result["files"].append(entry)

        if participant is None or modality is None:
            result["unknown_files"].append(relative)
        else:
            participant_set.add(participant)
            modality_counts[modality] += 1

    result["participants"] = sorted(participant_set)
    result["expected_participants"] = EXPECTED_PARTICIPANTS
    result["participant_set_pass"] = (
        sorted(participant_set) == sorted(EXPECTED_PARTICIPANTS)
    )

    result["modality_counts"] = dict(sorted(modality_counts.items()))
    result["expected_modality_counts"] = EXPECTED_MODALITY_COUNTS

    result["modality_count_status"] = {
        modality: (
            modality_counts.get(modality, 0)
            == expected_count
        )
        for modality, expected_count in EXPECTED_MODALITY_COUNTS.items()
    }

    return result


# =============================================================================
# CATEGORICAL VOCABULARY AUDIT
# =============================================================================

def audit_categorical_file(
    path: Path,
    sequence_root: Path,
    participant: str,
    modality: str,
) -> dict[str, Any]:
    contract = FILE_CONTRACTS[modality]

    fieldnames, rows, encoding = load_csv_rows(path)

    expected_columns = contract["columns"]
    categorical_columns = contract["categorical"]

    missing_columns = [
        column
        for column in expected_columns
        if column not in fieldnames
    ]

    unexpected_columns = [
        column
        for column in fieldnames
        if column not in expected_columns
    ]

    field_results: dict[str, Any] = {}

    for column in categorical_columns:
        if column not in fieldnames:
            field_results[column] = {
                "status": "SCHEMA_MISSING",
                "raw_value_count": 0,
                "distinct_raw_values": [],
                "normalized_casefold_groups": {},
                "empty_count": 0,
                "null_like_count": 0,
                "whitespace_variant_count": 0,
                "leading_trailing_whitespace_values": [],
                "case_variant_groups": {},
            }
            continue

        raw_counter: Counter[str] = Counter()
        normalized_groups: defaultdict[str, Counter[str]] = defaultdict(Counter)

        empty_count = 0
        null_like_count = 0
        whitespace_variant_count = 0

        whitespace_values: Counter[str] = Counter()

        for row in rows:
            raw_value = row.get(column, "")

            # IMPORTANT:
            # raw_value is used exactly as read.
            raw_counter[raw_value] += 1

            flags = raw_value_flags(raw_value)

            if flags["empty"]:
                empty_count += 1

            if flags["null_like"]:
                null_like_count += 1

            if flags["leading_or_trailing_whitespace"]:
                whitespace_variant_count += 1
                whitespace_values[raw_value] += 1

            normalized_key = raw_value.strip().casefold()
            normalized_groups[normalized_key][raw_value] += 1

        case_variant_groups: dict[str, dict[str, int]] = {}

        for normalized_key, variants in normalized_groups.items():
            if len(variants) > 1:
                display_key = (
                    canonical_value_display(normalized_key)
                    if normalized_key
                    else "<EMPTY>"
                )

                case_variant_groups[display_key] = (
                    json_safe_counter(variants)
                )

        field_results[column] = {
            "status": "PASS",
            "row_count": len(rows),
            "distinct_raw_value_count": len(raw_counter),
            "raw_value_counts": json_safe_counter(raw_counter),
            "distinct_normalized_casefold_count": len(normalized_groups),
            "empty_count": empty_count,
            "null_like_count": null_like_count,
            "whitespace_variant_count": whitespace_variant_count,
            "leading_trailing_whitespace_values": (
                json_safe_counter(whitespace_values)
            ),
            "case_or_whitespace_variant_groups": case_variant_groups,
        }

    schema_status = (
        "PASS"
        if not missing_columns
        else "FAIL"
    )

    relative = normalize_path_for_display(path, sequence_root)

    return {
        "relative_path": relative,
        "participant": participant,
        "modality": modality,
        "encoding_used_for_read": encoding,
        "row_count": len(rows),
        "schema_status": schema_status,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "categorical_fields": field_results,
        "selected_gru_branch": bool(contract["selected_gru_branch"]),
    }


def audit_all_categorical_files(
    sequence_root: Path,
    inventory: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    file_results: list[dict[str, Any]] = []

    for index, entry in enumerate(inventory["files"], start=1):
        relative = entry["relative_path"]

        print(
            f"[{index:03d}/{len(inventory['files']):03d}] "
            f"{relative}",
            end="",
        )

        participant = entry["participant"]
        modality = entry["modality"]

        if participant is None or modality is None:
            print(" | IDENTITY=FAIL")
            continue

        path = sequence_root / Path(relative)

        try:
            result = audit_categorical_file(
                path=path,
                sequence_root=sequence_root,
                participant=participant,
                modality=modality,
            )

            file_results.append(result)

            if result["schema_status"] != "PASS":
                print(" | SCHEMA=FAIL")
            else:
                print(" | SCHEMA=PASS")

        except Exception as exc:
            print(" | READ=FAIL")

            file_results.append(
                {
                    "relative_path": relative,
                    "participant": participant,
                    "modality": modality,
                    "schema_status": "READ_ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                    "categorical_fields": {},
                }
            )

    summary = {
        "files_discovered": len(inventory["files"]),
        "files_audited": len(file_results),
        "schema_failures": sum(
            1
            for x in file_results
            if x.get("schema_status") != "PASS"
        ),
        "read_errors": sum(
            1
            for x in file_results
            if x.get("schema_status") == "READ_ERROR"
        ),
    }

    return file_results, summary


# =============================================================================
# GLOBAL VOCABULARY SUMMARY
# =============================================================================

def build_global_vocabulary(
    file_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build global raw-value observations.

    This does NOT define the final encoding.

    It merely records what exists.
    """
    global_fields: dict[str, dict[str, Any]] = {}

    for file_result in file_results:
        modality = file_result.get("modality")
        participant = file_result.get("participant")

        for field, field_result in file_result.get(
            "categorical_fields", {}
        ).items():

            key = f"{modality}.{field}"

            if key not in global_fields:
                global_fields[key] = {
                    "modality": modality,
                    "field": field,
                    "participants": set(),
                    "file_count": 0,
                    "row_count": 0,
                    "raw_value_counts": Counter(),
                    "empty_count": 0,
                    "null_like_count": 0,
                    "whitespace_variant_count": 0,
                }

            target = global_fields[key]

            target["participants"].add(participant)
            target["file_count"] += 1
            target["row_count"] += field_result.get(
                "row_count",
                0,
            )

            target["raw_value_counts"].update(
                field_result.get(
                    "raw_value_counts",
                    {},
                )
            )

            target["empty_count"] += field_result.get(
                "empty_count",
                0,
            )

            target["null_like_count"] += field_result.get(
                "null_like_count",
                0,
            )

            target["whitespace_variant_count"] += (
                field_result.get(
                    "whitespace_variant_count",
                    0,
                )
            )

    output: dict[str, Any] = {}

    for key, value in sorted(global_fields.items()):
        output[key] = {
            "modality": value["modality"],
            "field": value["field"],
            "participants": sorted(value["participants"]),
            "participant_count": len(value["participants"]),
            "file_count": value["file_count"],
            "row_count": value["row_count"],
            "distinct_raw_value_count": len(
                value["raw_value_counts"]
            ),
            "raw_value_counts": json_safe_counter(
                value["raw_value_counts"]
            ),
            "empty_count": value["empty_count"],
            "null_like_count": value["null_like_count"],
            "whitespace_variant_count": (
                value["whitespace_variant_count"]
            ),
        }

    return output


# =============================================================================
# REPRESENTATION-DECISION BOUNDARY
# =============================================================================

def build_representation_boundary(
    global_vocabulary: dict[str, Any],
) -> dict[str, Any]:
    """
    Explicitly record what this audit DOES and DOES NOT decide.

    This is intentionally conservative.

    The script does NOT choose:
      - integer/ordinal encoding
      - one-hot encoding
      - embedding dimensions
      - learned embeddings
      - category collapsing
      - unknown-category policy

    Those decisions belong to the next controlled stage.
    """

    fields = []

    for key, value in global_vocabulary.items():
        fields.append(
            {
                "field_key": key,
                "modality": value["modality"],
                "field": value["field"],
                "observed_distinct_raw_values": value[
                    "distinct_raw_value_count"
                ],
                "observed_values": value[
                    "raw_value_counts"
                ],
                "representation_status": "NOT_YET_DEFINED",
            }
        )

    return {
        "status": "PENDING_EXPLICIT_REPRESENTATION_DECISION",
        "encoding_performed": False,
        "representation_constructed": False,
        "categorical_fields": fields,
        "decisions_intentionally_not_made": [
            "integer_or_ordinal_encoding",
            "one_hot_encoding",
            "embedding_representation",
            "category_collapsing",
            "unknown_category_policy",
            "missing_category_policy",
        ],
    }


# =============================================================================
# SAFETY / ARCHITECTURE STATUS
# =============================================================================

def architecture_status() -> dict[str, Any]:
    return {
        "frozen_five_gru_architecture": True,
        "branches": [
            "Glucose -> GRU -> zG",
            "Insulin -> GRU -> zI",
            "Nutrition -> GRU -> zN",
            "Activity -> GRU -> zA",
            "Sleep -> GRU -> zS",
        ],
        "gru_branch_count": 5,
        "additional_insulin_gru": False,
        "additional_sleep_gru": False,
        "mlp_fusion_implemented": False,
        "digital_twin_implemented": False,
        "prediction_implemented": False,
        "what_if_implemented": False,
        "interactive_ui_implemented": False,
    }


# =============================================================================
# REPORT
# =============================================================================

def make_report(
    root: Path,
    sequence_root: Path,
    inventory: dict[str, Any],
    file_results: list[dict[str, Any]],
    audit_summary: dict[str, Any],
    global_vocabulary: dict[str, Any],
) -> dict[str, Any]:

    categorical_issue_count = 0
    whitespace_issue_count = 0
    null_like_issue_count = 0

    for result in file_results:
        for field_result in result.get(
            "categorical_fields",
            {},
        ).values():

            null_like_issue_count += int(
                field_result.get(
                    "null_like_count",
                    0,
                )
            )

            whitespace_issue_count += int(
                field_result.get(
                    "whitespace_variant_count",
                    0,
                )
            )

            if (
                field_result.get("empty_count", 0) > 0
                or field_result.get("null_like_count", 0) > 0
                or field_result.get("whitespace_variant_count", 0) > 0
                or field_result.get(
                    "case_or_whitespace_variant_groups",
                    {},
                )
            ):
                categorical_issue_count += 1

    identity_pass = (
        not inventory["unknown_files"]
        and inventory["participant_set_pass"]
        and all(
            inventory["modality_count_status"].values()
        )
    )

    schema_pass = (
        audit_summary["schema_failures"] == 0
        and audit_summary["read_errors"] == 0
    )

    overall_audit_pass = identity_pass and schema_pass

    return {
        "report_metadata": {
            "project": PROJECT_NAME,
            "audit_name": (
                "Categorical Vocabulary / Representation "
                "Contract Audit"
            ),
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "project_root": str(root),
            "sequence_input_directory": str(
                sequence_root
            ),
            "read_only": True,
        },

        "scope": {
            "dataset_modified": False,
            "sequence_inputs_modified": False,
            "values_transformed": False,
            "rows_deleted": False,
            "resampling_performed": False,
            "interpolation_performed": False,
            "imputation_performed": False,
            "normalization_performed": False,
            "feature_engineering_performed": False,
            "categorical_encoding_performed": False,
            "representation_constructed": False,
            "event_type_constructed": False,
            "windows_created": False,
            "targets_created": False,
            "model_trained": False,
            "mlp_fusion_implemented": False,
            "digital_twin_implemented": False,
            "prediction_implemented": False,
            "what_if_implemented": False,
            "interactive_ui_implemented": False,
        },

        "frozen_inventory": inventory,

        "architecture": architecture_status(),

        "categorical_fields_in_scope": {
            "selected_five_gru_branches": {
                "nutrition": [
                    "meal_type",
                    "meal_tag",
                ],
                "activity": [
                    "activity_type",
                    "intensity",
                ],
                "basal_insulin": [
                    "insulin_kind",
                ],
            },
            "sleep_summary_documented_fields": [
                "calendar_date",
                "validation",
            ],
            "sleep_timeseries": [],
            "glucose": [],
            "bolus_insulin": [],
        },

        "file_audits": file_results,

        "global_vocabulary": global_vocabulary,

        "representation_boundary": build_representation_boundary(
            global_vocabulary
        ),

        "summary": {
            **audit_summary,
            "identity_pass": identity_pass,
            "schema_pass": schema_pass,
            "overall_read_only_audit_pass": overall_audit_pass,
            "fields_with_categorical_observations": len(
                global_vocabulary
            ),
            "fields_with_any_review_condition": (
                categorical_issue_count
            ),
            "total_null_like_observations": (
                null_like_issue_count
            ),
            "total_whitespace_variant_observations": (
                whitespace_issue_count
            ),
        },

        "interpretation": {
            "audit_result": (
                "PASS"
                if overall_audit_pass
                else "FAIL"
            ),
            "meaning": (
                "The frozen files can be identified and their "
                "categorical/raw domains can be documented. "
                "This audit does not select or perform a "
                "categorical encoding method."
            ),
            "next_stage": (
                "Freeze the explicit model-ready categorical "
                "representation contract before implementing "
                "real-data five-GRU execution."
            ),
        },
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    root = project_root_from_script()

    sequence_root = (
        root
        / "data"
        / "derived"
        / "t1d_uom_v1.0.3_sequence_inputs"
    )

    report_dir = (
        root
        / "reports"
        / "data_quality"
    )

    report_path = (
        report_dir
        / "categorical_vocabulary_contract_audit.json"
    )

    print("=" * 80)
    print("T1D-UOM CATEGORICAL VOCABULARY / REPRESENTATION CONTRACT AUDIT")
    print("=" * 80)

    print("IMPORTANT: READ-ONLY.")
    print("No dataset files will be modified.")
    print("No sequence-input files will be modified.")
    print("No values will be transformed.")
    print("No rows will be deleted.")
    print("No resampling will be performed.")
    print("No interpolation will be performed.")
    print("No imputation will be performed.")
    print("No normalization will be performed.")
    print("No feature engineering will be performed.")
    print("No categorical encoding will be performed.")
    print("No event_type will be constructed.")
    print("No windows will be created.")
    print("No targets will be created.")
    print("No model will be trained.")
    print("No MLP Fusion will be implemented.")
    print("No Digital Twin will be implemented.")
    print("No Prediction will be implemented.")
    print("No What-if implementation will be created.")
    print("No Interactive UI will be created.")
    print()

    print(f"Project root:       {root}")
    print(f"Sequence inputs:    {sequence_root}")
    print(f"Audit report:       {report_path}")

    # ---------------------------------------------------------------------
    # 1. DIRECTORY VALIDATION
    # ---------------------------------------------------------------------

    print_section("1. DIRECTORY VALIDATION")

    if not root.is_dir():
        fail(
            f"Project root does not exist:\n{root}",
            report_path,
        )

    if not sequence_root.is_dir():
        fail(
            f"Sequence-input directory not found:\n"
            f"{sequence_root}",
            report_path,
        )

    print("Project root: PASS")
    print("Sequence-input directory: PASS")

    # ---------------------------------------------------------------------
    # 2. INVENTORY
    # ---------------------------------------------------------------------

    print_section("2. FROZEN SEQUENCE-INPUT INVENTORY")

    inventory = audit_inventory(
        root=root,
        sequence_root=sequence_root,
    )

    print(
        "Sequence-input CSV files discovered: "
        f"{inventory['discovered_file_count']}"
    )

    if not inventory["file_count_pass"]:
        fail(
            "Frozen sequence-input file count mismatch.\n"
            f"Expected: {EXPECTED_FILE_COUNT}\n"
            f"Found:    {inventory['discovered_file_count']}",
            report_path,
        )

    print("Frozen sequence-input file count: PASS")

    if inventory["unknown_files"]:
        print()
        print("Unable to identify file(s):")
        for item in inventory["unknown_files"]:
            print(f"  - {item}")

        fail(
            "One or more sequence-input files could not be "
            "identified using the frozen filename contract.",
            report_path,
        )

    # ---------------------------------------------------------------------
    # 3. COHORT
    # ---------------------------------------------------------------------

    print_section("3. FROZEN COHORT VALIDATION")

    print(
        "Participants represented: "
        f"{len(inventory['participants'])}"
    )

    for participant in inventory["participants"]:
        print(f"  {participant}")

    if not inventory["participant_set_pass"]:
        missing = sorted(
            set(EXPECTED_PARTICIPANTS)
            - set(inventory["participants"])
        )

        unexpected = sorted(
            set(inventory["participants"])
            - set(EXPECTED_PARTICIPANTS)
        )

        if missing:
            print("Missing frozen participants:")
            for participant in missing:
                print(f"  - {participant}")

        if unexpected:
            print("Unexpected participants:")
            for participant in unexpected:
                print(f"  - {participant}")

        fail(
            "Frozen participant cohort validation failed.",
            report_path,
        )

    print("Frozen 13-participant cohort: PASS")

    # ---------------------------------------------------------------------
    # 4. MODALITY COUNTS
    # ---------------------------------------------------------------------

    print_section("4. MODALITY FAMILY VALIDATION")

    modality_pass = True

    for modality, expected in EXPECTED_MODALITY_COUNTS.items():
        actual = inventory["modality_counts"].get(
            modality,
            0,
        )

        status = "PASS" if actual == expected else "FAIL"

        print(
            f"{modality:<18} "
            f"files={actual:<3} "
            f"expected={expected:<3} "
            f"{status}"
        )

        if status == "FAIL":
            modality_pass = False

    if not modality_pass:
        fail(
            "Frozen modality family counts do not match "
            "the established inventory contract.",
            report_path,
        )

    print("Modality family counts: PASS")

    # ---------------------------------------------------------------------
    # 5. ARCHITECTURE
    # ---------------------------------------------------------------------

    print_section("5. FROZEN FIVE-GRU ARCHITECTURE")

    print("Glucose    -> GRU -> zG")
    print("Insulin    -> GRU -> zI")
    print("Nutrition  -> GRU -> zN")
    print("Activity   -> GRU -> zA")
    print("Sleep      -> GRU -> zS")
    print("zG,zI,zN,zA,zS -> MLP Fusion")
    print("MLP Fusion -> Unified Patient State")
    print("Unified Patient State -> DIGITAL TWIN")
    print("DIGITAL TWIN -> Prediction / What-if")
    print("Prediction / What-if -> Interactive UI")

    print()
    print("This audit implements NONE of the above model components.")

    # ---------------------------------------------------------------------
    # 6. CATEGORICAL FIELD CONTRACT
    # ---------------------------------------------------------------------

    print_section("6. CATEGORICAL FIELDS IN SCOPE")

    print("Selected five-GRU branches:")

    print("  Nutrition:")
    print("    - meal_type")
    print("    - meal_tag")

    print("  Activity:")
    print("    - activity_type")
    print("    - intensity")

    print("  Insulin:")
    print("    - insulin_kind")

    print()
    print("Documented Sleep-summary categorical/raw fields:")
    print("    - calendar_date")
    print("    - validation")

    print()
    print("Glucose categorical fields: NONE")
    print("Bolus-insulin categorical fields: NONE")
    print("Sleep-time-series categorical fields: NONE")

    # ---------------------------------------------------------------------
    # 7. ACTUAL VOCABULARY AUDIT
    # ---------------------------------------------------------------------

    print_section("7. COMPLETE 86-FILE CATEGORICAL VOCABULARY AUDIT")

    file_results, audit_summary = audit_all_categorical_files(
        sequence_root=sequence_root,
        inventory=inventory,
    )

    # ---------------------------------------------------------------------
    # 8. GLOBAL VOCABULARY
    # ---------------------------------------------------------------------

    print_section("8. GLOBAL OBSERVED CATEGORICAL VOCABULARIES")

    global_vocabulary = build_global_vocabulary(
        file_results
    )

    if not global_vocabulary:
        print("No categorical fields were discovered.")
    else:
        for key, value in global_vocabulary.items():
            print()
            print(
                f"{key} "
                f"(files={value['file_count']}, "
                f"participants={value['participant_count']})"
            )

            print(
                "  Distinct raw values: "
                f"{value['distinct_raw_value_count']}"
            )

            for raw_value, count in value[
                "raw_value_counts"
            ].items():
                print(
                    f"    {canonical_value_display(raw_value)!r}"
                    f" : {count}"
                )

            if value["empty_count"] > 0:
                print(
                    "  Empty observations: "
                    f"{value['empty_count']}"
                )

            if value["null_like_count"] > 0:
                print(
                    "  Null-like observations: "
                    f"{value['null_like_count']}"
                )

            if value["whitespace_variant_count"] > 0:
                print(
                    "  Leading/trailing whitespace "
                    "observations: "
                    f"{value['whitespace_variant_count']}"
                )

    # ---------------------------------------------------------------------
    # 9. REPORT
    # ---------------------------------------------------------------------

    report = make_report(
        root=root,
        sequence_root=sequence_root,
        inventory=inventory,
        file_results=file_results,
        audit_summary=audit_summary,
        global_vocabulary=global_vocabulary,
    )

    print_section("9. AUDIT SUMMARY")

    print(
        f"Files discovered:                 "
        f"{audit_summary['files_discovered']}"
    )

    print(
        f"Files audited:                    "
        f"{audit_summary['files_audited']}"
    )

    print(
        f"Schema failures:                  "
        f"{audit_summary['schema_failures']}"
    )

    print(
        f"Read errors:                      "
        f"{audit_summary['read_errors']}"
    )

    print(
        "Identity/cohort/modality status:   "
        f"{'PASS' if inventory['participant_set_pass'] else 'FAIL'}"
    )

    print(
        "Schema status:                    "
        f"{'PASS' if audit_summary['schema_failures'] == 0 else 'FAIL'}"
    )

    print(
        "Categorical encoding performed:   NO"
    )

    print(
        "Representation constructed:      NO"
    )

    # ---------------------------------------------------------------------
    # 10. WRITE JSON REPORT
    # ---------------------------------------------------------------------

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    print()
    print("Audit report saved:")
    print(f"  {report_path}")

    # ---------------------------------------------------------------------
    # 11. FINAL STATUS
    # ---------------------------------------------------------------------

    overall_pass = report["summary"][
        "overall_read_only_audit_pass"
    ]

    print()
    print("=" * 80)

    if overall_pass:
        print(
            "T1D-UOM CATEGORICAL VOCABULARY / "
            "REPRESENTATION CONTRACT AUDIT COMPLETE"
        )
        print("=" * 80)
        print()
        print("FINAL RESULT: PASS")
        print()
        print(
            "All frozen sequence-input files were successfully "
            "identified and audited for categorical vocabulary."
        )
        print(
            f"Files audited: "
            f"{audit_summary['files_audited']}/"
            f"{EXPECTED_FILE_COUNT}"
        )
        print("Frozen 13-participant cohort: PASS")
        print("Modality family counts: PASS")
        print("Categorical-field schema: PASS")
        print()
        print(
            "IMPORTANT: PASS here means the categorical/raw "
            "domains were successfully inspected and documented."
        )
        print(
            "It does NOT mean that a categorical encoding method "
            "has been selected."
        )
        print()
        print(
            "Categorical encoding: NOT PERFORMED"
        )
        print(
            "Model-ready representation: NOT CONSTRUCTED"
        )
        print(
            "Five GRUs: NOT IMPLEMENTED BY THIS AUDIT"
        )
        print(
            "MLP Fusion: NOT IMPLEMENTED"
        )
        print(
            "Digital Twin: NOT IMPLEMENTED"
        )
        print(
            "Prediction / What-if: NOT IMPLEMENTED"
        )
        print(
            "Interactive UI: NOT IMPLEMENTED"
        )
        print()
        print(
            "NEXT CONTROLLED STAGE:"
        )
        print(
            "Freeze the explicit categorical representation "
            "contract using the observed vocabulary in this report."
        )
        print()
        print(
            "No dataset files were modified."
        )
        print(
            "No sequence-input files were modified."
        )
        print("=" * 80)

        # PASS even if informational null-like/whitespace observations
        # exist. They are deliberately reported, not silently changed.
        raise SystemExit(0)

    print(
        "T1D-UOM CATEGORICAL VOCABULARY / "
        "REPRESENTATION CONTRACT AUDIT FAILED"
    )
    print("=" * 80)
    print()
    print(
        "A genuine inventory, identity, read, or schema problem "
        "was detected."
    )
    print()
    print("Inspect the JSON report for exact file-level details.")
    print()
    print("No dataset files were modified.")
    print("No sequence-input files were modified.")
    print("No categorical encoding was performed.")
    print("No representation was constructed.")
    print("No model was trained.")
    print("Frozen architecture was not changed.")
    print("=" * 80)

    raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("AUDIT INTERRUPTED BY USER.")
        print("No dataset or sequence-input files were modified.")
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as exc:
        print()
        print("=" * 80)
        print("UNEXPECTED AUDIT ERROR")
        print("=" * 80)
        print(f"{type(exc).__name__}: {exc}")
        print()
        print("No dataset or sequence-input files were modified.")
        print("No categorical encoding was performed.")
        print("No representation was constructed.")
        print("No model was trained.")
        print("=" * 80)
        raise SystemExit(1)