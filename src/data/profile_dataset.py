from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# =============================================================================
# T1D-UOM VERIFIED DATASET PROFILE
# =============================================================================
#
# READ-ONLY DATASET PROFILER
#
# This script:
#   - reads the verified derived dataset
#   - profiles files, rows, columns, missingness and duplicates
#   - validates modality/file-family consistency
#   - validates participant mapping
#   - validates timestamp quality
#   - verifies the two already-approved timestamp corrections
#
# This script DOES NOT:
#   - modify raw data
#   - modify derived data
#   - rename columns
#   - remove duplicates
#   - impute missing values
#   - normalize values
#   - resample data
#   - create windows
#   - create model features
#   - train models
#
# FROZEN ARCHITECTURE:
#
#   Glucose   -> GRU
#   Insulin   -> GRU
#   Nutrition -> GRU
#   Activity  -> GRU
#   Sleep     -> GRU
#                    |
#                    v
#              MLP Fusion
#
# =============================================================================


# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DERIVED_DATASET = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "t1d_uom_v1.0.3_timestamp_corrected"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
)

REPORT_PATH = (
    REPORT_DIR
    / "dataset_profile.json"
)

TIMESTAMP_MANIFEST = (
    REPORT_DIR
    / "timestamp_corrections.json"
)


# -----------------------------------------------------------------------------
# FROZEN PROJECT EXPECTATIONS
# -----------------------------------------------------------------------------

EXPECTED_CSV_COUNT = 112

PROJECT_MODALITIES = (
    "glucose",
    "insulin",
    "nutrition",
    "activity",
    "sleep",
)

EXPECTED_FILE_COUNTS = {
    "activity": 17,
    "glucose": 17,
    "basal_insulin": 14,
    "bolus_insulin": 16,
    "nutrition": 15,
    "sleep_summary": 18,
    "sleep_timeseries": 15,
}


FROZEN_FIVE_MODALITY_COHORT = [
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


FROZEN_FULL_CORE_COHORT = [
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
]


# -----------------------------------------------------------------------------
# APPROVED TIMESTAMP CORRECTIONS
# -----------------------------------------------------------------------------

APPROVED_TIMESTAMP_CORRECTIONS = [
    {
        "relative_path": "Nutrition Data/UoMNutrition2320.csv",
        "column": "meal_ts",
        "original": "02/12/2033 20:00",
        "corrected": "02/12/2023 20:00",
    },
    {
        "relative_path": "Nutrition Data/UoMNutrition2404.csv",
        "column": "meal_ts",
        "original": "22/04/2204 11:45",
        "corrected": "22/04/2024 11:45",
    },
]


# -----------------------------------------------------------------------------
# TIMESTAMP COLUMNS
# -----------------------------------------------------------------------------
#
# These are based on the ACTUAL dataset column naming convention observed
# in the verified dataset.
# -----------------------------------------------------------------------------

TIMESTAMP_COLUMNS = {
    "activity": (
        "activity_ts",
        "start_time_s",
    ),
    "glucose": (
        "bg_ts",
    ),
    "basal_insulin": (
        "basal_ts",
    ),
    "bolus_insulin": (
        "bolus_ts",
    ),
    "nutrition": (
        "meal_ts",
    ),
    "sleep_summary": (
        "start_date_ts",
    ),
    "sleep_timeseries": (
        "sleep_ts",
    ),
}


MIN_REASONABLE_YEAR = 2000
MAX_REASONABLE_YEAR = 2030


# -----------------------------------------------------------------------------
# BASIC UTILITIES
# -----------------------------------------------------------------------------

def fail(message: str) -> None:
    print()
    print("=" * 80)
    print("PROFILE FAILED")
    print("=" * 80)
    print(message)
    print()
    print("No dataset files were modified.")
    print("=" * 80)
    sys.exit(1)


def relative_path(path: Path) -> str:
    return path.relative_to(DERIVED_DATASET).as_posix()


def read_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.reader(handle)

        try:
            header = next(reader)
        except StopIteration:
            return [], []

        rows = list(reader)

    return (
        [str(x).strip() for x in header],
        rows,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def row_key(row: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(x) for x in row)


def duplicate_statistics(
    rows: Sequence[Sequence[str]],
) -> Tuple[int, int]:

    counts = Counter(
        row_key(row)
        for row in rows
    )

    duplicate_rows = sum(
        count
        for count in counts.values()
        if count > 1
    )

    duplicate_extra = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return (
        duplicate_rows,
        duplicate_extra,
    )


def count_missing_cells(
    rows: Sequence[Sequence[str]],
    column_count: int,
) -> int:

    missing = 0

    for row in rows:

        for index in range(column_count):

            value = (
                row[index]
                if index < len(row)
                else ""
            )

            if value.strip() == "":
                missing += 1

    return missing


# -----------------------------------------------------------------------------
# MODALITY IDENTIFICATION
# -----------------------------------------------------------------------------

def identify_modality(path: Path) -> Optional[str]:

    rel = relative_path(path)
    name = path.name

    if rel.startswith("Activity Data/"):
        return "activity"

    if rel.startswith("Glucose Data/"):
        return "glucose"

    if rel.startswith("Insulin Data/Basal Data/"):
        return "basal_insulin"

    if rel.startswith("Insulin Data/Bolus Data/"):
        return "bolus_insulin"

    if rel.startswith("Nutrition Data/"):
        return "nutrition"

    if rel.startswith("Sleep Data/"):

        # IMPORTANT:
        # sleeptime files are the sleep-summary family.
        if (
            name.startswith("UoM")
            and "sleeptime" in name.lower()
        ):
            return "sleep_summary"

        # UoMsleep files are the sleep-timeseries family.
        if name.startswith("UoMsleep"):
            return "sleep_timeseries"

    return None


# -----------------------------------------------------------------------------
# PARTICIPANT IDENTIFICATION
# -----------------------------------------------------------------------------

def identify_participant(path: Path) -> Optional[str]:

    match = re.search(
        r"(23\d{2}|24\d{2})",
        path.stem,
    )

    if not match:
        return None

    return f"UoM{match.group(1)}"


# -----------------------------------------------------------------------------
# SCHEMA FAMILY VALIDATION
# -----------------------------------------------------------------------------
#
# IMPORTANT:
#
# We DO NOT invent a canonical schema.
#
# Instead:
#   1. Every file is assigned to a known file family.
#   2. Headers are collected from all files in that family.
#   3. The dominant header becomes the family reference header.
#   4. Files differing from that reference are reported.
#
# This allows legitimate real-world columns to remain untouched.
# -----------------------------------------------------------------------------

def header_signature(header: Sequence[str]) -> Tuple[str, ...]:
    return tuple(header)


def validate_family_schemas(
    file_profiles: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:

    families: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for item in file_profiles:

        modality = item["modality"]

        if modality is not None:
            families[modality].append(item)

    family_results: Dict[str, Any] = {}

    for modality in sorted(families):

        items = families[modality]

        signatures = Counter(
            header_signature(item["header"])
            for item in items
        )

        dominant_header, dominant_count = (
            signatures.most_common(1)[0]
        )

        invalid_files = []

        for item in items:

            signature = header_signature(
                item["header"]
            )

            if signature != dominant_header:

                invalid_files.append(
                    {
                        "relative_path": item[
                            "relative_path"
                        ],
                        "header": item["header"],
                        "expected_reference_header": list(
                            dominant_header
                        ),
                        "missing_from_file": sorted(
                            set(dominant_header)
                            - set(item["header"])
                        ),
                        "additional_in_file": sorted(
                            set(item["header"])
                            - set(dominant_header)
                        ),
                    }
                )

        family_results[modality] = {
            "file_count": len(items),
            "distinct_header_signatures": len(
                signatures
            ),
            "reference_header": list(
                dominant_header
            ),
            "reference_header_file_count": (
                dominant_count
            ),
            "schema_valid_count": (
                len(items) - len(invalid_files)
            ),
            "schema_invalid_count": len(
                invalid_files
            ),
            "schema_invalid_files": invalid_files,
        }

    return family_results


# -----------------------------------------------------------------------------
# TIMESTAMP QUALITY
# -----------------------------------------------------------------------------

def extract_year(value: str) -> Optional[int]:

    value = value.strip()

    if not value:
        return None

    match = re.match(
        r"^\d{2}/\d{2}/(\d{4})",
        value,
    )

    if match:
        return int(match.group(1))

    match = re.match(
        r"^(\d{4})[-/]\d{2}[-/]\d{2}",
        value,
    )

    if match:
        return int(match.group(1))

    return None


def inspect_timestamps(
    modality: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> Dict[str, Any]:

    candidates = [
        column
        for column in TIMESTAMP_COLUMNS.get(
            modality,
            (),
        )
        if column in header
    ]

    result = {
        "columns": candidates,
        "parse_failures": 0,
        "suspicious_values": 0,
        "examples": [],
    }

    for column in candidates:

        index = header.index(column)

        for row_number, row in enumerate(
            rows,
            start=2,
        ):

            if index >= len(row):
                continue

            value = row[index].strip()

            if not value:
                continue

            year = extract_year(value)

            if year is None:
                # Only flag values that look like date/timestamp strings.
                if any(
                    character in value
                    for character in (
                        "/",
                        "-",
                        ":",
                    )
                ):
                    result[
                        "parse_failures"
                    ] += 1

                continue

            if not (
                MIN_REASONABLE_YEAR
                <= year
                <= MAX_REASONABLE_YEAR
            ):

                result[
                    "suspicious_values"
                ] += 1

                if len(
                    result["examples"]
                ) < 20:

                    result["examples"].append(
                        {
                            "row": row_number,
                            "column": column,
                            "value": value,
                            "year": year,
                        }
                    )

    return result


# -----------------------------------------------------------------------------
# TIMESTAMP CORRECTION VALIDATION
# -----------------------------------------------------------------------------

def load_manifest() -> Optional[Dict[str, Any]]:

    if not TIMESTAMP_MANIFEST.exists():
        return None

    try:

        with TIMESTAMP_MANIFEST.open(
            "r",
            encoding="utf-8",
        ) as handle:

            return json.load(handle)

    except (
        OSError,
        json.JSONDecodeError,
    ):

        return None


def verify_approved_corrections(
    file_profiles: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:

    profile_map = {
        item["relative_path"]: item
        for item in file_profiles
    }

    results = []

    for correction in (
        APPROVED_TIMESTAMP_CORRECTIONS
    ):

        rel = correction[
            "relative_path"
        ]

        path = DERIVED_DATASET / Path(rel)

        result = {
            **correction,
            "status": "REVIEW",
            "original_count": 0,
            "corrected_count": 0,
        }

        if not path.exists():

            result["status"] = "FAIL"
            result["reason"] = (
                "file does not exist"
            )

            results.append(result)
            continue

        try:

            header, rows = read_csv(path)

        except Exception as exc:

            result["status"] = "FAIL"
            result["reason"] = str(exc)

            results.append(result)
            continue

        column = correction["column"]

        if column not in header:

            result["status"] = "FAIL"
            result["reason"] = (
                f"column {column!r} not found"
            )

            results.append(result)
            continue

        index = header.index(column)

        original = correction[
            "original"
        ]

        corrected = correction[
            "corrected"
        ]

        for row in rows:

            if index >= len(row):
                continue

            value = row[index].strip()

            if value == original:
                result[
                    "original_count"
                ] += 1

            if value == corrected:
                result[
                    "corrected_count"
                ] += 1

        if (
            result["original_count"] == 0
            and result["corrected_count"] >= 1
        ):
            result["status"] = "PASS"

        results.append(result)

    return {
        "manifest_exists": (
            TIMESTAMP_MANIFEST.exists()
        ),
        "results": results,
    }


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main() -> None:

    print("=" * 80)
    print("T1D-UOM VERIFIED DATASET PROFILE")
    print("=" * 80)
    print()
    print("IMPORTANT: READ-ONLY.")
    print(
        "Raw and derived datasets will NOT be modified."
    )
    print()
    print(
        f"Project root:    {PROJECT_ROOT}"
    )
    print(
        f"Derived dataset: {DERIVED_DATASET}"
    )
    print(
        f"Report:          {REPORT_PATH}"
    )
    print()

    if not DERIVED_DATASET.exists():

        fail(
            "Verified derived dataset does not exist:\n"
            f"{DERIVED_DATASET}"
        )

    # -------------------------------------------------------------------------
    # DISCOVERY
    # -------------------------------------------------------------------------

    csv_files = sorted(
        DERIVED_DATASET.rglob("*.csv"),
        key=lambda path: relative_path(path),
    )

    print("-" * 80)
    print("1. STRUCTURE")
    print("-" * 80)

    print(
        f"CSV files discovered: {len(csv_files)}"
    )

    print(
        f"Expected CSV files:   {EXPECTED_CSV_COUNT}"
    )

    # -------------------------------------------------------------------------
    # FILE PROFILING
    # -------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("2. FILE-LEVEL PROFILING")
    print("-" * 80)

    file_profiles = []

    failed_files = []

    total_rows = 0
    total_cells = 0
    total_missing = 0

    duplicate_rows_total = 0
    duplicate_extra_total = 0

    duplicate_files = []

    participant_modalities = defaultdict(set)

    modality_files = Counter()
    modality_rows = Counter()
    modality_participants = defaultdict(set)

    unknown_modality_files = []
    unknown_participant_files = []

    for number, path in enumerate(
        csv_files,
        start=1,
    ):

        rel = relative_path(path)

        print(
            f"[{number:03d}/{len(csv_files)}] "
            f"{rel}"
        )

        modality = identify_modality(path)
        participant = identify_participant(path)

        if modality is None:
            unknown_modality_files.append(
                rel
            )

        if participant is None:
            unknown_participant_files.append(
                rel
            )

        try:

            header, rows = read_csv(path)

        except Exception as exc:

            failed_files.append(
                {
                    "relative_path": rel,
                    "error": str(exc),
                }
            )

            continue

        row_count = len(rows)
        column_count = len(header)

        missing_cells = count_missing_cells(
            rows,
            column_count,
        )

        duplicate_rows, duplicate_extra = (
            duplicate_statistics(rows)
        )

        timestamp_quality = {
            "columns": [],
            "parse_failures": 0,
            "suspicious_values": 0,
            "examples": [],
        }

        if modality is not None:

            timestamp_quality = (
                inspect_timestamps(
                    modality,
                    header,
                    rows,
                )
            )

        total_rows += row_count

        total_cells += (
            row_count
            * column_count
        )

        total_missing += missing_cells

        duplicate_rows_total += (
            duplicate_rows
        )

        duplicate_extra_total += (
            duplicate_extra
        )

        if duplicate_extra > 0:
            duplicate_files.append(rel)

        if modality is not None:

            modality_files[modality] += 1

            modality_rows[modality] += (
                row_count
            )

            if participant is not None:

                modality_participants[
                    modality
                ].add(participant)

                project_modality = modality

                if modality in {
                    "basal_insulin",
                    "bolus_insulin",
                }:
                    project_modality = "insulin"

                elif modality in {
                    "sleep_summary",
                    "sleep_timeseries",
                }:
                    project_modality = "sleep"

                participant_modalities[
                    participant
                ].add(project_modality)

        file_profiles.append(
            {
                "relative_path": rel,
                "filename": path.name,
                "participant": participant,
                "modality": modality,
                "rows": row_count,
                "columns": column_count,
                "header": header,
                "missing_cells": missing_cells,
                "duplicate_rows": duplicate_rows,
                "duplicate_extra": duplicate_extra,
                "timestamp": timestamp_quality,
                "sha256": sha256_file(path),
            }
        )

    # -------------------------------------------------------------------------
    # FAMILY SCHEMA VALIDATION
    # -------------------------------------------------------------------------

    family_schema_results = (
        validate_family_schemas(
            file_profiles
        )
    )

    # -------------------------------------------------------------------------
    # PARTICIPANTS
    # -------------------------------------------------------------------------

    participants = sorted(
        {
            item["participant"]
            for item in file_profiles
            if item["participant"]
        }
    )

    computed_five_modality = sorted(
        participant
        for participant in participants
        if set(PROJECT_MODALITIES).issubset(
            participant_modalities[
                participant
            ]
        )
    )

    # -------------------------------------------------------------------------
    # TIMESTAMP TOTALS
    # -------------------------------------------------------------------------

    timestamp_parse_failures = sum(
        item["timestamp"][
            "parse_failures"
        ]
        for item in file_profiles
    )

    suspicious_timestamp_values = sum(
        item["timestamp"][
            "suspicious_values"
        ]
        for item in file_profiles
    )

    suspicious_timestamp_files = [
        item["relative_path"]
        for item in file_profiles
        if item["timestamp"][
            "suspicious_values"
        ] > 0
    ]

    timestamp_issue_files = [
        item["relative_path"]
        for item in file_profiles
        if item["timestamp"][
            "parse_failures"
        ] > 0
    ]

    # -------------------------------------------------------------------------
    # TIMESTAMP CORRECTIONS
    # -------------------------------------------------------------------------

    correction_results = (
        verify_approved_corrections(
            file_profiles
        )
    )

    manifest = load_manifest()

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    missingness = (
        (
            total_missing
            / total_cells
        )
        * 100
        if total_cells
        else 0.0
    )

    print()
    print("=" * 80)
    print("PROFILE COMPLETE")
    print("=" * 80)
    print()

    print(
        f"CSV files:       "
        f"{len(csv_files)} / "
        f"{EXPECTED_CSV_COUNT}"
    )

    print(
        f"Successfully inspected: "
        f"{len(file_profiles)}"
    )

    print(
        f"Failed files:    "
        f"{len(failed_files)}"
    )

    print(
        f"Participants:    "
        f"{len(participants)}"
    )

    print(
        f"Total rows:      "
        f"{total_rows:,}"
    )

    print(
        f"Total cells:     "
        f"{total_cells:,}"
    )

    print(
        f"Missing cells:   "
        f"{total_missing:,}"
    )

    print(
        f"Missingness:     "
        f"{missingness:.4f}%"
    )

    print(
        f"Duplicate rows:  "
        f"{duplicate_rows_total:,}"
    )

    print(
        f"Duplicate extra: "
        f"{duplicate_extra_total:,}"
    )

    # -------------------------------------------------------------------------
    # MODALITY SUMMARY
    # -------------------------------------------------------------------------

    print()
    print("Modality summary:")
    print("-" * 80)

    for modality in (
        "activity",
        "basal_insulin",
        "bolus_insulin",
        "glucose",
        "nutrition",
        "sleep_summary",
        "sleep_timeseries",
    ):

        family = family_schema_results.get(
            modality,
            {},
        )

        print(
            f"{modality:<18}"
            f"files={modality_files[modality]:<3} "
            f"participants="
            f"{len(modality_participants[modality]):<3} "
            f"rows="
            f"{modality_rows[modality]:>9,} "
            f"schema-valid="
            f"{family.get('schema_valid_count', 0):<3} "
            f"schema-invalid="
            f"{family.get('schema_invalid_count', 0):<3}"
        )

    # -------------------------------------------------------------------------
    # PROJECT COVERAGE
    # -------------------------------------------------------------------------

    print()
    print("Five-modality project coverage:")
    print("-" * 80)

    for participant in participants:

        modalities = [
            modality
            for modality in PROJECT_MODALITIES
            if modality
            in participant_modalities[
                participant
            ]
        ]

        print(
            f"{participant}: "
            + ", ".join(modalities)
        )

    # -------------------------------------------------------------------------
    # FROZEN COHORTS
    # -------------------------------------------------------------------------

    print()
    print("Frozen five-project-modality cohort:")
    print("-" * 80)

    for participant in (
        FROZEN_FIVE_MODALITY_COHORT
    ):
        print(
            f"  - {participant}"
        )

    print()
    print("Frozen full-core cohort:")
    print("-" * 80)

    for participant in (
        FROZEN_FULL_CORE_COHORT
    ):
        print(
            f"  - {participant}"
        )

    # -------------------------------------------------------------------------
    # TIMESTAMP QUALITY
    # -------------------------------------------------------------------------

    print()
    print("Timestamp quality:")
    print("-" * 80)

    print(
        f"Timestamp parse failures: "
        f"{timestamp_parse_failures}"
    )

    print(
        f"Suspicious timestamp values: "
        f"{suspicious_timestamp_values}"
    )

    print(
        f"Files with timestamp issues: "
        f"{len(timestamp_issue_files)}"
    )

    print(
        f"Files with suspicious timestamps: "
        f"{len(suspicious_timestamp_files)}"
    )

    if suspicious_timestamp_files:

        print()
        print(
            "Suspicious-timestamp files:"
        )

        for rel in suspicious_timestamp_files:

            print(
                f"  - {rel}"
            )

    # -------------------------------------------------------------------------
    # APPROVED CORRECTIONS
    # -------------------------------------------------------------------------

    print()
    print(
        "Approved timestamp correction reconciliation:"
    )
    print("-" * 80)

    if manifest is not None:
        print(
            "Timestamp correction manifest: "
            "readable"
        )
    else:
        print(
            "Timestamp correction manifest: "
            "not readable/found"
        )

    for result in correction_results[
        "results"
    ]:

        print(
            f"{result['relative_path']}: "
            f"{result['status']}"
        )

        print(
            f"    original value count: "
            f"{result['original_count']}"
        )

        print(
            f"    corrected value count: "
            f"{result['corrected_count']}"
        )

    # -------------------------------------------------------------------------
    # SCHEMA REVIEW
    # -------------------------------------------------------------------------

    print()
    print("Schema-family validation:")
    print("-" * 80)

    any_schema_invalid = False

    for modality in sorted(
        family_schema_results
    ):

        family = family_schema_results[
            modality
        ]

        print(
            f"{modality}: "
            f"{family['schema_valid_count']} valid, "
            f"{family['schema_invalid_count']} invalid, "
            f"{family['distinct_header_signatures']} "
            f"distinct header signature(s)"
        )

        if family[
            "schema_invalid_count"
        ]:

            any_schema_invalid = True

            for item in family[
                "schema_invalid_files"
            ]:

                print(
                    f"  - "
                    f"{item['relative_path']}"
                )

                if item[
                    "missing_from_file"
                ]:

                    print(
                        "      missing from "
                        "reference header: "
                        + ", ".join(
                            item[
                                "missing_from_file"
                            ]
                        )
                    )

                if item[
                    "additional_in_file"
                ]:

                    print(
                        "      additional columns: "
                        + ", ".join(
                            item[
                                "additional_in_file"
                            ]
                        )
                    )

    # -------------------------------------------------------------------------
    # UNKNOWN FILES
    # -------------------------------------------------------------------------

    print()
    print("Unknown participant files:")
    print("-" * 80)

    if unknown_participant_files:

        for rel in unknown_participant_files:
            print(f"  - {rel}")

    else:
        print("  None")

    print()
    print("Unknown modality files:")
    print("-" * 80)

    if unknown_modality_files:

        for rel in unknown_modality_files:
            print(f"  - {rel}")

    else:
        print("  None")

    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    report = {
        "report_name": (
            "T1D-UOM Verified Dataset Profile"
        ),
        "profile_version": "3.0",
        "read_only": True,
        "project_root": str(PROJECT_ROOT),
        "derived_dataset": str(
            DERIVED_DATASET
        ),
        "expected_csv_count": (
            EXPECTED_CSV_COUNT
        ),
        "csv_files_discovered": len(
            csv_files
        ),
        "successfully_inspected": len(
            file_profiles
        ),
        "failed_files": failed_files,
        "participants": participants,
        "participant_count": len(
            participants
        ),
        "total_rows": total_rows,
        "total_cells": total_cells,
        "missing_cells": total_missing,
        "missingness_percent": round(
            missingness,
            6,
        ),
        "duplicate_rows": (
            duplicate_rows_total
        ),
        "duplicate_extra": (
            duplicate_extra_total
        ),
        "duplicate_containing_files": (
            duplicate_files
        ),
        "modality_summary": {
            modality: {
                "files": modality_files[
                    modality
                ],
                "expected_files": (
                    EXPECTED_FILE_COUNTS[
                        modality
                    ]
                ),
                "participants": sorted(
                    modality_participants[
                        modality
                    ]
                ),
                "participant_count": len(
                    modality_participants[
                        modality
                    ]
                ),
                "rows": modality_rows[
                    modality
                ],
            }
            for modality in (
                "activity",
                "basal_insulin",
                "bolus_insulin",
                "glucose",
                "nutrition",
                "sleep_summary",
                "sleep_timeseries",
            )
        },
        "family_schema_validation": (
            family_schema_results
        ),
        "project_modalities": list(
            PROJECT_MODALITIES
        ),
        "participant_modalities": {
            participant: sorted(
                participant_modalities[
                    participant
                ]
            )
            for participant in participants
        },
        "five_modality_cohort_computed": (
            computed_five_modality
        ),
        "five_modality_cohort_frozen": (
            FROZEN_FIVE_MODALITY_COHORT
        ),
        "full_core_cohort_frozen": (
            FROZEN_FULL_CORE_COHORT
        ),
        "timestamp_quality": {
            "parse_failures": (
                timestamp_parse_failures
            ),
            "suspicious_values": (
                suspicious_timestamp_values
            ),
            "timestamp_issue_files": (
                timestamp_issue_files
            ),
            "suspicious_timestamp_files": (
                suspicious_timestamp_files
            ),
        },
        "approved_timestamp_corrections": (
            correction_results
        ),
        "unknown_participant_files": (
            unknown_participant_files
        ),
        "unknown_modality_files": (
            unknown_modality_files
        ),
        "expected_file_counts": (
            EXPECTED_FILE_COUNTS
        ),
        "file_profiles": file_profiles,
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        with REPORT_PATH.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                report,
                handle,
                indent=2,
                ensure_ascii=False,
            )

    except OSError as exc:

        fail(
            "Could not save profile report:\n"
            f"{exc}"
        )

    print()
    print("Report saved to:")
    print(REPORT_PATH)

    # -------------------------------------------------------------------------
    # FINAL STATUS
    # -------------------------------------------------------------------------

    structure_pass = (
        len(csv_files)
        == EXPECTED_CSV_COUNT
        and len(file_profiles)
        == EXPECTED_CSV_COUNT
        and not failed_files
        and not unknown_modality_files
        and not unknown_participant_files
    )

    timestamp_pass = (
        timestamp_parse_failures == 0
        and suspicious_timestamp_values == 0
        and all(
            result["status"] == "PASS"
            for result in correction_results[
                "results"
            ]
        )
    )

    schema_pass = not any_schema_invalid

    print()

    if (
        structure_pass
        and timestamp_pass
        and schema_pass
    ):

        print(
            "STRUCTURAL PROFILE STATUS: PASS"
        )

        print()
        print(
            "Verified dataset structure, "
            "participant mapping, modality mapping, "
            "file-family schemas, and timestamp "
            "quality are consistent."
        )

    else:

        print(
            "STRUCTURAL PROFILE STATUS: "
            "REVIEW REQUIRED"
        )

        if not structure_pass:

            print(
                "  - Structural/file mapping "
                "checks require review."
            )

        if not schema_pass:

            print(
                "  - One or more files differ "
                "from their modality-family "
                "reference schema."
            )

        if not timestamp_pass:

            print(
                "  - Timestamp quality or "
                "approved correction checks "
                "require review."
            )

    print()
    print(
        "This stage is descriptive only."
    )

    print(
        "No data transformation or model "
        "preparation was performed."
    )

    print(
        "Raw and derived dataset files "
        "were not modified."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()