"""
T1D-UOM SEQUENCE INPUT AUDIT
============================

READ-ONLY validation stage before sequence preparation.

This script:
    - validates the frozen modeling dataset;
    - validates the frozen 13-participant cohort;
    - validates five-modality coverage;
    - inventories only the frozen cohort for sequence preparation;
    - audits timestamp availability/parsing;
    - records duplicate timestamps and source ordering;
    - audits the two retained sleep representations;
    - validates the frozen neural architecture;
    - writes a JSON audit report.

This script DOES NOT:
    - modify raw data;
    - modify the timestamp-corrected dataset;
    - modify the modeling dataset;
    - delete rows;
    - alter values;
    - alter timestamps;
    - create sequences;
    - normalize data;
    - train a model;
    - create additional neural branches.

FROZEN ARCHITECTURE
-------------------

    Glucose   -> GRU -> zG ┐
    Insulin   -> GRU -> zI │
    Nutrition -> GRU -> zN ├-> MLP Fusion -> Unified Patient State
    Activity  -> GRU -> zA │
    Sleep     -> GRU -> zS ┘
                                      |
                                      v
                                 DIGITAL TWIN
                                  /         \
                                 v           v
                            Prediction     What-if
                                 \           /
                                  v         v
                               Interactive UI

IMPORTANT:
    The modeling dataset physically contains 19 participants.

    Only these 13 participants are frozen for sequence preparation:

        UoM2301
        UoM2302
        UoM2304
        UoM2305
        UoM2306
        UoM2307
        UoM2308
        UoM2309
        UoM2313
        UoM2314
        UoM2401
        UoM2403
        UoM2405

    Additional participants remain physically present in the modeling
    dataset but are NOT used for sequence preparation:

        UoM2303
        UoM2310
        UoM2312
        UoM2315
        UoM2320
        UoM2404
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ============================================================================
# 1. PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DATASET = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "t1d_uom_v1.0.3_modeling"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
)

REPORT_PATH = (
    REPORT_DIR
    / "sequence_input_audit.json"
)

EXCLUSION_MANIFEST = (
    REPORT_DIR
    / "modeling_dataset_exclusions.json"
)

DATASET_MANIFEST = (
    REPORT_DIR
    / "modeling_dataset_manifest.json"
)


# ============================================================================
# 2. FROZEN DATASET CONTRACT
# ============================================================================

EXPECTED_MODELING_CSV_COUNT = 110

FROZEN_COHORT = [
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

FROZEN_COHORT_SET = set(FROZEN_COHORT)

REQUIRED_MODALITIES = {
    "glucose",
    "insulin",
    "nutrition",
    "activity",
    "sleep",
}


# ============================================================================
# 3. FROZEN SYSTEM ARCHITECTURE
# ============================================================================
#
# IMPORTANT:
# Keep the variable name ARCHITECTURE exactly as written.
# The previous script failed because it later referenced "architecture"
# even though only "ARCHITECTURE" had been defined.
#
# This architecture is deliberately explicit and frozen.
# No additional neural branch is introduced for sleep_summary.
# Both sleep representations remain part of the single Sleep modality.
# ============================================================================

ARCHITECTURE = {
    "branches": {
        "glucose": {
            "encoder": "GRU",
            "latent": "zG",
        },
        "insulin": {
            "encoder": "GRU",
            "latent": "zI",
        },
        "nutrition": {
            "encoder": "GRU",
            "latent": "zN",
        },
        "activity": {
            "encoder": "GRU",
            "latent": "zA",
        },
        "sleep": {
            "encoder": "GRU",
            "latent": "zS",
        },
    },
    "fusion": {
        "type": "MLP",
        "inputs": [
            "zG",
            "zI",
            "zN",
            "zA",
            "zS",
        ],
        "output": "Unified Patient State",
    },
    "downstream": {
        "state": "DIGITAL TWIN",
        "outputs": [
            "Prediction",
            "What-if",
        ],
        "interface": "Interactive UI",
    },
}


# ============================================================================
# 4. FILE-NAME FAMILY DETECTION
# ============================================================================

FILE_PATTERNS = {
    "glucose": re.compile(
        r"^UoMGlucose(?P<pid>\d+)\.csv$",
        re.IGNORECASE,
    ),
    "activity": re.compile(
        r"^UoMActivity(?P<pid>\d+)\.csv$",
        re.IGNORECASE,
    ),
    "basal_insulin": re.compile(
        r"^UoMBasal(?P<pid>\d+)\.csv$",
        re.IGNORECASE,
    ),
    "bolus_insulin": re.compile(
        r"^UoMBolus(?P<pid>\d+)\.csv$",
        re.IGNORECASE,
    ),
    "nutrition": re.compile(
        r"^UoMNutrition(?P<pid>\d+)\.csv$",
        re.IGNORECASE,
    ),
    "sleep_summary": re.compile(
        r"^UoM(?P<pid>\d+)sleeptime\.csv$",
        re.IGNORECASE,
    ),
    "sleep_timeseries": re.compile(
        r"^UoMsleep(?P<pid>\d+)\.csv$",
        re.IGNORECASE,
    ),
}


# ============================================================================
# 5. TIMESTAMP COLUMN CONTRACT
# ============================================================================

TIMESTAMP_COLUMNS = {
    "glucose": [
        "glucose_ts",
        "bg_ts",
        "timestamp",
        "ts",
        "datetime",
    ],
    "activity": [
        "activity_ts",
        "timestamp",
        "ts",
        "datetime",
    ],
    "basal_insulin": [
        "basal_ts",
        "timestamp",
        "ts",
        "datetime",
    ],
    "bolus_insulin": [
        "bolus_ts",
        "timestamp",
        "ts",
        "datetime",
    ],
    "nutrition": [
        "meal_ts",
        "nutrition_ts",
        "timestamp",
        "ts",
        "datetime",
    ],
    "sleep_summary": [
        "start_date_ts",
        "sleep_ts",
        "timestamp",
        "ts",
        "datetime",
    ],
    "sleep_timeseries": [
        "sleep_ts",
        "timestamp",
        "ts",
        "datetime",
    ],
}


# ============================================================================
# 6. GENERAL HELPERS
# ============================================================================

def fatal(message: str) -> None:
    """Terminate the audit without touching any dataset."""
    print()
    print("=" * 80)
    print("SEQUENCE INPUT AUDIT FAILED")
    print("=" * 80)
    print(message)
    print()
    print("IMPORTANT:")
    print("  No dataset files were modified.")
    print("=" * 80)
    raise SystemExit(1)


def relative_model_path(path: Path) -> str:
    """Return path relative to the modeling dataset."""
    return path.relative_to(MODEL_DATASET).as_posix()


def clean_header(value: Any) -> str:
    """Normalize only the in-memory CSV header representation."""
    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lstrip("\ufeff")
    )


def is_missing(value: Any) -> bool:
    """Return True for common missing-value representations."""
    if value is None:
        return True

    text = str(value).strip().lower()

    return text in {
        "",
        "na",
        "n/a",
        "nan",
        "null",
        "none",
        "missing",
    }


def detect_file_family(
    path: Path,
) -> Optional[Tuple[str, str]]:
    """
    Identify modality family and participant from the frozen
    modeling-dataset filename.
    """
    for family, pattern in FILE_PATTERNS.items():
        match = pattern.match(path.name)

        if match:
            participant = f"UoM{match.group('pid')}"
            return family, participant

    return None


def project_modality_from_family(
    family: str,
) -> str:
    """
    Collapse physical file families into the five frozen architecture
    modalities.
    """
    if family in {
        "basal_insulin",
        "bolus_insulin",
    }:
        return "insulin"

    if family in {
        "sleep_summary",
        "sleep_timeseries",
    }:
        return "sleep"

    return family


# ============================================================================
# 7. TIMESTAMP PARSING
# ============================================================================

def parse_timestamp(
    value: Any,
) -> Optional[datetime]:
    """
    Parse common timestamp representations.

    This function ONLY parses an in-memory value.
    It never writes anything back to the dataset.
    """
    if is_missing(value):
        return None

    text = str(value).strip()

    candidates = [
        text,
        text.replace("Z", "+00:00"),
    ]

    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    # Numeric epoch fallback.
    try:
        number = float(text)

        if (
            100_000_000
            <= number
            <= 10_000_000_000
        ):
            return datetime.fromtimestamp(number)

        if (
            100_000_000_000
            <= number
            <= 10_000_000_000_000
        ):
            return datetime.fromtimestamp(
                number / 1000.0
            )

    except (
        ValueError,
        OverflowError,
        OSError,
    ):
        pass

    return None


def choose_timestamp_column(
    family: str,
    headers: Sequence[str],
) -> Optional[str]:
    """
    Choose the established timestamp column for a file family.

    The function first uses the known family-specific names and only
    falls back to a unique generic timestamp-like column.
    """
    normalized_headers = [
        clean_header(header)
        for header in headers
    ]

    known_candidates = TIMESTAMP_COLUMNS.get(
        family,
        [],
    )

    for candidate in known_candidates:
        if candidate in normalized_headers:
            return candidate

    generic_candidates = [
        header
        for header in normalized_headers
        if (
            header.lower().endswith("_ts")
            or header.lower()
            in {
                "timestamp",
                "datetime",
                "date_time",
            }
        )
    ]

    if len(generic_candidates) == 1:
        return generic_candidates[0]

    return None


# ============================================================================
# 8. CSV FILE AUDIT
# ============================================================================

def audit_file(
    path: Path,
    family: str,
    participant: str,
) -> Dict[str, Any]:
    """
    Read and audit one frozen-cohort CSV.

    No data are modified.
    """

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            fatal(
                "CSV has no header:\n"
                f"  {relative_model_path(path)}"
            )

        headers = [
            clean_header(header)
            for header in reader.fieldnames
            if header is not None
        ]

        timestamp_column = choose_timestamp_column(
            family,
            headers,
        )

        row_count = 0
        total_cells = 0
        missing_cells = 0

        timestamp_values: List[datetime] = []

        timestamp_parse_failures = 0
        missing_timestamp_count = 0

        for raw_row in reader:

            row_count += 1

            row = {
                clean_header(key): value
                for key, value in raw_row.items()
                if key is not None
            }

            for header in headers:
                total_cells += 1

                if is_missing(
                    row.get(header, "")
                ):
                    missing_cells += 1

            if timestamp_column is None:
                continue

            raw_timestamp = row.get(
                timestamp_column,
                "",
            )

            if is_missing(raw_timestamp):
                missing_timestamp_count += 1
                continue

            parsed = parse_timestamp(
                raw_timestamp
            )

            if parsed is None:
                timestamp_parse_failures += 1
            else:
                timestamp_values.append(parsed)

    # ------------------------------------------------------------------------
    # Timestamp uniqueness and source ordering.
    # ------------------------------------------------------------------------

    unique_timestamp_count = len(
        set(timestamp_values)
    )

    duplicate_timestamp_rows = (
        len(timestamp_values)
        - unique_timestamp_count
    )

    out_of_order_transitions = sum(
        1
        for previous, current
        in zip(
            timestamp_values,
            timestamp_values[1:],
        )
        if current < previous
    )

    # ------------------------------------------------------------------------
    # Positive time deltas after sorting.
    # This is descriptive only.
    # ------------------------------------------------------------------------

    sorted_timestamps = sorted(
        timestamp_values
    )

    positive_deltas = [
        (current - previous).total_seconds()
        for previous, current
        in zip(
            sorted_timestamps,
            sorted_timestamps[1:],
        )
        if (
            current - previous
        ).total_seconds() > 0
    ]

    delta_summary = {
        "count": len(positive_deltas),
        "minimum_seconds": (
            min(positive_deltas)
            if positive_deltas
            else None
        ),
        "median_seconds": (
            statistics.median(
                positive_deltas
            )
            if positive_deltas
            else None
        ),
        "mean_seconds": (
            statistics.mean(
                positive_deltas
            )
            if positive_deltas
            else None
        ),
        "maximum_seconds": (
            max(positive_deltas)
            if positive_deltas
            else None
        ),
    }

    return {
        "file": relative_model_path(path),
        "participant": participant,
        "modality_family": family,
        "project_modality": (
            project_modality_from_family(
                family
            )
        ),
        "row_count": row_count,
        "column_count": len(headers),
        "headers": headers,
        "timestamp_column": timestamp_column,
        "timestamp": {
            "non_missing_count": len(
                timestamp_values
            ),
            "missing_count": missing_timestamp_count,
            "parse_failure_count": (
                timestamp_parse_failures
            ),
            "unique_count": (
                unique_timestamp_count
            ),
            "duplicate_timestamp_rows": (
                duplicate_timestamp_rows
            ),
            "out_of_order_transitions": (
                out_of_order_transitions
            ),
            "minimum": (
                min(timestamp_values).isoformat()
                if timestamp_values
                else None
            ),
            "maximum": (
                max(timestamp_values).isoformat()
                if timestamp_values
                else None
            ),
            "delta_seconds": delta_summary,
        },
        "missing_cells": missing_cells,
        "total_cells": total_cells,
        "missingness_fraction": (
            missing_cells / total_cells
            if total_cells
            else 0.0
        ),
    }


# ============================================================================
# 9. MANIFEST VALIDATION
# ============================================================================

def validate_json_manifest(
    path: Path,
) -> str:
    """
    Validate only that a manifest exists and contains valid JSON.
    """
    if not path.exists():
        return "missing"

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            json.load(handle)

        return "readable"

    except Exception as exc:
        fatal(
            "Invalid JSON manifest:\n"
            f"  {path}\n"
            f"  Error: {exc}"
        )

    return "invalid"


# ============================================================================
# 10. ARCHITECTURE VALIDATION
# ============================================================================

def validate_frozen_architecture() -> None:
    """
    Validate the exact frozen five-branch architecture.
    """

    expected_branches = {
        "glucose": ("GRU", "zG"),
        "insulin": ("GRU", "zI"),
        "nutrition": ("GRU", "zN"),
        "activity": ("GRU", "zA"),
        "sleep": ("GRU", "zS"),
    }

    actual_branches = ARCHITECTURE.get(
        "branches",
        {},
    )

    if set(actual_branches) != set(
        expected_branches
    ):
        fatal(
            "Frozen architecture branch set changed.\n"
            f"Expected: {sorted(expected_branches)}\n"
            f"Observed: {sorted(actual_branches)}"
        )

    for modality, (
        expected_encoder,
        expected_latent,
    ) in expected_branches.items():

        actual = actual_branches.get(
            modality,
            {},
        )

        if actual.get("encoder") != expected_encoder:
            fatal(
                f"Frozen architecture encoder changed "
                f"for {modality}."
            )

        if actual.get("latent") != expected_latent:
            fatal(
                f"Frozen architecture latent changed "
                f"for {modality}."
            )

    fusion = ARCHITECTURE.get(
        "fusion",
        {},
    )

    if fusion.get("type") != "MLP":
        fatal(
            "Frozen fusion architecture changed. "
            "Expected MLP."
        )

    expected_inputs = [
        "zG",
        "zI",
        "zN",
        "zA",
        "zS",
    ]

    if fusion.get("inputs") != expected_inputs:
        fatal(
            "Frozen MLP fusion inputs changed.\n"
            f"Expected: {expected_inputs}\n"
            f"Observed: {fusion.get('inputs')}"
        )

    if fusion.get("output") != (
        "Unified Patient State"
    ):
        fatal(
            "Frozen fusion output changed."
        )

    downstream = ARCHITECTURE.get(
        "downstream",
        {},
    )

    if downstream.get("state") != "DIGITAL TWIN":
        fatal(
            "Frozen downstream state changed."
        )

    if downstream.get("outputs") != [
        "Prediction",
        "What-if",
    ]:
        fatal(
            "Frozen Digital Twin outputs changed."
        )

    if downstream.get("interface") != "Interactive UI":
        fatal(
            "Frozen interface architecture changed."
        )


# ============================================================================
# 11. MAIN
# ============================================================================

def main() -> None:

    print("=" * 80)
    print("T1D-UOM SEQUENCE INPUT AUDIT")
    print("=" * 80)
    print()
    print("IMPORTANT: READ-ONLY.")
    print("No dataset files will be modified.")
    print("No values will be transformed.")
    print("No sequences will be created.")
    print("No model will be trained.")

    print()
    print(
        f"Project root:       {PROJECT_ROOT}"
    )
    print(
        f"Modeling dataset:   {MODEL_DATASET}"
    )
    print(
        f"Report:             {REPORT_PATH}"
    )

    # ========================================================================
    # 1. MODELING DATASET VALIDATION
    # ========================================================================

    print()
    print("-" * 80)
    print("1. MODELING DATASET VALIDATION")
    print("-" * 80)

    if not MODEL_DATASET.exists():
        fatal(
            "Modeling dataset does not exist:\n"
            f"  {MODEL_DATASET}"
        )

    all_csv_files = sorted(
        MODEL_DATASET.rglob("*.csv")
    )

    print(
        "Modeling CSV files discovered: "
        f"{len(all_csv_files)}"
    )

    if len(all_csv_files) != (
        EXPECTED_MODELING_CSV_COUNT
    ):
        fatal(
            "Unexpected modeling CSV count.\n"
            f"Expected: {EXPECTED_MODELING_CSV_COUNT}\n"
            f"Observed: {len(all_csv_files)}"
        )

    print("Frozen modeling file count: PASS")

    # ========================================================================
    # 2. FREEZE MANIFEST VALIDATION
    # ========================================================================

    print()
    print("-" * 80)
    print("2. FREEZE MANIFEST VALIDATION")
    print("-" * 80)

    exclusion_manifest_status = (
        validate_json_manifest(
            EXCLUSION_MANIFEST
        )
    )

    dataset_manifest_status = (
        validate_json_manifest(
            DATASET_MANIFEST
        )
    )

    if exclusion_manifest_status == "readable":
        print(
            "PASS: modeling_dataset_exclusions.json"
        )
    else:
        print(
            "WARNING: modeling_dataset_exclusions.json "
            "not found"
        )

    if dataset_manifest_status == "readable":
        print(
            "PASS: modeling_dataset_manifest.json"
        )
    else:
        print(
            "WARNING: modeling_dataset_manifest.json "
            "not found"
        )

    # ========================================================================
    # 3. FILE INVENTORY
    # ========================================================================

    inventory: List[
        Tuple[Path, str, str]
    ] = []

    for path in all_csv_files:

        detected = detect_file_family(path)

        if detected is None:
            fatal(
                "Unknown CSV filename pattern:\n"
                f"  {relative_model_path(path)}"
            )

        family, participant = detected

        inventory.append(
            (
                path,
                family,
                participant,
            )
        )

    # ========================================================================
    # 4. PARTICIPANT / MODALITY INVENTORY
    # ========================================================================

    print()
    print("-" * 80)
    print("3. MODELING DATASET PARTICIPANT INVENTORY")
    print("-" * 80)

    coverage: Dict[
        str,
        set,
    ] = {}

    for _, family, participant in inventory:

        coverage.setdefault(
            participant,
            set(),
        ).add(
            project_modality_from_family(
                family
            )
        )

    represented_participants = set(
        coverage
    )

    print(
        "Participants represented in modeling dataset: "
        f"{len(represented_participants)}"
    )

    print()
    print("All represented participants:")

    for participant in sorted(
        represented_participants
    ):

        if participant in FROZEN_COHORT_SET:
            marker = "[FROZEN COHORT]"
        else:
            marker = "[ADDITIONAL / NOT FROZEN]"

        modalities = ", ".join(
            sorted(
                coverage[participant]
            )
        )

        print(
            f"  {participant}: "
            f"{modalities} "
            f"{marker}"
        )

    # ========================================================================
    # 5. FROZEN COHORT VALIDATION
    # ========================================================================

    print()
    print("-" * 80)
    print("4. FROZEN 13-PARTICIPANT COHORT VALIDATION")
    print("-" * 80)

    missing_frozen_participants = (
        FROZEN_COHORT_SET
        - represented_participants
    )

    if missing_frozen_participants:
        fatal(
            "Frozen cohort participant(s) missing:\n"
            + "\n".join(
                f"  - {participant}"
                for participant
                in sorted(
                    missing_frozen_participants
                )
            )
        )

    additional_participants = (
        represented_participants
        - FROZEN_COHORT_SET
    )

    print(
        "Frozen cohort participant presence: PASS"
    )
    print(
        f"Frozen cohort size: {len(FROZEN_COHORT)}"
    )
    print(
        "Additional participants physically present "
        "but NOT frozen: "
        f"{len(additional_participants)}"
    )

    if additional_participants:

        print()
        print(
            "These participants remain in the modeling "
            "dataset but are excluded from sequence preparation:"
        )

        for participant in sorted(
            additional_participants
        ):
            print(
                f"  - {participant}"
            )

    # ========================================================================
    # 6. FIVE-MODALITY ARCHITECTURE COVERAGE
    # ========================================================================

    print()
    print("-" * 80)
    print("5. FROZEN FIVE-MODALITY ARCHITECTURE COVERAGE")
    print("-" * 80)

    for participant in FROZEN_COHORT:

        available_modalities = coverage[
            participant
        ]

        missing_modalities = (
            REQUIRED_MODALITIES
            - available_modalities
        )

        print(
            f"{participant}: "
            f"{', '.join(sorted(available_modalities))}"
        )

        if missing_modalities:
            fatal(
                f"{participant} is missing required "
                f"modality/modalities: "
                f"{', '.join(sorted(missing_modalities))}"
            )

    print()
    print(
        "Five-modality coverage for frozen "
        "13-participant cohort: PASS"
    )

    # ========================================================================
    # 7. FROZEN FILE-LEVEL TEMPORAL AUDIT
    # ========================================================================

    print()
    print("-" * 80)
    print("6. FROZEN-COHORT FILE-LEVEL TEMPORAL AUDIT")
    print("-" * 80)

    frozen_inventory = [
        item
        for item in inventory
        if item[2] in FROZEN_COHORT_SET
    ]

    print(
        "Files audited for sequence preparation: "
        f"{len(frozen_inventory)}"
    )

    print(
        "Additional non-frozen participant files "
        "are NOT used in this sequence-input audit."
    )

    file_results: List[
        Dict[str, Any]
    ] = []

    for index, (
        path,
        family,
        participant,
    ) in enumerate(
        frozen_inventory,
        start=1,
    ):

        print(
            f"[{index:03d}/{len(frozen_inventory)}] "
            f"{relative_model_path(path)}"
        )

        result = audit_file(
            path,
            family,
            participant,
        )

        file_results.append(result)

    # ========================================================================
    # 8. TEMPORAL QUALITY
    # ========================================================================

    print()
    print("-" * 80)
    print("7. TEMPORAL QUALITY SUMMARY")
    print("-" * 80)

    files_without_timestamp_column = [
        result["file"]
        for result in file_results
        if result["timestamp_column"] is None
    ]

    files_with_parse_failures = [
        {
            "file": result["file"],
            "count": result[
                "timestamp"
            ][
                "parse_failure_count"
            ],
        }
        for result in file_results
        if result[
            "timestamp"
        ][
            "parse_failure_count"
        ] > 0
    ]

    files_with_duplicate_timestamps = [
        {
            "file": result["file"],
            "duplicate_timestamp_rows": result[
                "timestamp"
            ][
                "duplicate_timestamp_rows"
            ],
        }
        for result in file_results
        if result[
            "timestamp"
        ][
            "duplicate_timestamp_rows"
        ] > 0
    ]

    files_with_out_of_order_timestamps = [
        {
            "file": result["file"],
            "out_of_order_transitions": result[
                "timestamp"
            ][
                "out_of_order_transitions"
            ],
        }
        for result in file_results
        if result[
            "timestamp"
        ][
            "out_of_order_transitions"
        ] > 0
    ]

    print(
        "Files with undetected timestamp column: "
        f"{len(files_without_timestamp_column)}"
    )

    print(
        "Files with timestamp parse failures: "
        f"{len(files_with_parse_failures)}"
    )

    print(
        "Files with duplicate timestamps: "
        f"{len(files_with_duplicate_timestamps)}"
    )

    print(
        "Files with out-of-order timestamps: "
        f"{len(files_with_out_of_order_timestamps)}"
    )

    # ========================================================================
    # 9. MODALITY-FAMILY SUMMARY
    # ========================================================================

    print()
    print("-" * 80)
    print("8. FROZEN-COHORT MODALITY-FAMILY SUMMARY")
    print("-" * 80)

    family_summary: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for result in file_results:

        family = result[
            "modality_family"
        ]

        entry = family_summary.setdefault(
            family,
            {
                "files": 0,
                "participants": set(),
                "rows": 0,
                "timestamp_columns": Counter(),
                "timestamp_parse_failures": 0,
                "missing_timestamps": 0,
                "duplicate_timestamp_rows": 0,
                "out_of_order_transitions": 0,
            },
        )

        entry["files"] += 1

        entry["participants"].add(
            result["participant"]
        )

        entry["rows"] += result[
            "row_count"
        ]

        entry[
            "timestamp_columns"
        ][
            str(
                result["timestamp_column"]
            )
        ] += 1

        entry[
            "timestamp_parse_failures"
        ] += result[
            "timestamp"
        ][
            "parse_failure_count"
        ]

        entry[
            "missing_timestamps"
        ] += result[
            "timestamp"
        ][
            "missing_count"
        ]

        entry[
            "duplicate_timestamp_rows"
        ] += result[
            "timestamp"
        ][
            "duplicate_timestamp_rows"
        ]

        entry[
            "out_of_order_transitions"
        ] += result[
            "timestamp"
        ][
            "out_of_order_transitions"
        ]

    serializable_family_summary = {}

    for family in sorted(
        family_summary
    ):

        entry = family_summary[
            family
        ]

        serializable_family_summary[
            family
        ] = {
            "files": entry["files"],
            "participants": sorted(
                entry["participants"]
            ),
            "rows": entry["rows"],
            "timestamp_columns": dict(
                entry[
                    "timestamp_columns"
                ]
            ),
            "timestamp_parse_failures": (
                entry[
                    "timestamp_parse_failures"
                ]
            ),
            "missing_timestamps": (
                entry[
                    "missing_timestamps"
                ]
            ),
            "duplicate_timestamp_rows": (
                entry[
                    "duplicate_timestamp_rows"
                ]
            ),
            "out_of_order_transitions": (
                entry[
                    "out_of_order_transitions"
                ]
            ),
        }

        print(
            f"{family:18s} "
            f"files={entry['files']:2d} "
            f"participants="
            f"{len(entry['participants']):2d} "
            f"rows={entry['rows']:8d}"
        )

        print(
            "  timestamp columns: "
            + ", ".join(
                f"{name}={count}"
                for name, count
                in entry[
                    "timestamp_columns"
                ].items()
            )
        )

    # ========================================================================
    # 10. SLEEP REPRESENTATION AUDIT
    # ========================================================================

    print()
    print("-" * 80)
    print("9. SLEEP REPRESENTATION AUDIT")
    print("-" * 80)

    sleep_summary_files = [
        result
        for result in file_results
        if result[
            "modality_family"
        ] == "sleep_summary"
    ]

    sleep_timeseries_files = [
        result
        for result in file_results
        if result[
            "modality_family"
        ] == "sleep_timeseries"
    ]

    print(
        "Sleep-summary files in frozen cohort: "
        f"{len(sleep_summary_files)}"
    )

    print(
        "Sleep-time-series files in frozen cohort: "
        f"{len(sleep_timeseries_files)}"
    )

    print()
    print("Architecture constraint:")
    print("  Sleep -> GRU -> zS")
    print()
    print(
        "The two retained sleep representations are "
        "audited separately."
    )

    print(
        "No additional Sleep neural branch is created."
    )

    print(
        "The representation/sequence construction decision "
        "does not change the frozen architecture."
    )

    # ========================================================================
    # 11. ARCHITECTURE CONTRACT
    # ========================================================================

    print()
    print("-" * 80)
    print("10. FROZEN ARCHITECTURE CONTRACT")
    print("-" * 80)

    validate_frozen_architecture()

    print("Glucose   -> GRU -> zG")
    print("Insulin   -> GRU -> zI")
    print("Nutrition -> GRU -> zN")
    print("Activity  -> GRU -> zA")
    print("Sleep     -> GRU -> zS")
    print("                 \\")
    print("                  -> MLP Fusion")
    print("                     -> Unified Patient State")
    print("                        -> DIGITAL TWIN")
    print("                           -> Prediction / What-if")
    print("                              -> Interactive UI")

    print()
    print("Frozen architecture contract: PASS")

    # ========================================================================
    # 12. AUDIT STATUS
    # ========================================================================
    #
    # Missing timestamp columns and parse failures are blocking because
    # sequence construction cannot safely proceed without timestamp data.
    #
    # Duplicate timestamps and out-of-order source rows are NOT automatically
    # classified as dataset corruption. They are explicitly recorded as
    # sequence-preparation requirements.
    #
    # This audit does not repair them.
    # ========================================================================

    if (
        files_without_timestamp_column
        or files_with_parse_failures
    ):
        audit_status = "REVIEW REQUIRED"
    else:
        audit_status = "PASS"

    # ========================================================================
    # 13. WRITE JSON REPORT
    # ========================================================================

    report = {
        "report_name": (
            "T1D-UOM Sequence Input Audit"
        ),
        "report_version": "3.0",
        "read_only": True,
        "project_root": str(
            PROJECT_ROOT
        ),
        "modeling_dataset": str(
            MODEL_DATASET
        ),
        "modeling_dataset_csv_count": (
            len(all_csv_files)
        ),
        "expected_modeling_csv_count": (
            EXPECTED_MODELING_CSV_COUNT
        ),
        "all_participants": sorted(
            represented_participants
        ),
        "frozen_cohort": FROZEN_COHORT,
        "frozen_cohort_size": len(
            FROZEN_COHORT
        ),
        "additional_non_frozen_participants": (
            sorted(
                additional_participants
            )
        ),
        "frozen_files_audited": len(
            file_results
        ),
        "manifest_status": {
            "modeling_dataset_exclusions.json": (
                exclusion_manifest_status
            ),
            "modeling_dataset_manifest.json": (
                dataset_manifest_status
            ),
        },
        "architecture": ARCHITECTURE,
        "coverage": {
            participant: sorted(
                coverage[participant]
            )
            for participant
            in FROZEN_COHORT
        },
        "temporal_quality": {
            "files_with_undetected_timestamp_column": (
                files_without_timestamp_column
            ),
            "files_with_timestamp_parse_failures": (
                files_with_parse_failures
            ),
            "files_with_duplicate_timestamps": (
                files_with_duplicate_timestamps
            ),
            "files_with_out_of_order_timestamps": (
                files_with_out_of_order_timestamps
            ),
            "duplicate_timestamps_are_blocking": False,
            "out_of_order_timestamps_are_blocking": False,
            "sequence_preparation_requirements": [
                (
                    "Deterministic chronological ordering "
                    "must be established."
                ),
                (
                    "Duplicate timestamps require an explicit "
                    "deterministic tie policy."
                ),
                (
                    "No rows are removed by this audit."
                ),
                (
                    "No timestamps are modified by this audit."
                ),
            ],
        },
        "family_summary": (
            serializable_family_summary
        ),
        "sleep_representation": {
            "sleep_summary_files": (
                len(sleep_summary_files)
            ),
            "sleep_timeseries_files": (
                len(sleep_timeseries_files)
            ),
            "neural_sleep_branches": 1,
            "latent": "zS",
        },
        "file_results": file_results,
        "status": audit_status,
        "next_stage": (
            (
                "Proceed to sequence-preparation design. "
                "The sequence builder must explicitly define "
                "chronological ordering and duplicate-timestamp "
                "tie handling."
            )
            if audit_status == "PASS"
            else
            (
                "Do not create sequences until timestamp "
                "column and parse failures are resolved."
            )
        ),
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    # ========================================================================
    # 14. FINAL TERMINAL OUTPUT
    # ========================================================================

    print()
    print("=" * 80)
    print("T1D-UOM SEQUENCE INPUT AUDIT COMPLETE")
    print("=" * 80)

    print()
    print("COMPLETE MODELING DATASET")
    print(
        f"  CSV files:                 "
        f"{len(all_csv_files)}"
    )
    print(
        f"  Participants represented: "
        f"{len(represented_participants)}"
    )

    print()
    print("FROZEN MODELING COHORT")
    print(
        f"  Participants:              "
        f"{len(FROZEN_COHORT)}"
    )

    for participant in FROZEN_COHORT:
        print(
            f"    - {participant}"
        )

    print()
    print("ADDITIONAL NON-FROZEN PARTICIPANTS")

    if additional_participants:
        for participant in sorted(
            additional_participants
        ):
            print(
                f"    - {participant}"
            )
    else:
        print("    None")

    print()
    print("FROZEN FIVE-MODALITY COVERAGE")
    print("  Status: PASS")

    print()
    print("TEMPORAL AUDIT")
    print(
        "  Files audited:             "
        f"{len(file_results)}"
    )
    print(
        "  Timestamp column failures: "
        f"{len(files_without_timestamp_column)}"
    )
    print(
        "  Timestamp parse failures:  "
        f"{len(files_with_parse_failures)}"
    )
    print(
        "  Files with duplicate timestamps: "
        f"{len(files_with_duplicate_timestamps)}"
    )
    print(
        "  Files with out-of-order timestamps: "
        f"{len(files_with_out_of_order_timestamps)}"
    )

    print()
    print("FROZEN ARCHITECTURE")
    print("  Glucose   -> GRU -> zG")
    print("  Insulin   -> GRU -> zI")
    print("  Nutrition -> GRU -> zN")
    print("  Activity  -> GRU -> zA")
    print("  Sleep     -> GRU -> zS")
    print(
        "  zG,zI,zN,zA,zS -> MLP Fusion"
    )
    print(
        "  MLP Fusion -> Unified Patient State"
    )
    print(
        "  Unified Patient State -> DIGITAL TWIN"
    )
    print(
        "  DIGITAL TWIN -> Prediction / What-if"
    )
    print(
        "  Prediction / What-if -> Interactive UI"
    )

    print()
    print("Report saved to:")
    print(
        f"  {REPORT_PATH}"
    )

    print()
    print(
        "SEQUENCE INPUT AUDIT STATUS: "
        f"{audit_status}"
    )

    if audit_status == "PASS":

        print()
        print("RESULT:")
        print(
            "  Frozen 13-participant cohort is present "
            "and five-modality complete."
        )
        print(
            "  All audited files have detectable and "
            "parseable timestamp columns."
        )
        print(
            "  Duplicate timestamps and source ordering "
            "issues are recorded for deterministic "
            "sequence preparation."
        )
        print(
            "  No dataset values were modified."
        )

        print()
        print("NEXT STAGE:")
        print(
            "  Proceed to sequence-preparation design."
        )
        print(
            "  The sequence builder must explicitly define "
            "chronological sorting and duplicate-timestamp "
            "tie handling."
        )

    else:

        print()
        print("IMPORTANT:")
        print(
            "  Do NOT create modeling sequences yet."
        )
        print(
            "  Review the timestamp failures reported above."
        )

    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()