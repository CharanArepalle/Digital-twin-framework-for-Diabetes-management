"""
T1D-UOM
TARGETED GRU SCHEMA DIFFERENCE INSPECTION

Purpose
-------
Read-only inspection of the already verified sequence-input CSV files.

This script exists ONLY to determine why:
    Insulin -> [1, 2] features
    Sleep   -> [6, 14] features

It does NOT:
    - modify any dataset
    - modify sequence inputs
    - modify manifests
    - reorder rows
    - delete rows
    - resample
    - interpolate
    - impute
    - normalize
    - engineer features
    - create windows
    - create targets
    - train a model
    - change the frozen architecture
    - create another GRU branch
    - implement MLP Fusion
    - implement Digital Twin
    - implement Prediction
    - implement What-if
    - implement Interactive UI

FROZEN ARCHITECTURE
-------------------

Glucose   -> GRU -> zG
Insulin   -> GRU -> zI
Nutrition -> GRU -> zN
Activity  -> GRU -> zA
Sleep     -> GRU -> zS

This script only inspects the real CSV schemas so that the
single Insulin branch and single Sleep branch can later be
connected correctly without guessing.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


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

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
)

REPORT_PATH = (
    REPORT_DIR
    / "gru_schema_difference_inspection.json"
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
    "UoM2314",
    "UoM2401",
    "UoM2403",
    "UoM2405",
)


# ============================================================================
# HELPERS
# ============================================================================

def identify_participant(filename: str) -> str:
    """
    Extract participant identifier from known T1D-UOM filename families.
    """

    patterns = (
        r"UoMActivity(\d{4})",
        r"UoMGlucose(\d{4})",
        r"UoMNutrition(\d{4})",
        r"UoMBasal(\d{4})",
        r"UoMBolus(\d{4})",
        r"UoMsleep(\d{4})",
        r"UoM(\d{4})sleeptime",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            filename,
            flags=re.IGNORECASE,
        )

        if match:
            return f"UoM{match.group(1)}"

    raise RuntimeError(
        f"Unable to identify participant from filename:\n{filename}"
    )


def identify_family(relative_path: Path, filename: str) -> str:
    """
    Identify the exact source family relevant to the schema inspection.
    """

    path_text = relative_path.as_posix().lower()
    name = filename.lower()

    if "insulin data/basal data/" in path_text:
        return "basal_insulin"

    if "insulin data/bolus data/" in path_text:
        return "bolus_insulin"

    if "sleep data/" in path_text and "sleeptime" in name:
        return "sleep_summary"

    if "sleep data/" in path_text and "uomsleep" in name:
        return "sleep_timeseries"

    return "other"


def read_header_and_row_count(path: Path) -> tuple[List[str], int]:
    """
    Read the CSV header and count rows.

    No values are changed.
    """

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.reader(handle)

        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError(
                f"Empty CSV file:\n{path}"
            )

        row_count = sum(1 for _ in reader)

    return header, row_count


def normalized_columns(columns: List[str]) -> List[str]:
    """
    Normalize only for comparison/display logic.

    Original column names remain unchanged.
    """

    return [
        column.strip().lower()
        for column in columns
    ]


def identify_timestamp_column(
    family: str,
    columns: List[str],
) -> str:

    normalized = {
        column.strip().lower(): column
        for column in columns
    }

    if family == "basal_insulin":
        candidates = ("basal_ts",)

    elif family == "bolus_insulin":
        candidates = ("bolus_ts",)

    elif family == "sleep_summary":
        candidates = ("start_date_ts",)

    elif family == "sleep_timeseries":
        candidates = ("sleep_ts",)

    else:
        raise RuntimeError(
            f"Unsupported family for timestamp identification: {family}"
        )

    matches = [
        normalized[name]
        for name in candidates
        if name in normalized
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one timestamp column for {family}, "
            f"but found {matches}.\n"
            f"Columns: {columns}"
        )

    return matches[0]


def format_columns(columns: List[str]) -> str:
    if not columns:
        return "    <none>"

    return "\n".join(
        f"    [{index:02d}] {column}"
        for index, column in enumerate(columns, start=1)
    )


def signature(columns: List[str]) -> tuple[str, ...]:
    """
    Case/whitespace-insensitive schema signature.
    """

    return tuple(
        column.strip().lower()
        for column in columns
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 80)
    print("T1D-UOM TARGETED GRU SCHEMA DIFFERENCE INSPECTION")
    print("=" * 80)

    print()
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
    print("No windows will be created.")
    print("No targets will be created.")
    print("No model will be trained.")

    print()
    print(f"Project root:       {PROJECT_ROOT}")
    print(f"Sequence inputs:    {SEQUENCE_INPUT_DIR}")
    print(f"Inspection report:  {REPORT_PATH}")

    # ------------------------------------------------------------------------
    # DIRECTORY VALIDATION
    # ------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("1. SEQUENCE INPUT DIRECTORY VALIDATION")
    print("-" * 80)

    if not SEQUENCE_INPUT_DIR.exists():
        raise RuntimeError(
            f"Sequence-input directory does not exist:\n"
            f"{SEQUENCE_INPUT_DIR}"
        )

    all_csv_files = sorted(
        path
        for path in SEQUENCE_INPUT_DIR.rglob("*.csv")
        if path.is_file()
    )

    print(
        f"Sequence-input CSV files discovered: {len(all_csv_files)}"
    )

    if len(all_csv_files) != 86:
        raise RuntimeError(
            "Expected exactly 86 frozen-cohort sequence-input files, "
            f"but discovered {len(all_csv_files)}."
        )

    print("Frozen sequence-input file count: PASS")

    # ------------------------------------------------------------------------
    # TARGETED FAMILY COLLECTION
    # ------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("2. TARGETED FAMILY COLLECTION")
    print("-" * 80)

    target_families = (
        "basal_insulin",
        "bolus_insulin",
        "sleep_summary",
        "sleep_timeseries",
    )

    family_files: Dict[str, List[Path]] = {
        family: []
        for family in target_families
    }

    for path in all_csv_files:

        relative_path = path.relative_to(
            SEQUENCE_INPUT_DIR
        )

        family = identify_family(
            relative_path,
            path.name,
        )

        if family in family_files:
            participant = identify_participant(path.name)

            if participant not in FROZEN_COHORT:
                raise RuntimeError(
                    f"Non-frozen participant found in sequence inputs:\n"
                    f"{relative_path}\n"
                    f"Participant: {participant}"
                )

            family_files[family].append(path)

    expected_counts = {
        "basal_insulin": 12,
        "bolus_insulin": 13,
        "sleep_summary": 11,
        "sleep_timeseries": 11,
    }

    for family in target_families:

        actual = len(family_files[family])
        expected = expected_counts[family]

        print(
            f"{family:<18} files={actual} "
            f"expected={expected}"
        )

        if actual != expected:
            raise RuntimeError(
                f"Unexpected file count for {family}: "
                f"{actual} != {expected}"
            )

    print()
    print("Targeted family counts: PASS")

    # ------------------------------------------------------------------------
    # SCHEMA INSPECTION
    # ------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("3. ACTUAL INSULIN AND SLEEP SCHEMA INSPECTION")
    print("-" * 80)

    report_families = {}

    for family in target_families:

        print()
        print("=" * 80)
        print(f"{family.upper()}")
        print("=" * 80)

        records = []
        signatures = defaultdict(list)

        for path in sorted(family_files[family]):

            relative_path = path.relative_to(
                SEQUENCE_INPUT_DIR
            )

            participant = identify_participant(
                path.name
            )

            columns, row_count = read_header_and_row_count(
                path
            )

            timestamp_column = identify_timestamp_column(
                family,
                columns,
            )

            feature_columns = [
                column
                for column in columns
                if column.strip().lower()
                != timestamp_column.strip().lower()
            ]

            record = {
                "participant": participant,
                "relative_path": relative_path.as_posix(),
                "row_count": row_count,
                "timestamp_column": timestamp_column,
                "all_columns": columns,
                "feature_columns": feature_columns,
                "feature_count": len(feature_columns),
            }

            records.append(record)

            signatures[
                signature(feature_columns)
            ].append(participant)

        # ------------------------------------------------------------
        # FAMILY SUMMARY
        # ------------------------------------------------------------

        feature_counts = sorted(
            {
                record["feature_count"]
                for record in records
            }
        )

        print()
        print(
            f"Files: {len(records)}"
        )

        print(
            f"Feature counts observed: {feature_counts}"
        )

        print()

        for record in records:

            print(
                f"{record['participant']} | "
                f"{record['relative_path']}"
            )

            print(
                f"  rows: {record['row_count']}"
            )

            print(
                f"  timestamp: {record['timestamp_column']}"
            )

            print(
                f"  feature count: {record['feature_count']}"
            )

            print("  feature columns:")

            print(
                format_columns(
                    record["feature_columns"]
                )
            )

            print()

        # ------------------------------------------------------------
        # SCHEMA SIGNATURE SUMMARY
        # ------------------------------------------------------------

        print(
            "-" * 80
        )

        print(
            f"{family} feature-schema signatures: "
            f"{len(signatures)}"
        )

        signature_records = []

        for index, (sig, participants) in enumerate(
            sorted(signatures.items()),
            start=1,
        ):

            columns = list(sig)

            print()
            print(
                f"Signature {index}:"
            )

            print(
                f"  Feature count: {len(columns)}"
            )

            print(
                "  Normalized feature columns:"
            )

            print(
                format_columns(columns)
            )

            print(
                "  Participants/files:"
            )

            for participant in sorted(participants):
                print(
                    f"    - {participant}"
                )

            signature_records.append(
                {
                    "feature_count": len(columns),
                    "normalized_feature_columns": columns,
                    "participants": sorted(participants),
                }
            )

        report_families[family] = {
            "file_count": len(records),
            "feature_counts_observed": feature_counts,
            "schema_signature_count": len(signatures),
            "schema_signatures": signature_records,
            "files": records,
        }

    # ------------------------------------------------------------------------
    # ARCHITECTURE SAFETY CHECK
    # ------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("4. FROZEN ARCHITECTURE SAFETY CHECK")
    print("-" * 80)

    print()
    print("Frozen architecture:")
    print()
    print("Glucose   -> GRU -> zG")
    print("Insulin   -> GRU -> zI")
    print("Nutrition -> GRU -> zN")
    print("Activity  -> GRU -> zA")
    print("Sleep     -> GRU -> zS")
    print()
    print("zG,zI,zN,zA,zS -> MLP Fusion")
    print("MLP Fusion -> Unified Patient State")
    print("Unified Patient State -> DIGITAL TWIN")
    print("DIGITAL TWIN -> Prediction / What-if")
    print("Prediction / What-if -> Interactive UI")

    print()
    print("Architecture modification: NONE")
    print("Additional Insulin GRU branch: NO")
    print("Additional Sleep GRU branch: NO")
    print("MLP Fusion implemented: NO")
    print("Digital Twin implemented: NO")
    print("Prediction implemented: NO")
    print("What-if implemented: NO")
    print("Interactive UI implemented: NO")

    # ------------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------------

    report = {
        "project": "T1D-UOM",
        "purpose": (
            "Targeted read-only inspection of Insulin and Sleep "
            "sequence-input schema differences before real GRU connection."
        ),
        "read_only": True,
        "sequence_input_directory": str(
            SEQUENCE_INPUT_DIR
        ),
        "frozen_cohort": list(
            FROZEN_COHORT
        ),
        "architecture": {
            "glucose": "Glucose -> GRU -> zG",
            "insulin": "Insulin -> GRU -> zI",
            "nutrition": "Nutrition -> GRU -> zN",
            "activity": "Activity -> GRU -> zA",
            "sleep": "Sleep -> GRU -> zS",
            "fusion": "zG,zI,zN,zA,zS -> MLP Fusion",
            "unified_state": (
                "MLP Fusion -> Unified Patient State"
            ),
            "digital_twin": (
                "Unified Patient State -> DIGITAL TWIN"
            ),
            "downstream": (
                "DIGITAL TWIN -> Prediction / What-if -> "
                "Interactive UI"
            ),
        },
        "targeted_families": report_families,
        "architecture_changed": False,
        "additional_gru_branches_created": False,
        "data_modified": False,
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

    print()
    print("-" * 80)
    print("5. WRITING INSPECTION REPORT")
    print("-" * 80)

    print(
        f"Inspection report saved:\n"
        f"  {REPORT_PATH}"
    )

    # ------------------------------------------------------------------------
    # FINAL STATUS
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("T1D-UOM TARGETED GRU SCHEMA INSPECTION COMPLETE")
    print("=" * 80)

    print()
    print("RESULT:")
    print("  Actual Insulin schemas have been inspected.")
    print("  Actual Sleep schemas have been inspected.")
    print("  No architecture change was made.")
    print("  No data was modified.")
    print("  No model was trained.")

    print()
    print("IMPORTANT:")
    print(
        "  This inspection does NOT decide how the differing "
        "schemas are combined."
    )
    print(
        "  No additional GRU branch has been introduced."
    )
    print(
        "  The frozen single Insulin -> GRU -> zI branch remains."
    )
    print(
        "  The frozen single Sleep -> GRU -> zS branch remains."
    )

    print()
    print("NEXT STAGE:")
    print(
        "  Use the actual column-level findings above to define "
        "the minimal architecture-consistent input contract."
    )

    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:

        print()
        print("=" * 80)
        print("TARGETED GRU SCHEMA INSPECTION FAILED")
        print("=" * 80)
        print()
        print(str(exc))
        print()
        print("IMPORTANT:")
        print("  No dataset files were modified.")
        print("  No sequence-input files were modified.")
        print("  No model was trained.")
        print()
        sys.exit(1)