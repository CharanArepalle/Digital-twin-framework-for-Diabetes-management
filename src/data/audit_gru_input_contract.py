"""
T1D-UOM
GRU INPUT CONTRACT AUDIT

Purpose
-------
Read-only audit of the VERIFIED sequence-input files before connecting
them to the five frozen modality-specific GRU branches.

FROZEN ARCHITECTURE
-------------------

Glucose   -> GRU -> zG
Insulin   -> GRU -> zI
Nutrition -> GRU -> zN
Activity  -> GRU -> zA
Sleep     -> GRU -> zS

This script DOES NOT:
    - modify any dataset
    - modify sequence inputs
    - reorder rows
    - delete rows
    - resample
    - interpolate
    - impute
    - normalize
    - create windows
    - create targets
    - train a model
    - implement MLP Fusion
    - implement Unified Patient State
    - implement Digital Twin
    - implement Prediction
    - implement What-if
    - implement Interactive UI

It only inspects the already verified sequence-input CSV files and
determines their exact schema/tensor-input contract.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


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
    / "gru_input_contract_audit.json"
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
# FROZEN MODALITIES
# ============================================================================

ARCHITECTURE_MODALITIES = (
    "glucose",
    "insulin",
    "nutrition",
    "activity",
    "sleep",
)


# ============================================================================
# PATH -> MODALITY IDENTIFICATION
# ============================================================================

def identify_modality(relative_path: Path) -> str:
    """
    Identify the architecture modality from the verified sequence-input path.
    """

    parts = [part.lower() for part in relative_path.parts]
    filename = relative_path.name.lower()

    if "activity data" in parts:
        return "activity"

    if "glucose data" in parts:
        return "glucose"

    if "nutrition data" in parts:
        return "nutrition"

    if "insulin data" in parts:
        return "insulin"

    if "sleep data" in parts:
        return "sleep"

    # Defensive filename fallback.
    if "activity" in filename:
        return "activity"

    if "glucose" in filename:
        return "glucose"

    if "nutrition" in filename:
        return "nutrition"

    if "basal" in filename or "bolus" in filename:
        return "insulin"

    if "sleep" in filename:
        return "sleep"

    raise RuntimeError(
        f"Unable to identify modality from sequence-input file:\n"
        f"{relative_path}"
    )


# ============================================================================
# PARTICIPANT IDENTIFICATION
# ============================================================================

def identify_participant(filename: str) -> str:
    """
    Extract UoM participant identifier from a filename.

    Examples:
        UoMActivity2301.csv -> UoM2301
        UoMGlucose2301.csv  -> UoM2301
        UoMBasal2301.csv    -> UoM2301
        UoM2301sleeptime.csv -> UoM2301
        UoMsleep2301.csv    -> UoM2301
    """

    match = re.search(r"UoM(?:Activity|Glucose|Nutrition|Basal|Bolus)?(\d{4})",
                      filename,
                      flags=re.IGNORECASE)

    if match is None:
        match = re.search(
            r"UoMsleep(\d{4})",
            filename,
            flags=re.IGNORECASE,
        )

    if match is None:
        raise RuntimeError(
            f"Unable to identify participant from filename:\n"
            f"{filename}"
        )

    return f"UoM{match.group(1)}"


# ============================================================================
# TIMESTAMP IDENTIFICATION
# ============================================================================

TIMESTAMP_CANDIDATES = {
    "activity": {"activity_ts"},
    "glucose": {"bg_ts"},
    "nutrition": {"meal_ts"},
    "sleep": {"sleep_ts", "start_date_ts"},
    "insulin": {"basal_ts", "bolus_ts"},
}


def identify_timestamp_column(
    modality: str,
    columns: List[str],
) -> str:
    """
    Identify the timestamp column without transforming the data.
    """

    normalized = {
        column.strip().lower(): column
        for column in columns
    }

    candidates = TIMESTAMP_CANDIDATES[modality]

    matches = [
        normalized[candidate]
        for candidate in candidates
        if candidate in normalized
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple timestamp columns detected for modality "
            f"{modality}: {matches}"
        )

    raise RuntimeError(
        f"No recognized timestamp column found for modality "
        f"{modality}.\n"
        f"Columns: {columns}"
    )


# ============================================================================
# CSV SCHEMA READING
# ============================================================================

def read_csv_schema(path: Path) -> Tuple[List[str], int]:
    """
    Read only the CSV header and count rows.

    Values are not modified.
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
                f"CSV is empty:\n{path}"
            )

        if not header:
            raise RuntimeError(
                f"CSV has an empty header:\n{path}"
            )

        row_count = sum(1 for _ in reader)

    return header, row_count


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def discover_csv_files() -> List[Path]:
    if not SEQUENCE_INPUT_DIR.exists():
        raise RuntimeError(
            f"Sequence-input directory does not exist:\n"
            f"{SEQUENCE_INPUT_DIR}"
        )

    files = sorted(
        path
        for path in SEQUENCE_INPUT_DIR.rglob("*.csv")
        if path.is_file()
    )

    return files


# ============================================================================
# MAIN AUDIT
# ============================================================================

def main() -> None:

    print("=" * 80)
    print("T1D-UOM GRU INPUT CONTRACT AUDIT")
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
    print("No windows will be created.")
    print("No targets will be created.")
    print("No model will be trained.")

    print()
    print(f"Project root:       {PROJECT_ROOT}")
    print(f"Sequence inputs:    {SEQUENCE_INPUT_DIR}")
    print(f"Audit report:       {REPORT_PATH}")

    # ------------------------------------------------------------------------
    # 1. DIRECTORY
    # ------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("1. SEQUENCE INPUT DIRECTORY VALIDATION")
    print("-" * 80)

    files = discover_csv_files()

    print(f"Sequence-input CSV files discovered: {len(files)}")

    if len(files) != 86:
        raise RuntimeError(
            f"Expected exactly 86 frozen-cohort sequence-input CSV files, "
            f"but discovered {len(files)}."
        )

    print("Frozen sequence-input file count: PASS")

    # ------------------------------------------------------------------------
    # 2. FILE INVENTORY
    # ------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("2. FROZEN COHORT FILE INVENTORY")
    print("-" * 80)

    participant_files: Dict[str, List[str]] = defaultdict(list)
    modality_files: Dict[str, List[str]] = defaultdict(list)

    records: List[dict] = []

    for index, path in enumerate(files, start=1):

        relative_path = path.relative_to(SEQUENCE_INPUT_DIR)

        participant = identify_participant(path.name)
        modality = identify_modality(relative_path)

        if participant not in FROZEN_COHORT:
            raise RuntimeError(
                f"Non-frozen participant found in sequence inputs:\n"
                f"{relative_path}\n"
                f"Participant: {participant}"
            )

        participant_files[participant].append(
            relative_path.as_posix()
        )

        modality_files[modality].append(
            relative_path.as_posix()
        )

        print(
            f"[{index:03d}/{len(files):03d}] "
            f"{relative_path.as_posix()}"
        )

    # ------------------------------------------------------------------------
    # 3. FIVE-MODALITY COVERAGE
    # ------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("3. FROZEN FIVE-MODALITY COVERAGE")
    print("-" * 80)

    for participant in FROZEN_COHORT:

        participant_modalities = set()

        for relative in participant_files[participant]:

            modality = identify_modality(
                Path(relative)
            )

            participant_modalities.add(modality)

        missing = set(ARCHITECTURE_MODALITIES) - participant_modalities

        print(
            f"{participant}: "
            f"{', '.join(sorted(participant_modalities))}"
        )

        if missing:
            raise RuntimeError(
                f"{participant} is missing architecture modality/ies: "
                f"{sorted(missing)}"
            )

    print()
    print("Five-modality coverage: PASS")

    # ------------------------------------------------------------------------
    # 4. SCHEMA AUDIT
    # ------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("4. REAL SEQUENCE-INPUT SCHEMA AUDIT")
    print("-" * 80)

    schema_by_modality: Dict[str, List[dict]] = defaultdict(list)

    total_rows = 0

    for path in files:

        relative_path = path.relative_to(SEQUENCE_INPUT_DIR)

        participant = identify_participant(path.name)
        modality = identify_modality(relative_path)

        columns, row_count = read_csv_schema(path)

        timestamp_column = identify_timestamp_column(
            modality,
            columns,
        )

        feature_columns = [
            column
            for column in columns
            if column != timestamp_column
        ]

        if not feature_columns:
            raise RuntimeError(
                f"No feature columns remain after excluding timestamp "
                f"column in:\n{relative_path}"
            )

        record = {
            "participant": participant,
            "modality": modality,
            "relative_path": relative_path.as_posix(),
            "timestamp_column": timestamp_column,
            "column_count": len(columns),
            "feature_count": len(feature_columns),
            "columns": columns,
            "feature_columns": feature_columns,
            "row_count": row_count,
        }

        schema_by_modality[modality].append(record)

        total_rows += row_count

    # ------------------------------------------------------------------------
    # 5. MODALITY FEATURE CONTRACT
    # ------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("5. MODALITY FEATURE DIMENSION CONTRACT")
    print("-" * 80)

    modality_contract = {}

    for modality in ARCHITECTURE_MODALITIES:

        records_for_modality = schema_by_modality[modality]

        if not records_for_modality:
            raise RuntimeError(
                f"No sequence-input files found for modality: {modality}"
            )

        feature_counts = sorted(
            {
                record["feature_count"]
                for record in records_for_modality
            }
        )

        timestamp_columns = sorted(
            {
                record["timestamp_column"]
                for record in records_for_modality
            }
        )

        print()
        print(f"{modality.upper()}")
        print(
            f"  Files:              {len(records_for_modality)}"
        )
        print(
            f"  Feature dimensions: {feature_counts}"
        )
        print(
            f"  Timestamp columns:  {timestamp_columns}"
        )

        modality_contract[modality] = {
            "file_count": len(records_for_modality),
            "feature_dimensions": feature_counts,
            "timestamp_columns": timestamp_columns,
            "records": records_for_modality,
        }

    # ------------------------------------------------------------------------
    # 6. FEATURE-DIMENSION CONSISTENCY
    # ------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("6. WITHIN-MODALITY FEATURE CONSISTENCY")
    print("-" * 80)

    consistency_status = {}

    for modality in ARCHITECTURE_MODALITIES:

        dimensions = modality_contract[modality]["feature_dimensions"]

        if len(dimensions) == 1:
            status = "PASS"
        else:
            status = "REQUIRES REVIEW"

        consistency_status[modality] = status

        print(
            f"{modality:<12} "
            f"feature dimensions={dimensions} "
            f"-> {status}"
        )

    # ------------------------------------------------------------------------
    # 7. SLEEP REPRESENTATION INSPECTION
    # ------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("7. SLEEP REPRESENTATION INSPECTION")
    print("-" * 80)

    sleep_records = schema_by_modality["sleep"]

    sleep_summary = []
    sleep_timeseries = []

    for record in sleep_records:

        timestamp = record["timestamp_column"]

        if timestamp == "start_date_ts":
            sleep_summary.append(record)

        elif timestamp == "sleep_ts":
            sleep_timeseries.append(record)

        else:
            raise RuntimeError(
                f"Unexpected Sleep timestamp column: {timestamp}"
            )

    print(
        f"Sleep-summary files:      {len(sleep_summary)}"
    )

    print(
        f"Sleep-time-series files:  {len(sleep_timeseries)}"
    )

    print()
    print("Frozen architecture constraint:")
    print("  Sleep -> GRU -> zS")
    print()
    print(
        "No second Sleep neural branch is introduced by this audit."
    )
    print(
        "No combination rule is invented by this audit."
    )
    print(
        "Sleep representation selection remains explicitly unresolved "
        "until its schema is reviewed."
    )

    # ------------------------------------------------------------------------
    # 8. FINAL CONTRACT STATUS
    # ------------------------------------------------------------------------

    all_consistent = all(
        status == "PASS"
        for status in consistency_status.values()
    )

    print()
    print("-" * 80)
    print("8. GRU INPUT CONTRACT STATUS")
    print("-" * 80)

    print()
    print("Frozen architecture:")
    print("  Glucose   -> GRU -> zG")
    print("  Insulin   -> GRU -> zI")
    print("  Nutrition -> GRU -> zN")
    print("  Activity  -> GRU -> zA")
    print("  Sleep     -> GRU -> zS")

    print()

    if all_consistent:
        print(
            "Within-modality feature dimensionality: PASS"
        )
    else:
        print(
            "Within-modality feature dimensionality: "
            "REQUIRES REVIEW"
        )

    print(
        "Sleep single-branch architecture constraint: PASS"
    )

    # ------------------------------------------------------------------------
    # 9. REPORT
    # ------------------------------------------------------------------------

    report = {
        "project": "T1D-UOM",
        "purpose": "GRU input contract audit",
        "read_only": True,
        "sequence_input_directory": str(SEQUENCE_INPUT_DIR),
        "frozen_cohort": list(FROZEN_COHORT),
        "architecture": {
            "glucose": "Glucose -> GRU -> zG",
            "insulin": "Insulin -> GRU -> zI",
            "nutrition": "Nutrition -> GRU -> zN",
            "activity": "Activity -> GRU -> zA",
            "sleep": "Sleep -> GRU -> zS",
            "fusion": "zG,zI,zN,zA,zS -> MLP Fusion",
            "downstream": (
                "MLP Fusion -> Unified Patient State -> "
                "DIGITAL TWIN -> Prediction / What-if -> "
                "Interactive UI"
            ),
        },
        "file_count": len(files),
        "total_rows": total_rows,
        "modality_contract": modality_contract,
        "consistency_status": consistency_status,
        "sleep_representation": {
            "sleep_summary_files": len(sleep_summary),
            "sleep_timeseries_files": len(sleep_timeseries),
            "single_sleep_branch_required": True,
            "representation_selection_made": False,
        },
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
    print("9. WRITING AUDIT REPORT")
    print("-" * 80)

    print(
        f"Report saved:\n  {REPORT_PATH}"
    )

    # ------------------------------------------------------------------------
    # 10. FINAL STATUS
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)

    if all_consistent:
        print("T1D-UOM GRU INPUT CONTRACT AUDIT COMPLETED")
        print("=" * 80)
        print()
        print("RESULT:")
        print("  Frozen five-modality sequence inputs were inspected.")
        print("  Feature dimensions were determined from the real files.")
        print("  Within-modality feature consistency: PASS")
        print("  Sleep single-branch architecture: PRESERVED")
        print()
        print("No dataset values were modified.")
        print("No sequence inputs were modified.")
        print("No model was trained.")
        print()
        print("NEXT STAGE:")
        print("  Use the audited dimensions to define the real")
        print("  five-GRU input contract.")
    else:
        print("T1D-UOM GRU INPUT CONTRACT AUDIT REQUIRES REVIEW")
        print("=" * 80)
        print()
        print(
            "At least one modality has inconsistent feature "
            "dimensions across files."
        )
        print()
        print(
            "DO NOT connect the real data to the GRUs yet."
        )

    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 80)
        print("GRU INPUT CONTRACT AUDIT FAILED")
        print("=" * 80)
        print(str(exc))
        print()
        print("IMPORTANT:")
        print("  No dataset files were modified.")
        print("  No sequence-input files were modified.")
        print("  No model was trained.")
        print()
        sys.exit(1)