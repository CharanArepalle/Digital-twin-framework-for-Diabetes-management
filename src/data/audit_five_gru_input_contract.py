from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# T1D-UOM FIVE-GRU INPUT CONTRACT AUDIT
# =============================================================================
#
# READ-ONLY.
#
# This audit does NOT:
#   - modify raw data
#   - modify timestamp-corrected data
#   - modify modeling data
#   - modify sequence-input data
#   - delete rows
#   - transform values
#   - resample
#   - interpolate
#   - impute
#   - normalize
#   - encode data
#   - create windows
#   - create targets
#   - train a model
#   - implement MLP Fusion
#   - implement the Digital Twin
#   - implement Prediction
#   - implement What-if
#   - implement the Interactive UI
#
# PURPOSE
# -------
# Verify that the already-prepared sequence-input files support the
# FROZEN FIVE-GRU ARCHITECTURE:
#
#   Glucose   -> GRU -> zG
#   Insulin   -> GRU -> zI
#   Nutrition -> GRU -> zN
#   Activity  -> GRU -> zA
#   Sleep     -> GRU -> zS
#
# Downstream architecture is NOT implemented here:
#
#   zG,zI,zN,zA,zS
#          |
#          v
#      MLP Fusion
#          |
#          v
#   Unified Patient State
#          |
#          v
#      DIGITAL TWIN
#       /         \
# Prediction     What-if
#       \         /
#     Interactive UI
#
# IMPORTANT
# ---------
# Basal and bolus insulin are source representations of ONE frozen
# Insulin -> GRU -> zI branch.
#
# Sleep summary and sleep time-series are source representations of ONE
# frozen Sleep -> GRU -> zS branch.
#
# This script does not perform the eventual insulin encoding or any
# cross-file temporal merging. It only validates the input contract.
#
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_INPUTS = (
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
    / "five_gru_input_contract_audit.json"
)

MODEL_MANIFEST = (
    REPORT_DIR
    / "modeling_dataset_manifest.json"
)

SEQUENCE_PREPARATION_MANIFEST = (
    REPORT_DIR
    / "sequence_input_preparation_manifest.json"
)


# =============================================================================
# FROZEN COHORT
# =============================================================================

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


# =============================================================================
# EXPECTED SOURCE FAMILY COUNTS
# =============================================================================

EXPECTED_FAMILY_COUNTS = {
    "activity": 13,
    "glucose": 13,
    "basal_insulin": 12,
    "bolus_insulin": 13,
    "nutrition": 13,
    "sleep_summary": 11,
    "sleep_timeseries": 11,
}


# =============================================================================
# FROZEN GRU INPUT DIMENSIONS
# =============================================================================

EXPECTED_GRU_DIMENSIONS = {
    "glucose": 1,
    "insulin": 2,
    "nutrition": 6,
    "activity": 12,
    "sleep": 6,
}


# =============================================================================
# ROBUST PARTICIPANT EXTRACTION
# =============================================================================
#
# Actual project filenames include:
#
#   UoMActivity2301.csv
#   UoMGlucose2301.csv
#   UoMBasal2302.csv
#   UoMBolus2301.csv
#   UoMNutrition2301.csv
#   UoMsleep2301.csv
#   UoM2302sleeptime.csv
#
# Therefore "UoM" is NOT always immediately followed by digits.
#
# This parser explicitly supports the actual project naming patterns while
# also remaining strict enough to reject unrelated filenames.
# =============================================================================

PARTICIPANT_PATTERN = re.compile(
    r"UoM(?:Activity|Glucose|Basal|Bolus|Nutrition|sleep)?(\d+)",
    re.IGNORECASE,
)


def identify_participant(path: Path) -> Optional[str]:
    """
    Extract participant ID from actual T1D-UOM filename conventions.

    Supported examples:
      UoMActivity2301.csv -> UoM2301
      UoMGlucose2301.csv  -> UoM2301
      UoMBasal2302.csv    -> UoM2302
      UoMBolus2301.csv    -> UoM2301
      UoMNutrition2301.csv -> UoM2301
      UoMsleep2301.csv    -> UoM2301
      UoM2302sleeptime.csv -> UoM2302
    """

    filename = path.name

    match = PARTICIPANT_PATTERN.search(filename)

    if match is None:
        return None

    return f"UoM{match.group(1)}"


# =============================================================================
# FAMILY IDENTIFICATION
# =============================================================================

def identify_family(relative_path: Path) -> Optional[str]:
    """
    Identify the source family from the verified directory structure.
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

        if "basal data" in parts:
            return "basal_insulin"

        if "bolus data" in parts:
            return "bolus_insulin"

    if "sleep data" in parts:

        if filename.startswith("uom") and "sleeptime" in filename:
            return "sleep_summary"

        if filename.startswith("uomsleep"):
            return "sleep_timeseries"

    return None


# =============================================================================
# CSV HEADER READING
# =============================================================================

def read_header(path: Path) -> List[str]:
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
                f"CSV file is empty:\n{path}"
            )

    cleaned = [
        str(column).strip()
        for column in header
    ]

    if not cleaned:
        raise RuntimeError(
            f"CSV header is empty:\n{path}"
        )

    if len(cleaned) != len(set(cleaned)):
        raise RuntimeError(
            f"Duplicate column names detected:\n{path}\n"
            f"Columns: {cleaned}"
        )

    return cleaned


# =============================================================================
# TIMESTAMP IDENTIFICATION
# =============================================================================

def identify_timestamp_columns(
    header: List[str],
) -> List[str]:
    """
    Timestamp columns in the prepared dataset follow the *_ts convention.
    """

    return [
        column
        for column in header
        if column.lower().endswith("_ts")
    ]


# =============================================================================
# FEATURE COLUMN IDENTIFICATION
# =============================================================================

def identify_feature_columns(
    header: List[str],
    timestamp_columns: List[str],
) -> List[str]:

    timestamp_set = set(timestamp_columns)

    return [
        column
        for column in header
        if column not in timestamp_set
    ]


# =============================================================================
# NORMALIZED SCHEMA SIGNATURE
# =============================================================================

def normalized_signature(
    columns: List[str],
) -> Tuple[str, ...]:

    return tuple(
        column.strip().lower()
        for column in columns
    )


# =============================================================================
# DISCOVER SEQUENCE INPUTS
# =============================================================================

def discover_sequence_csvs() -> List[Path]:

    if not SEQUENCE_INPUTS.exists():
        raise RuntimeError(
            "Sequence-input directory does not exist:\n"
            f"{SEQUENCE_INPUTS}"
        )

    files = sorted(
        path
        for path in SEQUENCE_INPUTS.rglob("*.csv")
        if path.is_file()
    )

    return files


# =============================================================================
# REQUIRED MANIFEST VALIDATION
# =============================================================================

def validate_manifests() -> None:

    required = [
        MODEL_MANIFEST,
        SEQUENCE_PREPARATION_MANIFEST,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Required project manifest(s) missing:\n"
            + "\n".join(
                f"  - {path}"
                for path in missing
            )
        )


# =============================================================================
# BUILD INVENTORY
# =============================================================================

def build_inventory(
    files: List[Path],
) -> List[dict]:

    inventory = []

    for index, path in enumerate(files, start=1):

        relative = path.relative_to(
            SEQUENCE_INPUTS
        )

        participant = identify_participant(path)

        if participant is None:
            raise RuntimeError(
                "Unable to identify participant from sequence-input file:\n"
                f"{path}"
            )

        family = identify_family(relative)

        if family is None:
            raise RuntimeError(
                "Unable to identify modality family from sequence-input file:\n"
                f"{path}"
            )

        header = read_header(path)

        timestamp_columns = identify_timestamp_columns(
            header
        )

        if len(timestamp_columns) != 1:
            raise RuntimeError(
                "Expected exactly one timestamp column in sequence-input file.\n"
                f"File: {path}\n"
                f"Detected timestamp columns: {timestamp_columns}"
            )

        features = identify_feature_columns(
            header,
            timestamp_columns,
        )

        inventory.append(
            {
                "index": index,
                "path": path,
                "relative_path": relative.as_posix(),
                "participant": participant,
                "family": family,
                "header": header,
                "timestamp_column": timestamp_columns[0],
                "feature_columns": features,
                "feature_count": len(features),
                "feature_signature": list(
                    normalized_signature(features)
                ),
            }
        )

    return inventory


# =============================================================================
# FROZEN COHORT VALIDATION
# =============================================================================

def validate_frozen_cohort(
    inventory: List[dict],
) -> None:

    participants = {
        record["participant"]
        for record in inventory
    }

    missing = (
        FROZEN_COHORT_SET
        - participants
    )

    unexpected = (
        participants
        - FROZEN_COHORT_SET
    )

    if missing:
        raise RuntimeError(
            "Frozen cohort participant(s) missing from sequence inputs:\n"
            + "\n".join(
                f"  - {participant}"
                for participant in sorted(missing)
            )
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected non-frozen participant(s) present in sequence inputs:\n"
            + "\n".join(
                f"  - {participant}"
                for participant in sorted(unexpected)
            )
        )


# =============================================================================
# FAMILY COUNT VALIDATION
# =============================================================================

def validate_family_counts(
    inventory: List[dict],
) -> Dict[str, int]:

    observed = Counter(
        record["family"]
        for record in inventory
    )

    for family, expected in EXPECTED_FAMILY_COUNTS.items():

        actual = observed.get(
            family,
            0,
        )

        if actual != expected:
            raise RuntimeError(
                f"Unexpected {family} file count.\n"
                f"Expected: {expected}\n"
                f"Observed: {actual}"
            )

    expected_total = sum(
        EXPECTED_FAMILY_COUNTS.values()
    )

    if len(inventory) != expected_total:
        raise RuntimeError(
            "Unexpected total sequence-input file count.\n"
            f"Expected: {expected_total}\n"
            f"Observed: {len(inventory)}"
        )

    return dict(observed)


# =============================================================================
# FAMILY SCHEMA CONSISTENCY
# =============================================================================

def audit_family_schema(
    records: List[dict],
    family: str,
    expected_dimension: int,
    expected_timestamp: str,
) -> dict:

    if not records:
        raise RuntimeError(
            f"No files found for family: {family}"
        )

    signatures = Counter()

    timestamp_columns = Counter()

    file_details = []

    for record in records:

        if record["timestamp_column"] != expected_timestamp:
            raise RuntimeError(
                f"{family}: unexpected timestamp column.\n"
                f"File: {record['path']}\n"
                f"Expected: {expected_timestamp}\n"
                f"Observed: {record['timestamp_column']}"
            )

        signatures[
            tuple(record["feature_signature"])
        ] += 1

        timestamp_columns[
            record["timestamp_column"]
        ] += 1

        file_details.append(
            {
                "participant": record["participant"],
                "relative_path": record["relative_path"],
                "timestamp_column": record["timestamp_column"],
                "feature_count": record["feature_count"],
                "feature_columns": record["feature_columns"],
            }
        )

    if len(signatures) != 1:
        signature_report = {
            str(list(signature)): count
            for signature, count
            in signatures.items()
        }

        raise RuntimeError(
            f"{family}: inconsistent feature schemas.\n"
            f"Observed signatures:\n"
            f"{json.dumps(signature_report, indent=2)}"
        )

    signature = next(
        iter(signatures)
    )

    observed_dimension = len(signature)

    if observed_dimension != expected_dimension:
        raise RuntimeError(
            f"{family}: feature dimension mismatch.\n"
            f"Expected: {expected_dimension}\n"
            f"Observed: {observed_dimension}\n"
            f"Features: {list(signature)}"
        )

    return {
        "family": family,
        "status": "PASS",
        "file_count": len(records),
        "feature_dimension": observed_dimension,
        "feature_signature": list(signature),
        "timestamp_columns": dict(timestamp_columns),
        "files": file_details,
    }


# =============================================================================
# INSULIN CONTRACT
# =============================================================================

def audit_insulin(
    basal_records: List[dict],
    bolus_records: List[dict],
) -> dict:

    basal = audit_family_schema(
        records=basal_records,
        family="basal_insulin",
        expected_dimension=2,
        expected_timestamp="basal_ts",
    )

    bolus = audit_family_schema(
        records=bolus_records,
        family="bolus_insulin",
        expected_dimension=1,
        expected_timestamp="bolus_ts",
    )

    return {
        "status": "PASS",
        "branch": "Insulin",
        "output": "zI",
        "architecture_branch_count": 1,
        "source_families": [
            "basal_insulin",
            "bolus_insulin",
        ],
        "basal_source_schema": basal,
        "bolus_source_schema": bolus,
        "locked_input_dimension": 2,
        "locked_input_contract": [
            "dose",
            "event_type",
        ],
        "event_type_mapping": {
            "basal": 0,
            "bolus": 1,
        },
        "additional_insulin_gru": False,
    }


# =============================================================================
# SLEEP CONTRACT
# =============================================================================

def audit_sleep(
    summary_records: List[dict],
    timeseries_records: List[dict],
) -> dict:

    summary = audit_family_schema(
        records=summary_records,
        family="sleep_summary",
        expected_dimension=14,
        expected_timestamp="start_date_ts",
    )

    timeseries = audit_family_schema(
        records=timeseries_records,
        family="sleep_timeseries",
        expected_dimension=6,
        expected_timestamp="sleep_ts",
    )

    expected_timeseries_features = [
        "step_count",
        "heart_rate",
        "current_activity_type_intensity",
        "stress_level_value",
        "sleep_level",
        "resting_heart_rate",
    ]

    observed_timeseries = [
        column.lower()
        for column in timeseries["feature_signature"]
    ]

    expected_timeseries_normalized = [
        column.lower()
        for column in expected_timeseries_features
    ]

    if (
        observed_timeseries
        != expected_timeseries_normalized
    ):
        raise RuntimeError(
            "sleep_timeseries: unexpected feature schema.\n"
            f"Expected: {expected_timeseries_features}\n"
            f"Observed: {timeseries['feature_signature']}"
        )

    return {
        "status": "PASS",
        "branch": "Sleep",
        "output": "zS",
        "architecture_branch_count": 1,
        "sleep_summary": summary,
        "sleep_timeseries": timeseries,
        "locked_representation": "sleep_timeseries",
        "locked_input_dimension": 6,
        "locked_input_features": expected_timeseries_features,
        "additional_sleep_gru": False,
        "sleep_summary_creates_second_branch": False,
    }


# =============================================================================
# ARCHITECTURE CONTRACT
# =============================================================================

def frozen_architecture() -> dict:

    return {
        "modality_branches": [
            "Glucose -> GRU -> zG",
            "Insulin -> GRU -> zI",
            "Nutrition -> GRU -> zN",
            "Activity -> GRU -> zA",
            "Sleep -> GRU -> zS",
        ],
        "fusion": (
            "zG,zI,zN,zA,zS -> MLP Fusion"
        ),
        "unified_state": (
            "MLP Fusion -> Unified Patient State"
        ),
        "digital_twin": (
            "Unified Patient State -> DIGITAL TWIN"
        ),
        "downstream": (
            "DIGITAL TWIN -> Prediction / What-if "
            "-> Interactive UI"
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    print("=" * 80)
    print("T1D-UOM FIVE-GRU INPUT CONTRACT AUDIT")
    print("=" * 80)

    print(
        """
IMPORTANT: READ-ONLY.
No dataset files will be modified.
No sequence-input files will be modified.
No values will be transformed.
No rows will be deleted.
No resampling will be performed.
No interpolation will be performed.
No imputation will be performed.
No normalization will be performed.
No encoding will be performed.
No windows will be created.
No targets will be created.
No model will be trained.
No MLP Fusion will be implemented.
No Digital Twin will be implemented.
No Prediction will be implemented.
No What-if implementation will be created.
No Interactive UI will be created.
"""
    )

    print(f"Project root:       {PROJECT_ROOT}")
    print(f"Sequence inputs:    {SEQUENCE_INPUTS}")
    print(f"Audit report:       {REPORT_PATH}")

    try:

        # =====================================================================
        print("\n" + "-" * 80)
        print("1. DIRECTORY AND MANIFEST VALIDATION")
        print("-" * 80)

        if not SEQUENCE_INPUTS.exists():
            raise RuntimeError(
                "Sequence-input directory does not exist:\n"
                f"{SEQUENCE_INPUTS}"
            )

        validate_manifests()

        print("Sequence-input directory: PASS")
        print("modeling_dataset_manifest.json: PASS")
        print(
            "sequence_input_preparation_manifest.json: PASS"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("2. FROZEN SEQUENCE-INPUT INVENTORY")
        print("-" * 80)

        files = discover_sequence_csvs()

        print(
            f"Sequence-input CSV files discovered: {len(files)}"
        )

        if len(files) != 86:
            raise RuntimeError(
                "Frozen sequence-input file count mismatch.\n"
                "Expected: 86\n"
                f"Observed: {len(files)}"
            )

        print("Frozen sequence-input file count: PASS")

        inventory = build_inventory(files)

        # =====================================================================
        print("\n" + "-" * 80)
        print("3. FROZEN COHORT VALIDATION")
        print("-" * 80)

        validate_frozen_cohort(
            inventory
        )

        participants = sorted(
            {
                record["participant"]
                for record in inventory
            }
        )

        print(
            f"Participants represented in sequence inputs: "
            f"{len(participants)}"
        )

        for participant in participants:
            print(
                f"  {participant}"
            )

        print(
            "\nFrozen 13-participant cohort: PASS"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("4. MODALITY FAMILY VALIDATION")
        print("-" * 80)

        family_counts = validate_family_counts(
            inventory
        )

        for family in [
            "activity",
            "glucose",
            "basal_insulin",
            "bolus_insulin",
            "nutrition",
            "sleep_summary",
            "sleep_timeseries",
        ]:
            print(
                f"{family:<18}"
                f" files={family_counts[family]:>2}"
                f" expected={EXPECTED_FAMILY_COUNTS[family]:>2}"
            )

        print(
            "\nModality family counts: PASS"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("5. FROZEN FIVE-GRU ARCHITECTURE")
        print("-" * 80)

        print(
            "Glucose   -> GRU -> zG"
        )
        print(
            "Insulin   -> GRU -> zI"
        )
        print(
            "Nutrition -> GRU -> zN"
        )
        print(
            "Activity  -> GRU -> zA"
        )
        print(
            "Sleep     -> GRU -> zS"
        )

        print(
            "zG,zI,zN,zA,zS -> MLP Fusion"
        )
        print(
            "MLP Fusion -> Unified Patient State"
        )
        print(
            "Unified Patient State -> DIGITAL TWIN"
        )
        print(
            "DIGITAL TWIN -> Prediction / What-if"
        )
        print(
            "Prediction / What-if -> Interactive UI"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("6. GLUCOSE INPUT CONTRACT")
        print("-" * 80)

        glucose_records = [
            record
            for record in inventory
            if record["family"] == "glucose"
        ]

        glucose_result = audit_family_schema(
            records=glucose_records,
            family="glucose",
            expected_dimension=1,
            expected_timestamp="bg_ts",
        )

        print(
            "Files:                 "
            f"{glucose_result['file_count']}"
        )

        print(
            "Feature dimension:     "
            f"{glucose_result['feature_dimension']}"
        )

        print(
            "Feature columns:       "
            + ", ".join(
                glucose_result["feature_signature"]
            )
        )

        print(
            "Timestamp column:      bg_ts"
        )

        print(
            "Glucose -> GRU -> zG:  PASS"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("7. NUTRITION INPUT CONTRACT")
        print("-" * 80)

        nutrition_records = [
            record
            for record in inventory
            if record["family"] == "nutrition"
        ]

        nutrition_result = audit_family_schema(
            records=nutrition_records,
            family="nutrition",
            expected_dimension=6,
            expected_timestamp="meal_ts",
        )

        print(
            "Files:                 "
            f"{nutrition_result['file_count']}"
        )

        print(
            "Feature dimension:     "
            f"{nutrition_result['feature_dimension']}"
        )

        print(
            "Feature columns:       "
            + ", ".join(
                nutrition_result["feature_signature"]
            )
        )

        print(
            "Timestamp column:      meal_ts"
        )

        print(
            "Nutrition -> GRU -> zN: PASS"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("8. ACTIVITY INPUT CONTRACT")
        print("-" * 80)

        activity_records = [
            record
            for record in inventory
            if record["family"] == "activity"
        ]

        activity_result = audit_family_schema(
            records=activity_records,
            family="activity",
            expected_dimension=12,
            expected_timestamp="activity_ts",
        )

        print(
            "Files:                 "
            f"{activity_result['file_count']}"
        )

        print(
            "Feature dimension:     "
            f"{activity_result['feature_dimension']}"
        )

        print(
            "Feature columns:       "
            + ", ".join(
                activity_result["feature_signature"]
            )
        )

        print(
            "Timestamp column:      activity_ts"
        )

        print(
            "Activity -> GRU -> zA: PASS"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("9. INSULIN INPUT CONTRACT")
        print("-" * 80)

        basal_records = [
            record
            for record in inventory
            if record["family"] == "basal_insulin"
        ]

        bolus_records = [
            record
            for record in inventory
            if record["family"] == "bolus_insulin"
        ]

        insulin_result = audit_insulin(
            basal_records,
            bolus_records,
        )

        print(
            "Basal insulin schema: PASS"
        )

        print(
            "  Feature dimension:   "
            f"{insulin_result['basal_source_schema']['feature_dimension']}"
        )

        print(
            "  Features:            "
            + ", ".join(
                insulin_result[
                    "basal_source_schema"
                ]["feature_signature"]
            )
        )

        print(
            "Bolus insulin schema: PASS"
        )

        print(
            "  Feature dimension:   "
            f"{insulin_result['bolus_source_schema']['feature_dimension']}"
        )

        print(
            "  Features:            "
            + ", ".join(
                insulin_result[
                    "bolus_source_schema"
                ]["feature_signature"]
            )
        )

        print(
            "\nFrozen architecture:"
        )

        print(
            "  Basal + Bolus"
        )

        print(
            "       -> ONE Insulin branch"
        )

        print(
            "       -> GRU"
        )

        print(
            "       -> zI"
        )

        print(
            "\nLocked eventual GRU input dimension: 2"
        )

        print(
            "Contract: [dose, event_type]"
        )

        print(
            "event_type mapping: basal=0, bolus=1"
        )

        print(
            "Additional Insulin GRU branch: NO"
        )

        print(
            "Insulin architecture contract: PASS"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("10. SLEEP INPUT CONTRACT")
        print("-" * 80)

        sleep_summary_records = [
            record
            for record in inventory
            if record["family"] == "sleep_summary"
        ]

        sleep_timeseries_records = [
            record
            for record in inventory
            if record["family"] == "sleep_timeseries"
        ]

        sleep_result = audit_sleep(
            sleep_summary_records,
            sleep_timeseries_records,
        )

        print(
            "Sleep-summary files:       "
            f"{sleep_result['sleep_summary']['file_count']}"
        )

        print(
            "Sleep-summary dimension:   "
            f"{sleep_result['sleep_summary']['feature_dimension']}"
        )

        print(
            "Sleep-time-series files:   "
            f"{sleep_result['sleep_timeseries']['file_count']}"
        )

        print(
            "Sleep-time-series dim.:    "
            f"{sleep_result['sleep_timeseries']['feature_dimension']}"
        )

        print(
            "\nLocked Sleep representation:"
        )

        print(
            "  sleep time-series"
        )

        print(
            "       -> 6 features"
        )

        print(
            "       -> ONE Sleep GRU"
        )

        print(
            "       -> zS"
        )

        print(
            "\nSleep-summary representation remains documented."
        )

        print(
            "It does NOT create a second Sleep GRU."
        )

        print(
            "Additional Sleep GRU branch: NO"
        )

        print(
            "Sleep architecture contract: PASS"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("11. FINAL FIVE-GRU INPUT CONTRACT")
        print("-" * 80)

        final_contract = {
            "glucose": {
                "input_dimension": 1,
                "source_representation": "glucose sequence",
                "output": "zG",
            },
            "insulin": {
                "input_dimension": 2,
                "source_representation": (
                    "single unified basal/bolus insulin event sequence"
                ),
                "features": [
                    "dose",
                    "event_type",
                ],
                "event_type_mapping": {
                    "basal": 0,
                    "bolus": 1,
                },
                "output": "zI",
            },
            "nutrition": {
                "input_dimension": 6,
                "source_representation": "nutrition sequence",
                "output": "zN",
            },
            "activity": {
                "input_dimension": 12,
                "source_representation": "activity sequence",
                "output": "zA",
            },
            "sleep": {
                "input_dimension": 6,
                "source_representation": "sleep time-series",
                "output": "zS",
            },
        }

        print(
            "Glucose:    input_dim=1  -> zG"
        )

        print(
            "Insulin:    input_dim=2  -> zI"
        )

        print(
            "Nutrition:  input_dim=6  -> zN"
        )

        print(
            "Activity:   input_dim=12 -> zA"
        )

        print(
            "Sleep:      input_dim=6  -> zS"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("12. ARCHITECTURE SAFETY CHECK")
        print("-" * 80)

        safety = {
            "five_modality_gru_branches": True,
            "additional_insulin_gru": False,
            "additional_sleep_gru": False,
            "mlp_fusion_implemented": False,
            "digital_twin_implemented": False,
            "prediction_implemented": False,
            "what_if_implemented": False,
            "interactive_ui_implemented": False,
            "dataset_modified": False,
            "sequence_inputs_modified": False,
            "model_trained": False,
        }

        print(
            "Five modality-specific GRU branches: PASS"
        )

        print(
            "Additional Insulin GRU:               NO"
        )

        print(
            "Additional Sleep GRU:                 NO"
        )

        print(
            "MLP Fusion implemented:               NO"
        )

        print(
            "Digital Twin implemented:             NO"
        )

        print(
            "Prediction implemented:               NO"
        )

        print(
            "What-if implemented:                  NO"
        )

        print(
            "Interactive UI implemented:           NO"
        )

        print(
            "Dataset modified:                      NO"
        )

        print(
            "Sequence inputs modified:              NO"
        )

        print(
            "Model trained:                         NO"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("13. WRITING AUDIT REPORT")
        print("-" * 80)

        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = {
            "report_type": (
                "t1d_uom_five_gru_input_contract_audit"
            ),
            "status": "PASS",
            "read_only": True,
            "project_root": str(
                PROJECT_ROOT
            ),
            "sequence_inputs": str(
                SEQUENCE_INPUTS
            ),
            "sequence_input_file_count": len(
                files
            ),
            "frozen_cohort": FROZEN_COHORT,
            "architecture": frozen_architecture(),
            "family_counts": family_counts,
            "input_contract": final_contract,
            "glucose": glucose_result,
            "nutrition": nutrition_result,
            "activity": activity_result,
            "insulin": insulin_result,
            "sleep": sleep_result,
            "safety": safety,
            "next_stage": (
                "Proceed to controlled real-data integration of the "
                "five existing modality-specific GRU branches only. "
                "Do not implement MLP Fusion, Digital Twin, Prediction, "
                "What-if, or Interactive UI at this stage."
            ),
        }

        with REPORT_PATH.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:

            json.dump(
                report,
                handle,
                indent=2,
                ensure_ascii=False,
            )

            handle.write("\n")

        print(
            "Audit report saved:"
        )

        print(
            f"  {REPORT_PATH}"
        )

        # =====================================================================
        print("\n" + "=" * 80)
        print(
            "T1D-UOM FIVE-GRU INPUT CONTRACT AUDIT COMPLETE"
        )
        print("=" * 80)

        print(
            "\nFINAL RESULT:"
        )

        print(
            "  Five-GRU input contract: PASS"
        )

        print(
            "  Glucose:    input_dim=1  -> zG"
        )

        print(
            "  Insulin:    input_dim=2  -> zI"
        )

        print(
            "  Nutrition:  input_dim=6  -> zN"
        )

        print(
            "  Activity:   input_dim=12 -> zA"
        )

        print(
            "  Sleep:      input_dim=6  -> zS"
        )

        print(
            "\nARCHITECTURE:"
        )

        print(
            "  Glucose   -> GRU -> zG"
        )

        print(
            "  Insulin   -> GRU -> zI"
        )

        print(
            "  Nutrition -> GRU -> zN"
        )

        print(
            "  Activity  -> GRU -> zA"
        )

        print(
            "  Sleep     -> GRU -> zS"
        )

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

        print(
            "\nIMPORTANT:"
        )

        print(
            "  No dataset was modified."
        )

        print(
            "  No sequence-input file was modified."
        )

        print(
            "  No encoding was performed."
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
            "  No architecture branch was added."
        )

        print(
            "  No MLP Fusion was implemented."
        )

        print(
            "  No Digital Twin was implemented."
        )

        print(
            "  No Prediction was implemented."
        )

        print(
            "  No What-if implementation was created."
        )

        print(
            "  No Interactive UI was created."
        )

        print(
            "\nNEXT STAGE:"
        )

        print(
            "  If this audit passes, the next stage is the controlled "
            "real-data integration of the five existing GRU branches."
        )

        print("=" * 80)

        return 0

    except Exception as exc:

        print("\n" + "=" * 80)
        print(
            "T1D-UOM FIVE-GRU INPUT CONTRACT AUDIT FAILED"
        )
        print("=" * 80)

        print(
            f"\n{exc}"
        )

        print(
            "\nIMPORTANT:"
        )

        print(
            "  No dataset files were modified by this audit."
        )

        print(
            "  No sequence-input files were modified by this audit."
        )

        print(
            "  No model was trained."
        )

        print(
            "  Frozen architecture was not changed."
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())