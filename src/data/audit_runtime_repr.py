"""
T1D-UOM Runtime Representation Contract Audit
==============================================

READ-ONLY AUDIT
---------------

This script audits the frozen sequence-input layer against the frozen
five-branch GRU runtime architecture.

It DOES NOT:

    - modify CSV files
    - rename files
    - delete files
    - modify values
    - impute
    - interpolate
    - resample
    - normalize
    - encode categories
    - construct event_type
    - create windows
    - create targets
    - train models
    - implement GRUs
    - implement MLP Fusion
    - implement the Digital Twin
    - implement Prediction
    - implement What-if
    - implement the UI

Frozen architecture
-------------------

    Glucose    -> GRU -> zG
    Insulin    -> GRU -> zI
    Nutrition  -> GRU -> zN
    Activity   -> GRU -> zA
    Sleep      -> GRU -> zS

    zG,zI,zN,zA,zS
            |
            v
       MLP Fusion
            |
            v
    Unified Patient State
            |
            v
       DIGITAL TWIN
            |
            v
    Prediction / What-if
            |
            v
      Interactive UI

Important filename convention
-----------------------------

Participant IDs occur in several filename forms:

    UoMActivity2301.csv
    UoMGlucose2301.csv
    UoMBasal2302.csv
    UoMBolus2301.csv
    UoMNutrition2301.csv
    UoMsleep2301.csv
    UoM2302sleeptime.csv

Therefore participant extraction MUST NOT assume:

    UoM + digits

The participant identifier is extracted from the final four-digit
participant token in the filename.

Sleep convention
----------------

    UoMsleep*.csv
        -> sleep_timeseries
        -> single Sleep GRU

    UoM*sleeptime.csv
        -> sleep_summary
        -> audited separately
        -> NOT a second Sleep GRU

Insulin convention
------------------

Physical files:

    basal -> basal_dose
    bolus -> bolus_dose

Runtime branch:

    Basal + Bolus
          |
          v
    ONE Insulin GRU
          |
          v
         zI

The runtime representation contains:

    dose
    event_type

where:

    basal event_type = 0
    bolus event_type = 1

This script audits the derivation CONTRACT only.
It does not construct event_type.

Exit codes
----------

    0 = PASS
    1 = AUDIT FAILURE
    2 = EXECUTION / CONFIGURATION ERROR
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "t1d_uom_v1.0.3_sequence_inputs"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"

REPORT_PATH = REPORT_DIR / "runtime_repr_audit.json"


# ============================================================================
# FROZEN EXPECTATIONS
# ============================================================================

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


# ============================================================================
# PHYSICAL SCHEMAS
# ============================================================================

PHYSICAL_SCHEMAS = {

    "glucose": {
        "timestamp": "bg_ts",
        "numeric": [
            "value",
        ],
        "categorical": [],
        "structured": [],
    },

    "nutrition": {
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
    },

    "activity": {
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
    },

    "basal_insulin": {
        "timestamp": "basal_ts",
        "numeric": [
            "basal_dose",
        ],
        "categorical": [
            "insulin_kind",
        ],
        "structured": [],
    },

    "bolus_insulin": {
        "timestamp": "bolus_ts",
        "numeric": [
            "bolus_dose",
        ],
        "categorical": [],
        "structured": [],
    },

    "sleep_summary": {
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
    },

    "sleep_timeseries": {
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
    },
}


# ============================================================================
# RUNTIME CONTRACT
# ============================================================================

RUNTIME_CONTRACT = {

    "glucose": {
        "branch": "Glucose",
        "latent": "zG",
        "input_dim": 1,
        "runtime_features": [
            "value",
        ],
        "physical_source": {
            "value": "glucose.value",
        },
        "requires_encoding": [],
    },

    "insulin": {
        "branch": "Insulin",
        "latent": "zI",
        "input_dim": 2,
        "runtime_features": [
            "dose",
            "event_type",
        ],
        "physical_sources": {
            "dose": [
                "basal_insulin.basal_dose",
                "bolus_insulin.bolus_dose",
            ],
            "event_type": [
                "DERIVED_FROM_SOURCE_MODALITY",
            ],
        },
        "event_type_policy": {
            "basal_insulin": 0,
            "bolus_insulin": 1,
        },
        "requires_encoding": [],
        "derived_features": [
            "event_type",
        ],
    },

    "nutrition": {
        "branch": "Nutrition",
        "latent": "zN",
        "input_dim": 6,
        "runtime_features": [
            "carbs_g",
            "prot_g",
            "fat_g",
            "fibre_g",
            "meal_type",
            "meal_tag",
        ],
        "physical_source": {
            "carbs_g": "nutrition.carbs_g",
            "prot_g": "nutrition.prot_g",
            "fat_g": "nutrition.fat_g",
            "fibre_g": "nutrition.fibre_g",
            "meal_type": "nutrition.meal_type",
            "meal_tag": "nutrition.meal_tag",
        },
        "requires_encoding": [
            "meal_type",
            "meal_tag",
        ],
    },

    "activity": {
        "branch": "Activity",
        "latent": "zA",
        "input_dim": 12,
        "runtime_features": [
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
        "physical_source": {
            "activity_type": "activity.activity_type",
            "active_Kcal": "activity.active_Kcal",
            "step_count": "activity.step_count",
            "distance_m": "activity.distance_m",
            "duration_s": "activity.duration_s",
            "active_time_s": "activity.active_time_s",
            "start_time_s": "activity.start_time_s",
            "start_time_offset_s": "activity.start_time_offset_s",
            "met": "activity.met",
            "intensity": "activity.intensity",
            "motion_intensity_mean": "activity.motion_intensity_mean",
            "motion_intensity_max": "activity.motion_intensity_max",
        },
        "requires_encoding": [
            "activity_type",
            "intensity",
        ],
    },

    "sleep": {
        "branch": "Sleep",
        "latent": "zS",
        "input_dim": 6,
        "runtime_features": [
            "step_count",
            "heart_rate",
            "current_activity_type_intensity",
            "stress_level_value",
            "sleep_level",
            "resting_heart_rate",
        ],
        "physical_source": {
            "step_count": "sleep_timeseries.step_count",
            "heart_rate": "sleep_timeseries.heart_rate",
            "current_activity_type_intensity":
                "sleep_timeseries.current_activity_type_intensity",
            "stress_level_value":
                "sleep_timeseries.stress_level_value",
            "sleep_level": "sleep_timeseries.sleep_level",
            "resting_heart_rate":
                "sleep_timeseries.resting_heart_rate",
        },
        "requires_encoding": [],
    },
}


# ============================================================================
# FILENAME IDENTIFICATION
# ============================================================================

# These patterns deliberately describe the frozen filename families.

FILENAME_RULES = [
    (
        "activity",
        re.compile(
            r"^UoMActivity(?P<participant>\d{4})\.csv$",
            re.IGNORECASE,
        ),
    ),
    (
        "glucose",
        re.compile(
            r"^UoMGlucose(?P<participant>\d{4})\.csv$",
            re.IGNORECASE,
        ),
    ),
    (
        "basal_insulin",
        re.compile(
            r"^UoMBasal(?P<participant>\d{4})\.csv$",
            re.IGNORECASE,
        ),
    ),
    (
        "bolus_insulin",
        re.compile(
            r"^UoMBolus(?P<participant>\d{4})\.csv$",
            re.IGNORECASE,
        ),
    ),
    (
        "nutrition",
        re.compile(
            r"^UoMNutrition(?P<participant>\d{4})\.csv$",
            re.IGNORECASE,
        ),
    ),
    (
        "sleep_timeseries",
        re.compile(
            r"^UoMsleep(?P<participant>\d{4})\.csv$",
            re.IGNORECASE,
        ),
    ),
    (
        "sleep_summary",
        re.compile(
            r"^UoM(?P<participant>\d{4})sleeptime\.csv$",
            re.IGNORECASE,
        ),
    ),
]


def identify_file(
    path: Path,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return:

        participant
        modality
        filename_rule

    The participant is normalized to UoM####.

    Returns (None, None, None) if the filename does not match
    any frozen filename family.
    """

    filename = path.name

    for modality, pattern in FILENAME_RULES:

        match = pattern.match(filename)

        if match:
            participant_number = match.group("participant")

            participant = f"UoM{participant_number}"

            return (
                participant,
                modality,
                pattern.pattern,
            )

    return None, None, None


# ============================================================================
# CSV HEADER AUDIT
# ============================================================================

def read_header(path: Path) -> List[str]:
    """
    Read only the header.

    No CSV contents are changed.
    """

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.reader(handle)

        try:
            row = next(reader)
        except StopIteration:
            return []

    return [
        str(column).strip()
        for column in row
    ]


def expected_columns(modality: str) -> List[str]:

    schema = PHYSICAL_SCHEMAS[modality]

    return [
        schema["timestamp"],
        *schema["numeric"],
        *schema["categorical"],
        *schema["structured"],
    ]


def compare_schema(
    header: List[str],
    modality: str,
) -> Dict:

    expected = expected_columns(modality)

    expected_set = set(expected)
    actual_set = set(header)

    missing = [
        column
        for column in expected
        if column not in actual_set
    ]

    unexpected = [
        column
        for column in header
        if column not in expected_set
    ]

    duplicate_columns = [
        column
        for column, count in Counter(header).items()
        if count > 1
    ]

    exact_schema = (
        not missing
        and not unexpected
        and not duplicate_columns
        and len(header) == len(expected)
    )

    return {
        "status": "PASS" if exact_schema else "FAIL",
        "expected_columns": expected,
        "actual_columns": header,
        "missing_columns": missing,
        "unexpected_columns": unexpected,
        "duplicate_columns": duplicate_columns,
    }


# ============================================================================
# RUNTIME CONTRACT VALIDATION
# ============================================================================

def validate_runtime_contract(
    modality: str,
    schema_result: Dict,
) -> Dict:

    result = {
        "status": "PASS",
        "branch": None,
        "latent": None,
        "input_dim": None,
        "runtime_features": [],
        "physical_sources": {},
        "requires_encoding": [],
        "derived_features": [],
        "notes": [],
    }

    if modality == "sleep_timeseries":

        contract = RUNTIME_CONTRACT["sleep"]

        result.update(
            {
                "branch": contract["branch"],
                "latent": contract["latent"],
                "input_dim": contract["input_dim"],
                "runtime_features":
                    contract["runtime_features"],
                "physical_sources":
                    contract["physical_source"],
                "requires_encoding":
                    contract["requires_encoding"],
            }
        )

        if schema_result["status"] != "PASS":
            result["status"] = "FAIL"

        result["notes"].append(
            "sleep_timeseries is the sole source for the Sleep GRU."
        )

        return result

    if modality in RUNTIME_CONTRACT:

        contract = RUNTIME_CONTRACT[modality]

        result.update(
            {
                "branch": contract["branch"],
                "latent": contract["latent"],
                "input_dim": contract["input_dim"],
                "runtime_features":
                    contract["runtime_features"],
                "physical_sources":
                    contract.get("physical_source", {}),
                "requires_encoding":
                    contract.get("requires_encoding", []),
                "derived_features":
                    contract.get("derived_features", []),
            }
        )

        if schema_result["status"] != "PASS":
            result["status"] = "FAIL"

        return result

    # Insulin physical files are intentionally mapped into one branch.
    if modality in {
        "basal_insulin",
        "bolus_insulin",
    }:

        contract = RUNTIME_CONTRACT["insulin"]

        result.update(
            {
                "branch": contract["branch"],
                "latent": contract["latent"],
                "input_dim": contract["input_dim"],
                "runtime_features":
                    contract["runtime_features"],
                "physical_sources":
                    contract["physical_sources"],
                "requires_encoding":
                    contract["requires_encoding"],
                "derived_features":
                    contract["derived_features"],
            }
        )

        if schema_result["status"] != "PASS":
            result["status"] = "FAIL"

        if modality == "basal_insulin":
            result["notes"].append(
                "Basal rows map to dose and event_type=0."
            )

        if modality == "bolus_insulin":
            result["notes"].append(
                "Bolus rows map to dose and event_type=1."
            )

        result["notes"].append(
            "Both physical insulin modalities feed ONE Insulin GRU."
        )

        result["notes"].append(
            "event_type is a future derived runtime field; "
            "this audit does not construct it."
        )

        return result

    # Sleep summary is intentionally not attached to a second GRU.
    if modality == "sleep_summary":

        result.update(
            {
                "branch": "SleepSummaryReference",
                "latent": None,
                "input_dim": None,
                "runtime_features": [],
                "physical_sources": {},
                "requires_encoding": [],
                "derived_features": [],
            }
        )

        if schema_result["status"] != "PASS":
            result["status"] = "FAIL"

        result["notes"].append(
            "Audited physical schema only."
        )

        result["notes"].append(
            "Not a second Sleep GRU."
        )

        return result

    result["status"] = "FAIL"
    result["notes"].append(
        "No runtime contract exists for this modality."
    )

    return result


# ============================================================================
# PARTICIPANT / MODALITY MATRIX
# ============================================================================

def build_coverage_matrix(
    records: List[Dict],
) -> Dict[str, Dict[str, List[str]]]:

    matrix = defaultdict(lambda: defaultdict(list))

    for record in records:

        participant = record["participant"]
        modality = record["physical_modality"]

        if participant and modality:
            matrix[participant][modality].append(
                record["relative_path"]
            )

    return matrix


def audit_coverage(
    matrix: Dict[str, Dict[str, List[str]]],
) -> Dict:

    expected_participants = set(EXPECTED_PARTICIPANTS)

    actual_participants = set(matrix.keys())

    missing_participants = sorted(
        expected_participants - actual_participants
    )

    unexpected_participants = sorted(
        actual_participants - expected_participants
    )

    duplicate_participant_modality = []

    for participant, modalities in matrix.items():

        for modality, paths in modalities.items():

            if len(paths) > 1:

                duplicate_participant_modality.append(
                    {
                        "participant": participant,
                        "modality": modality,
                        "files": sorted(paths),
                    }
                )

    modality_counts = Counter()

    for record in matrix.values():

        for modality, paths in record.items():

            modality_counts[modality] += len(paths)

    return {
        "status": (
            "PASS"
            if (
                not missing_participants
                and not unexpected_participants
                and not duplicate_participant_modality
            )
            else "FAIL"
        ),
        "actual_participants":
            sorted(actual_participants),
        "missing_participants":
            missing_participants,
        "unexpected_participants":
            unexpected_participants,
        "duplicate_participant_modality":
            duplicate_participant_modality,
        "modality_counts":
            dict(sorted(modality_counts.items())),
    }


# ============================================================================
# OUTPUT
# ============================================================================

def print_section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def write_report(report: Dict) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    print("=" * 80)
    print("T1D-UOM RUNTIME REPRESENTATION CONTRACT AUDIT")
    print("=" * 80)

    print()
    print("MODE: READ-ONLY")
    print()
    print("No dataset files will be modified.")
    print("No sequence-input files will be modified.")
    print("No transformations will be performed.")
    print("No categorical encoding will be performed.")
    print("No event_type will be constructed.")
    print("No windows will be created.")
    print("No targets will be created.")
    print("No models will be trained.")

    print()
    print(f"Project root:")
    print(f"  {PROJECT_ROOT}")

    print()
    print("Sequence-input directory:")
    print(f"  {SEQUENCE_INPUT_DIR}")

    # ----------------------------------------------------------------------
    # DIRECTORY
    # ----------------------------------------------------------------------

    print_section("1. DIRECTORY VALIDATION")

    if not SEQUENCE_INPUT_DIR.exists():

        print("FAIL: sequence-input directory does not exist.")

        return 1

    if not SEQUENCE_INPUT_DIR.is_dir():

        print("FAIL: sequence-input path is not a directory.")

        return 1

    print("PASS: sequence-input directory exists.")

    # ----------------------------------------------------------------------
    # INVENTORY
    # ----------------------------------------------------------------------

    print_section("2. FILE INVENTORY")

    files = sorted(
        SEQUENCE_INPUT_DIR.rglob("*.csv"),
        key=lambda path:
            str(
                path.relative_to(
                    SEQUENCE_INPUT_DIR
                )
            ).lower(),
    )

    print(
        f"CSV files discovered: "
        f"{len(files)}"
    )

    inventory_pass = (
        len(files) == EXPECTED_FILE_COUNT
    )

    print(
        "Inventory check: "
        + ("PASS" if inventory_pass else "FAIL")
    )

    # ----------------------------------------------------------------------
    # FILE AUDIT
    # ----------------------------------------------------------------------

    print_section("3. FILE IDENTIFICATION + PHYSICAL SCHEMA")

    records = []

    classification_failures = []
    schema_failures = []
    runtime_failures = []

    for index, path in enumerate(files, start=1):

        relative_path = str(
            path.relative_to(
                SEQUENCE_INPUT_DIR
            )
        )

        participant, modality, filename_rule = identify_file(path)

        record = {
            "relative_path": relative_path,
            "filename": path.name,
            "participant": participant,
            "physical_modality": modality,
            "filename_rule": filename_rule,
            "identification_status": None,
            "schema_status": None,
            "schema": None,
            "runtime": None,
        }

        if participant is None or modality is None:

            record["identification_status"] = "FAIL"

            classification_failures.append(record)

            print(
                f"[{index:03d}/{len(files):03d}] "
                f"{relative_path} | "
                f"IDENTIFICATION FAIL"
            )

            records.append(record)

            continue

        record["identification_status"] = "PASS"

        header = read_header(path)

        schema_result = compare_schema(
            header,
            modality,
        )

        record["schema_status"] = schema_result["status"]
        record["schema"] = schema_result

        if schema_result["status"] == "FAIL":

            schema_failures.append(record)

        runtime_result = validate_runtime_contract(
            modality,
            schema_result,
        )

        record["runtime"] = runtime_result

        if runtime_result["status"] == "FAIL":

            runtime_failures.append(record)

        overall = (
            "PASS"
            if (
                record["identification_status"] == "PASS"
                and record["schema_status"] == "PASS"
                and runtime_result["status"] == "PASS"
            )
            else "FAIL"
        )

        print(
            f"[{index:03d}/{len(files):03d}] "
            f"{relative_path} | "
            f"{participant} | "
            f"{modality} | "
            f"{overall}"
        )

        records.append(record)

    # ----------------------------------------------------------------------
    # COVERAGE
    # ----------------------------------------------------------------------

    print_section("4. PARTICIPANT COVERAGE")

    coverage_matrix = build_coverage_matrix(
        records
    )

    coverage = audit_coverage(
        coverage_matrix
    )

    print(
        f"Participants discovered: "
        f"{len(coverage['actual_participants'])}"
    )

    for participant in coverage["actual_participants"]:
        print(f"  {participant}")

    if coverage["missing_participants"]:

        print()
        print("Missing participants:")

        for participant in coverage["missing_participants"]:
            print(f"  {participant}")

    if coverage["unexpected_participants"]:

        print()
        print("Unexpected participants:")

        for participant in coverage["unexpected_participants"]:
            print(f"  {participant}")

    print()
    print(
        "Participant coverage: "
        + coverage["status"]
    )

    # ----------------------------------------------------------------------
    # MODALITY COUNTS
    # ----------------------------------------------------------------------

    print_section("5. MODALITY COUNTS")

    modality_counts = coverage["modality_counts"]

    modality_count_pass = True

    for modality, expected_count in (
        EXPECTED_MODALITY_COUNTS.items()
    ):

        actual_count = modality_counts.get(
            modality,
            0,
        )

        status = (
            "PASS"
            if actual_count == expected_count
            else "FAIL"
        )

        if status == "FAIL":
            modality_count_pass = False

        print(
            f"{modality:<18} "
            f"actual={actual_count:<3} "
            f"expected={expected_count:<3} "
            f"{status}"
        )

    print()
    print(
        "Modality count validation: "
        + (
            "PASS"
            if modality_count_pass
            else "FAIL"
        )
    )

    # ----------------------------------------------------------------------
    # PARTICIPANT/MODALITY MATRIX
    # ----------------------------------------------------------------------

    print_section("6. PARTICIPANT × MODALITY COVERAGE")

    for participant in EXPECTED_PARTICIPANTS:

        modalities = coverage_matrix.get(
            participant,
            {},
        )

        print()
        print(participant)

        for modality in sorted(
            EXPECTED_MODALITY_COUNTS.keys()
        ):

            paths = modalities.get(
                modality,
                [],
            )

            if paths:

                print(
                    f"  {modality:<18} "
                    f"{len(paths)} file"
                )

            else:

                print(
                    f"  {modality:<18} "
                    f"0 files"
                )

    # ----------------------------------------------------------------------
    # FROZEN ARCHITECTURE
    # ----------------------------------------------------------------------

    print_section("7. FROZEN FIVE-GRU ARCHITECTURE")

    print("Glucose    -> GRU -> zG")
    print("Insulin    -> GRU -> zI")
    print("Nutrition  -> GRU -> zN")
    print("Activity   -> GRU -> zA")
    print("Sleep      -> GRU -> zS")
    print()
    print("zG,zI,zN,zA,zS -> MLP Fusion")
    print("MLP Fusion -> Unified Patient State")
    print("Unified Patient State -> DIGITAL TWIN")
    print("DIGITAL TWIN -> Prediction / What-if")
    print("Prediction / What-if -> Interactive UI")

    # ----------------------------------------------------------------------
    # RUNTIME CONTRACT
    # ----------------------------------------------------------------------

    print_section("8. RUNTIME REPRESENTATION CONTRACT")

    for branch_name, contract in RUNTIME_CONTRACT.items():

        print()
        print(
            f"{contract['branch']} "
            f"-> {contract['latent']}"
        )

        print(
            f"  input_dim = "
            f"{contract['input_dim']}"
        )

        print("  runtime features:")

        for feature in contract["runtime_features"]:
            print(
                f"    - {feature}"
            )

        encoding = contract.get(
            "requires_encoding",
            [],
        )

        if encoding:

            print("  future categorical encoding required:")

            for feature in encoding:
                print(
                    f"    - {feature}"
                )

    print()
    print("Insulin:")
    print("  basal + bolus -> ONE Insulin GRU -> zI")
    print("  basal event_type = 0")
    print("  bolus event_type = 1")
    print("  event_type is DERIVED later; not constructed here.")

    print()
    print("Sleep:")
    print("  sleep_timeseries -> ONE Sleep GRU -> zS")
    print("  sleep_summary -> audited separately")
    print("  second Sleep GRU -> NO")

    # ----------------------------------------------------------------------
    # SLEEP CHECK
    # ----------------------------------------------------------------------

    print_section("9. SLEEP REPRESENTATION SAFETY CHECK")

    sleep_summary_records = [
        record
        for record in records
        if record["physical_modality"]
        == "sleep_summary"
    ]

    sleep_timeseries_records = [
        record
        for record in records
        if record["physical_modality"]
        == "sleep_timeseries"
    ]

    sleep_summary_pass = all(
        record["schema_status"] == "PASS"
        for record in sleep_summary_records
    )

    sleep_timeseries_pass = all(
        record["schema_status"] == "PASS"
        for record in sleep_timeseries_records
    )

    sleep_pass = (
        len(sleep_summary_records) == 11
        and len(sleep_timeseries_records) == 11
        and sleep_summary_pass
        and sleep_timeseries_pass
    )

    print(
        f"sleep_summary files: "
        f"{len(sleep_summary_records)}"
    )

    print(
        f"sleep_timeseries files: "
        f"{len(sleep_timeseries_records)}"
    )

    print(
        "Sleep representation safety: "
        + (
            "PASS"
            if sleep_pass
            else "FAIL"
        )
    )

    # ----------------------------------------------------------------------
    # CLASSIFICATION DIAGNOSTICS
    # ----------------------------------------------------------------------

    print_section("10. DIAGNOSTIC COUNTS")

    print(
        f"Identification failures: "
        f"{len(classification_failures)}"
    )

    print(
        f"Physical schema failures: "
        f"{len(schema_failures)}"
    )

    print(
        f"Runtime contract failures: "
        f"{len(runtime_failures)}"
    )

    # ----------------------------------------------------------------------
    # FINAL STATUS
    # ----------------------------------------------------------------------

    overall_pass = (
        inventory_pass
        and not classification_failures
        and not schema_failures
        and not runtime_failures
        and coverage["status"] == "PASS"
        and modality_count_pass
        and sleep_pass
    )

    # ----------------------------------------------------------------------
    # REPORT
    # ----------------------------------------------------------------------

    report = {
        "audit_name":
            "T1D-UOM Runtime Representation Contract Audit",

        "audit_version":
            "2.0.0",

        "read_only":
            True,

        "project_root":
            str(PROJECT_ROOT),

        "sequence_input_directory":
            str(SEQUENCE_INPUT_DIR),

        "report_path":
            str(REPORT_PATH),

        "frozen_expectations": {
            "file_count":
                EXPECTED_FILE_COUNT,

            "participants":
                EXPECTED_PARTICIPANTS,

            "modality_counts":
                EXPECTED_MODALITY_COUNTS,
        },

        "inventory": {
            "files_discovered":
                len(files),

            "files_audited":
                len(records),

            "status":
                "PASS"
                if inventory_pass
                else "FAIL",
        },

        "identification": {
            "status":
                "PASS"
                if not classification_failures
                else "FAIL",

            "failure_count":
                len(classification_failures),

            "failures":
                classification_failures,
        },

        "coverage":
            coverage,

        "physical_schema": {
            "status":
                "PASS"
                if not schema_failures
                else "FAIL",

            "failure_count":
                len(schema_failures),

            "failures":
                schema_failures,
        },

        "runtime_contract": {
            "status":
                "PASS"
                if not runtime_failures
                else "FAIL",

            "failure_count":
                len(runtime_failures),

            "failures":
                runtime_failures,
        },

        "sleep_contract": {
            "sleep_summary_filename":
                "UoM####sleeptime.csv",

            "sleep_summary_physical_schema":
                "sleep_summary",

            "sleep_timeseries_filename":
                "UoMsleep####.csv",

            "sleep_timeseries_physical_schema":
                "sleep_timeseries",

            "single_sleep_gru_source":
                "sleep_timeseries",

            "additional_sleep_gru":
                False,

            "status":
                "PASS"
                if sleep_pass
                else "FAIL",
        },

        "insulin_contract": {
            "single_branch":
                True,

            "physical_sources": [
                "basal_insulin",
                "bolus_insulin",
            ],

            "runtime_features": [
                "dose",
                "event_type",
            ],

            "event_type_policy": {
                "basal_insulin": 0,
                "bolus_insulin": 1,
            },

            "event_type_constructed_by_audit":
                False,
        },

        "architecture": {
            "glucose":
                "GRU -> zG",

            "insulin":
                "GRU -> zI",

            "nutrition":
                "GRU -> zN",

            "activity":
                "GRU -> zA",

            "sleep":
                "GRU -> zS",

            "fusion":
                "zG,zI,zN,zA,zS -> MLP Fusion",

            "unified_state":
                "MLP Fusion -> Unified Patient State",

            "digital_twin":
                "Unified Patient State -> DIGITAL TWIN",

            "prediction":
                "DIGITAL TWIN -> Prediction / What-if",

            "ui":
                "Prediction / What-if -> Interactive UI",
        },

        "summary": {
            "inventory_pass":
                inventory_pass,

            "identification_pass":
                not classification_failures,

            "cohort_pass":
                coverage["status"] == "PASS",

            "modality_count_pass":
                modality_count_pass,

            "physical_schema_pass":
                not schema_failures,

            "runtime_contract_pass":
                not runtime_failures,

            "sleep_contract_pass":
                sleep_pass,

            "overall_pass":
                overall_pass,
        },

        "scope_controls": {
            "dataset_modified": False,
            "sequence_inputs_modified": False,
            "values_transformed": False,
            "rows_deleted": False,
            "rows_added": False,
            "resampling_performed": False,
            "interpolation_performed": False,
            "imputation_performed": False,
            "normalization_performed": False,
            "feature_engineering_performed": False,
            "categorical_encoding_performed": False,
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

        "files":
            records,
    }

    write_report(report)

    # ----------------------------------------------------------------------
    # FINAL TERMINAL SUMMARY
    # ----------------------------------------------------------------------

    print_section("11. FINAL RESULT")

    print(
        f"Files discovered: "
        f"{len(files)}"
    )

    print(
        f"Files audited: "
        f"{len(records)}"
    )

    print(
        "Identification: "
        + (
            "PASS"
            if not classification_failures
            else "FAIL"
        )
    )

    print(
        "Physical schemas: "
        + (
            "PASS"
            if not schema_failures
            else "FAIL"
        )
    )

    print(
        "Participant coverage: "
        + coverage["status"]
    )

    print(
        "Modality counts: "
        + (
            "PASS"
            if modality_count_pass
            else "FAIL"
        )
    )

    print(
        "Runtime contract: "
        + (
            "PASS"
            if not runtime_failures
            else "FAIL"
        )
    )

    print(
        "Sleep contract: "
        + (
            "PASS"
            if sleep_pass
            else "FAIL"
        )
    )

    print()
    print(
        f"JSON report:"
    )

    print(
        f"  {REPORT_PATH}"
    )

    print()

    if overall_pass:

        print("=" * 80)
        print("FINAL RESULT: PASS")
        print("=" * 80)

        print()
        print("The frozen sequence-input layer passes the runtime")
        print("representation contract audit.")

        print()
        print("No data was modified.")
        print("No preprocessing was performed.")
        print("No event_type was constructed.")
        print("No windows were created.")
        print("No model was trained.")

        return 0

    print("=" * 80)
    print("FINAL RESULT: FAIL")
    print("=" * 80)

    print()
    print("The audit detected one or more genuine contract failures.")

    if classification_failures:

        print()
        print(
            "First identification failure:"
        )

        print(
            f"  {classification_failures[0]['relative_path']}"
        )

    if schema_failures:

        print()
        print(
            "First schema failure:"
        )

        print(
            f"  {schema_failures[0]['relative_path']}"
        )

    if runtime_failures:

        print()
        print(
            "First runtime-contract failure:"
        )

        print(
            f"  {runtime_failures[0]['relative_path']}"
        )

    print()
    print("No dataset files were modified.")

    return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:

        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:

        print()
        print(
            "Audit interrupted by user."
        )

        raise SystemExit(130)

    except Exception as exc:

        print()
        print("=" * 80)
        print("AUDIT EXECUTION ERROR")
        print("=" * 80)
        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        print(
            "No intentional dataset modification was performed."
        )

        raise SystemExit(2)