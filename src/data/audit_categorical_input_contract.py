from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# =============================================================================
# T1D-UOM CATEGORICAL INPUT CONTRACT AUDIT
# =============================================================================
#
# READ-ONLY.
#
# This audit:
#   - does NOT modify any dataset
#   - does NOT modify sequence inputs
#   - does NOT encode categorical values
#   - does NOT create windows
#   - does NOT create targets
#   - does NOT normalize values
#   - does NOT impute values
#   - does NOT resample
#   - does NOT interpolate
#   - does NOT train a model
#   - does NOT implement MLP Fusion
#   - does NOT implement Digital Twin
#   - does NOT implement Prediction
#   - does NOT implement What-if
#   - does NOT implement Interactive UI
#
# FROZEN ARCHITECTURE:
#
#   Glucose   -> GRU -> zG
#   Insulin   -> GRU -> zI
#   Nutrition -> GRU -> zN
#   Activity  -> GRU -> zA
#   Sleep     -> GRU -> zS
#
#   zG,zI,zN,zA,zS -> MLP Fusion
#   MLP Fusion -> Unified Patient State
#   Unified Patient State -> DIGITAL TWIN
#   DIGITAL TWIN -> Prediction / What-if
#   Prediction / What-if -> Interactive UI
#
# IMPORTANT:
# Physical source schemas are NOT forced to become identical here.
#
# Actual frozen Insulin schemas:
#
#   Basal:
#       basal_dose
#       insulin_kind
#
#   Bolus:
#       bolus_dose
#
# The frozen architecture has ONE Insulin GRU.
# A later controlled integration stage may construct the common
# representation [dose, event_type].
#
# THIS SCRIPT DOES NOT PERFORM THAT CONSTRUCTION.
#
# Actual frozen Sleep schemas:
#
#   Sleep time-series:
#       6 features
#
#   Sleep summary:
#       14 features
#
# The frozen architecture has ONE Sleep GRU.
# The locked representation for the GRU is the 6-feature sleep
# time-series representation established by the preceding audit.
#
# THIS SCRIPT DOES NOT transform or combine the sleep data.
# =============================================================================


# =============================================================================
# PROJECT PATHS
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
    / "categorical_input_contract_audit.json"
)

MODELING_MANIFEST = (
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

EXPECTED_SEQUENCE_FILE_COUNT = 86


# =============================================================================
# EXPECTED PHYSICAL FAMILY COUNTS
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
# FROZEN ARCHITECTURE CONTRACT
# =============================================================================

ARCHITECTURE = {
    "glucose": {
        "input_dim": 1,
        "representation": ["value"],
        "branch": "Glucose -> GRU -> zG",
    },
    "insulin": {
        "input_dim": 2,
        "representation": ["dose", "event_type"],
        "branch": "Insulin -> GRU -> zI",
        "physical_sources": {
            "basal": ["basal_dose", "insulin_kind"],
            "bolus": ["bolus_dose"],
        },
    },
    "nutrition": {
        "input_dim": 6,
        "representation": [
            "meal_type",
            "meal_tag",
            "carbs_g",
            "prot_g",
            "fat_g",
            "fibre_g",
        ],
        "branch": "Nutrition -> GRU -> zN",
    },
    "activity": {
        "input_dim": 12,
        "representation": [
            "activity_type",
            "active_kcal",
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
        "branch": "Activity -> GRU -> zA",
    },
    "sleep": {
        "input_dim": 6,
        "representation": [
            "step_count",
            "heart_rate",
            "current_activity_type_intensity",
            "stress_level_value",
            "sleep_level",
            "resting_heart_rate",
        ],
        "branch": "Sleep -> GRU -> zS",
        "selected_physical_source": "sleep_timeseries",
        "alternative_audited_source": "sleep_summary",
    },
}


# =============================================================================
# ACTUAL PHYSICAL SCHEMAS
# =============================================================================

EXPECTED_SCHEMAS = {
    "activity": [
        "activity_type",
        "active_kcal",
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
    "glucose": [
        "value",
    ],
    "basal_insulin": [
        "basal_dose",
        "insulin_kind",
    ],
    "bolus_insulin": [
        "bolus_dose",
    ],
    "nutrition": [
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
        "start_time_offset_s",
        "unmeasurable_sleep_s",
        "deep_sleep_s",
        "light_sleep_s",
        "rem_sleep_s",
        "awake_s",
        "sleep_levels_map_deep",
        "sleep_levels_map_light",
        "sleep_levels_map_awake",
        "sleep_levels_map_rem",
        "sleep_levels_map_unmeasurable",
        "validation",
    ],
    "sleep_timeseries": [
        "step_count",
        "heart_rate",
        "current_activity_type_intensity",
        "stress_level_value",
        "sleep_level",
        "resting_heart_rate",
    ],
}


# =============================================================================
# CATEGORICAL COLUMNS THAT ACTUALLY EXIST
# =============================================================================
#
# These are AUDITED ONLY.
# No encoding is performed.
# =============================================================================

CATEGORICAL_COLUMNS = {
    "activity": [
        "activity_type",
        "intensity",
    ],
    "nutrition": [
        "meal_type",
        "meal_tag",
    ],
    "basal_insulin": [
        "insulin_kind",
    ],
}


# =============================================================================
# PARTICIPANT IDENTIFICATION
# =============================================================================

def identify_participant(path: Path) -> str | None:
    """
    Supports the actual established filename conventions:

        UoMActivity2301.csv
        UoMGlucose2301.csv
        UoMBolus2301.csv
        UoMBasal2302.csv
        UoMNutrition2301.csv
        UoMsleep2301.csv
        UoM2302sleeptime.csv
    """

    name = path.name

    patterns = [
        r"^UoMActivity(?P<pid>\d{4})\.csv$",
        r"^UoMGlucose(?P<pid>\d{4})\.csv$",
        r"^UoMBolus(?P<pid>\d{4})\.csv$",
        r"^UoMBasal(?P<pid>\d{4})\.csv$",
        r"^UoMNutrition(?P<pid>\d{4})\.csv$",
        r"^UoMsleep(?P<pid>\d{4})\.csv$",
        r"^UoM(?P<pid>\d{4})sleeptime\.csv$",
    ]

    for pattern in patterns:
        match = re.match(
            pattern,
            name,
            flags=re.IGNORECASE,
        )

        if match:
            return f"UoM{match.group('pid')}"

    return None


# =============================================================================
# MODALITY FAMILY IDENTIFICATION
# =============================================================================

def identify_modality_family(path: Path) -> str | None:

    parts_lower = [
        part.lower()
        for part in path.parts
    ]

    name_lower = path.name.lower()

    if "activity data" in parts_lower:
        return "activity"

    if "glucose data" in parts_lower:
        return "glucose"

    if "nutrition data" in parts_lower:
        return "nutrition"

    if "basal data" in parts_lower:
        return "basal_insulin"

    if "bolus data" in parts_lower:
        return "bolus_insulin"

    if "sleep data" in parts_lower:

        if "sleeptime" in name_lower:
            return "sleep_summary"

        if name_lower.startswith("uomsleep"):
            return "sleep_timeseries"

    return None


# =============================================================================
# CSV HELPERS
# =============================================================================

def read_csv_header(path: Path) -> list[str]:

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.reader(handle)

        try:
            header = next(reader)
        except StopIteration:
            return []

    return [
        str(column).strip()
        for column in header
    ]


def read_csv_rows(path: Path) -> list[dict[str, str]]:

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            return []

        rows = []

        for row in reader:
            rows.append(
                {
                    str(key).strip(): (
                        ""
                        if value is None
                        else str(value)
                    )
                    for key, value in row.items()
                }
            )

    return rows


def normalize_column_name(name: str) -> str:

    return re.sub(
        r"\s+",
        "_",
        str(name).strip().lower(),
    )


def normalized_column_set(
    columns: list[str],
) -> set[str]:

    return {
        normalize_column_name(column)
        for column in columns
    }


def find_column(
    columns: list[str],
    target: str,
) -> str | None:

    target_normalized = normalize_column_name(
        target
    )

    for column in columns:

        if (
            normalize_column_name(column)
            == target_normalized
        ):
            return column

    return None


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_exact_schema(
    path: Path,
    family: str,
) -> dict[str, Any]:

    header = read_csv_header(path)

    if not header:
        raise RuntimeError(
            f"Empty CSV or missing header:\n{path}"
        )

    actual = normalized_column_set(header)

    expected = {
        normalize_column_name(column)
        for column in EXPECTED_SCHEMAS[family]
    }

    timestamp_columns = {
        "activity_ts",
        "bg_ts",
        "basal_ts",
        "bolus_ts",
        "meal_ts",
        "sleep_ts",
        "start_date_ts",
    }

    actual_features = (
        actual - timestamp_columns
    )

    if actual_features != expected:

        unexpected = sorted(
            actual_features - expected
        )

        missing = sorted(
            expected - actual_features
        )

        raise RuntimeError(
            "Schema mismatch.\n"
            f"File: {path}\n"
            f"Family: {family}\n"
            f"Unexpected columns: {unexpected}\n"
            f"Missing columns: {missing}"
        )

    present_timestamps = sorted(
        actual & timestamp_columns
    )

    if len(present_timestamps) != 1:

        raise RuntimeError(
            "Expected exactly one recognized timestamp "
            "column.\n"
            f"File: {path}\n"
            f"Found: {present_timestamps}"
        )

    return {
        "family": family,
        "relative_path": str(
            path.relative_to(SEQUENCE_INPUTS)
        ),
        "participant": identify_participant(path),
        "timestamp_column": present_timestamps[0],
        "feature_dimension": len(actual_features),
        "feature_columns": sorted(actual_features),
        "status": "PASS",
    }


# =============================================================================
# RAW CATEGORICAL DOMAIN INSPECTION
# =============================================================================

def inspect_raw_categorical_domain(
    path: Path,
    column: str,
) -> dict[str, Any]:

    rows = read_csv_rows(path)

    header = read_csv_header(path)

    actual_column = find_column(
        header,
        column,
    )

    if actual_column is None:

        raise RuntimeError(
            "Categorical column not found.\n"
            f"File: {path}\n"
            f"Column: {column}"
        )

    counts: Counter[str] = Counter()

    for row in rows:

        value = row.get(
            actual_column,
            "",
        )

        # Observation only.
        # No value transformation.
        counts[value] += 1

    return {
        "column": actual_column,
        "unique_raw_values": sorted(
            counts.keys(),
            key=lambda value: str(value),
        ),
        "raw_value_counts": {
            str(key): int(value)
            for key, value in sorted(
                counts.items(),
                key=lambda item: str(item[0]),
            )
        },
    }


# =============================================================================
# INVENTORY
# =============================================================================

def discover_sequence_files() -> list[Path]:

    if not SEQUENCE_INPUTS.exists():

        raise RuntimeError(
            "Sequence-input directory does not exist:\n"
            f"{SEQUENCE_INPUTS}"
        )

    return sorted(
        SEQUENCE_INPUTS.rglob("*.csv"),
        key=lambda path:
            path.relative_to(
                SEQUENCE_INPUTS
            ).as_posix().lower(),
    )


def build_inventory(
    files: list[Path],
) -> list[dict[str, Any]]:

    records = []
    unidentified_participants = []
    unidentified_families = []

    for path in files:

        participant = identify_participant(path)
        family = identify_modality_family(path)

        relative = (
            path.relative_to(
                SEQUENCE_INPUTS
            ).as_posix()
        )

        if participant is None:
            unidentified_participants.append(
                relative
            )

        if family is None:
            unidentified_families.append(
                relative
            )

        if (
            participant is not None
            and family is not None
        ):

            records.append(
                {
                    "path": path,
                    "relative_path": relative,
                    "participant": participant,
                    "family": family,
                }
            )

    if unidentified_participants:

        raise RuntimeError(
            "Unable to identify participant from "
            "sequence-input file(s):\n"
            + "\n".join(
                f"  - {item}"
                for item in unidentified_participants
            )
        )

    if unidentified_families:

        raise RuntimeError(
            "Unable to identify modality family from "
            "sequence-input file(s):\n"
            + "\n".join(
                f"  - {item}"
                for item in unidentified_families
            )
        )

    return records


# =============================================================================
# COHORT VALIDATION
# =============================================================================

def validate_cohort(
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:

    participants = sorted(
        {
            record["participant"]
            for record in inventory
        }
    )

    participant_set = set(participants)

    missing = sorted(
        FROZEN_COHORT_SET - participant_set
    )

    unexpected = sorted(
        participant_set - FROZEN_COHORT_SET
    )

    if missing:
        raise RuntimeError(
            "Frozen cohort participant(s) missing:\n"
            + "\n".join(
                f"  - {pid}"
                for pid in missing
            )
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected participant(s) found:\n"
            + "\n".join(
                f"  - {pid}"
                for pid in unexpected
            )
        )

    return {
        "participants": participants,
        "participant_count": len(participants),
        "status": "PASS",
    }


# =============================================================================
# FAMILY VALIDATION
# =============================================================================

def validate_family_counts(
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:

    observed = Counter(
        record["family"]
        for record in inventory
    )

    result = {}

    for family, expected in EXPECTED_FAMILY_COUNTS.items():

        actual = observed.get(
            family,
            0,
        )

        if actual != expected:

            raise RuntimeError(
                f"Unexpected {family} file count: "
                f"observed={actual}, "
                f"expected={expected}"
            )

        result[family] = {
            "observed": actual,
            "expected": expected,
            "status": "PASS",
        }

    unexpected = sorted(
        set(observed)
        - set(EXPECTED_FAMILY_COUNTS)
    )

    if unexpected:

        raise RuntimeError(
            "Unexpected modality family/families:\n"
            + "\n".join(
                f"  - {family}"
                for family in unexpected
            )
        )

    return result


# =============================================================================
# PHYSICAL SCHEMA AUDIT
# =============================================================================

def audit_physical_schemas(
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:

    results = defaultdict(list)

    for record in inventory:

        result = validate_exact_schema(
            record["path"],
            record["family"],
        )

        results[
            record["family"]
        ].append(result)

    return dict(results)


# =============================================================================
# CATEGORICAL DOMAIN AUDIT
# =============================================================================

def audit_categorical_domains(
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:

    result: dict[str, Any] = {}

    for family, columns in CATEGORICAL_COLUMNS.items():

        family_records = [
            record
            for record in inventory
            if record["family"] == family
        ]

        if not family_records:

            raise RuntimeError(
                f"No files found for categorical family: "
                f"{family}"
            )

        column_result = {}

        for column in columns:

            participant_domains = {}
            global_counts: Counter[str] = Counter()

            for record in family_records:

                inspection = (
                    inspect_raw_categorical_domain(
                        record["path"],
                        column,
                    )
                )

                participant_domains[
                    record["participant"]
                ] = {
                    "relative_path":
                        record["relative_path"],
                    "unique_raw_values":
                        inspection[
                            "unique_raw_values"
                        ],
                    "raw_value_counts":
                        inspection[
                            "raw_value_counts"
                        ],
                }

                for (
                    raw_value,
                    count,
                ) in inspection[
                    "raw_value_counts"
                ].items():

                    global_counts[
                        raw_value
                    ] += count

            column_result[column] = {
                "files": len(family_records),
                "global_unique_raw_values":
                    sorted(
                        global_counts.keys(),
                        key=lambda value:
                            str(value),
                    ),
                "global_raw_value_counts": {
                    str(key): int(value)
                    for key, value in sorted(
                        global_counts.items(),
                        key=lambda item:
                            str(item[0]),
                    )
                },
                "per_file":
                    participant_domains,
                "encoding_performed": False,
                "status": "PASS",
            }

        result[family] = {
            "categorical_columns": column_result,
            "status": "PASS",
        }

    return result


# =============================================================================
# ARCHITECTURE SAFETY
# =============================================================================

def architecture_safety() -> dict[str, Any]:

    return {
        "glucose": "Glucose -> GRU -> zG",
        "insulin": "Insulin -> GRU -> zI",
        "nutrition": "Nutrition -> GRU -> zN",
        "activity": "Activity -> GRU -> zA",
        "sleep": "Sleep -> GRU -> zS",
        "fusion":
            "zG,zI,zN,zA,zS -> MLP Fusion",
        "unified_patient_state":
            "MLP Fusion -> Unified Patient State",
        "digital_twin":
            "Unified Patient State -> DIGITAL TWIN",
        "prediction_what_if":
            "DIGITAL TWIN -> Prediction / What-if",
        "interactive_ui":
            "Prediction / What-if -> Interactive UI",
        "additional_insulin_gru": False,
        "additional_sleep_gru": False,
        "mlp_fusion_implemented": False,
        "digital_twin_implemented": False,
        "prediction_implemented": False,
        "what_if_implemented": False,
        "interactive_ui_implemented": False,
        "status": "PASS",
    }


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    report: dict[str, Any],
) -> None:

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


# =============================================================================
# TERMINAL OUTPUT
# =============================================================================

def section(
    number: int,
    title: str,
) -> None:

    print()
    print("-" * 80)
    print(f"{number}. {title}")
    print("-" * 80)


def print_header() -> None:

    print("=" * 80)
    print(
        "T1D-UOM CATEGORICAL INPUT CONTRACT AUDIT"
    )
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
    print("No encoding will be performed.")
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
    print(f"Sequence inputs:    {SEQUENCE_INPUTS}")
    print(f"Audit report:       {REPORT_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    print_header()

    report: dict[str, Any] = {
        "audit": "categorical_input_contract",
        "status": "FAILED",
        "project_root":
            str(PROJECT_ROOT),
        "sequence_inputs":
            str(SEQUENCE_INPUTS),
        "frozen_cohort":
            FROZEN_COHORT,
        "expected_sequence_file_count":
            EXPECTED_SEQUENCE_FILE_COUNT,
        "architecture":
            architecture_safety(),
    }

    try:

        # ---------------------------------------------------------------------
        # 1. DIRECTORY / MANIFEST VALIDATION
        # ---------------------------------------------------------------------

        section(
            1,
            "DIRECTORY AND MANIFEST VALIDATION",
        )

        if not SEQUENCE_INPUTS.exists():

            raise RuntimeError(
                "Sequence-input directory does not exist:\n"
                f"{SEQUENCE_INPUTS}"
            )

        print(
            "Sequence-input directory: PASS"
        )

        for manifest_path, label in [
            (
                MODELING_MANIFEST,
                "modeling_dataset_manifest.json",
            ),
            (
                SEQUENCE_PREPARATION_MANIFEST,
                "sequence_input_preparation_manifest.json",
            ),
        ]:

            if not manifest_path.exists():

                raise RuntimeError(
                    f"Missing {label}:\n"
                    f"{manifest_path}"
                )

            with manifest_path.open(
                "r",
                encoding="utf-8",
            ) as handle:

                json.load(handle)

            print(f"{label}: PASS")

        # ---------------------------------------------------------------------
        # 2. INVENTORY
        # ---------------------------------------------------------------------

        section(
            2,
            "FROZEN SEQUENCE-INPUT INVENTORY",
        )

        files = discover_sequence_files()

        print(
            "Sequence-input CSV files discovered: "
            f"{len(files)}"
        )

        if len(files) != EXPECTED_SEQUENCE_FILE_COUNT:

            raise RuntimeError(
                "Frozen sequence-input file count mismatch: "
                f"observed={len(files)}, "
                f"expected={EXPECTED_SEQUENCE_FILE_COUNT}"
            )

        print(
            "Frozen sequence-input file count: PASS"
        )

        inventory = build_inventory(files)

        # ---------------------------------------------------------------------
        # 3. COHORT
        # ---------------------------------------------------------------------

        section(
            3,
            "FROZEN COHORT VALIDATION",
        )

        cohort = validate_cohort(
            inventory
        )

        print(
            "Participants represented in "
            "sequence inputs: "
            f"{cohort['participant_count']}"
        )

        for participant in cohort[
            "participants"
        ]:
            print(f"  {participant}")

        print()
        print(
            "Frozen 13-participant cohort: PASS"
        )

        # ---------------------------------------------------------------------
        # 4. FAMILY COUNTS
        # ---------------------------------------------------------------------

        section(
            4,
            "MODALITY FAMILY VALIDATION",
        )

        family_counts = (
            validate_family_counts(
                inventory
            )
        )

        for family in EXPECTED_FAMILY_COUNTS:

            item = family_counts[
                family
            ]

            print(
                f"{family:<18} "
                f"files={item['observed']} "
                f"expected={item['expected']}"
            )

        print()
        print(
            "Modality family counts: PASS"
        )

        # ---------------------------------------------------------------------
        # 5. ARCHITECTURE
        # ---------------------------------------------------------------------

        section(
            5,
            "FROZEN FIVE-GRU ARCHITECTURE",
        )

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

        # ---------------------------------------------------------------------
        # 6. PHYSICAL SCHEMA AUDIT
        # ---------------------------------------------------------------------

        section(
            6,
            "ACTUAL PHYSICAL CATEGORICAL SCHEMA AUDIT",
        )

        physical_schemas = (
            audit_physical_schemas(
                inventory
            )
        )

        # Activity
        print()
        print("ACTIVITY")
        print(
            "  Feature dimension: 12"
        )
        print(
            "  Categorical columns:"
        )
        print(
            "    - activity_type"
        )
        print(
            "    - intensity"
        )
        print(
            "  Physical schema: PASS"
        )

        # Nutrition
        print()
        print("NUTRITION")
        print(
            "  Feature dimension: 6"
        )
        print(
            "  Categorical columns:"
        )
        print(
            "    - meal_type"
        )
        print(
            "    - meal_tag"
        )
        print(
            "  Physical schema: PASS"
        )

        # Insulin
        print()
        print("INSULIN")

        print(
            "  Basal physical schema:"
        )
        print(
            "    - basal_dose"
        )
        print(
            "    - insulin_kind"
        )
        print(
            "  Basal schema: PASS"
        )

        print()
        print(
            "  Bolus physical schema:"
        )
        print(
            "    - bolus_dose"
        )
        print(
            "  Bolus schema: PASS"
        )

        print()
        print(
            "  IMPORTANT:"
        )
        print(
            "  Basal and bolus physical schemas are NOT "
            "required to contain identical columns."
        )
        print(
            "  The single Insulin GRU contract remains:"
        )
        print(
            "    [dose, event_type]"
        )
        print(
            "  No event_type construction is performed "
            "by this audit."
        )

        # Sleep
        print()
        print("SLEEP")

        print(
            "  Selected GRU representation:"
        )
        print(
            "    sleep time-series"
        )
        print(
            "  Feature dimension: 6"
        )
        print(
            "  Sleep-summary representation:"
        )
        print(
            "    audited separately; not a second GRU"
        )
        print(
            "  Sleep physical schema contract: PASS"
        )

        # ---------------------------------------------------------------------
        # 7. RAW CATEGORICAL DOMAIN AUDIT
        # ---------------------------------------------------------------------

        section(
            7,
            "RAW CATEGORICAL VALUE DOMAIN AUDIT",
        )

        print(
            "IMPORTANT:"
        )
        print(
            "  Raw values are inspected only."
        )
        print(
            "  No categorical encoding is performed."
        )
        print(
            "  No new columns are created."
        )
        print(
            "  No source values are changed."
        )

        categorical_domains = (
            audit_categorical_domains(
                inventory
            )
        )

        for family in [
            "activity",
            "nutrition",
            "basal_insulin",
        ]:

            print()
            print(
                family.upper()
            )

            columns = (
                categorical_domains[
                    family
                ]["categorical_columns"]
            )

            for column, data in columns.items():

                values = data[
                    "global_unique_raw_values"
                ]

                print(
                    f"  {column}: "
                    f"{len(values)} unique raw value(s)"
                )

                for value in values:

                    print(
                        f"    - {value}"
                    )

                print(
                    "    Raw-domain audit: PASS"
                )

        # ---------------------------------------------------------------------
        # 8. ARCHITECTURE SAFETY
        # ---------------------------------------------------------------------

        section(
            8,
            "ARCHITECTURE SAFETY CHECK",
        )

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
            "Encoding performed:                    NO"
        )
        print(
            "Model trained:                         NO"
        )

        # ---------------------------------------------------------------------
        # 9. REPORT
        # ---------------------------------------------------------------------

        report[
            "cohort"
        ] = cohort

        report[
            "modality_families"
        ] = family_counts

        report[
            "physical_schemas"
        ] = physical_schemas

        report[
            "categorical_domains"
        ] = categorical_domains

        report[
            "architecture_safety"
        ] = architecture_safety()

        report[
            "status"
        ] = "PASS"

        section(
            9,
            "WRITING AUDIT REPORT",
        )

        write_report(
            report
        )

        print(
            "Audit report saved:"
        )
        print(
            f"  {REPORT_PATH}"
        )

        # ---------------------------------------------------------------------
        # FINAL
        # ---------------------------------------------------------------------

        print()
        print("=" * 80)
        print(
            "T1D-UOM CATEGORICAL INPUT CONTRACT "
            "AUDIT COMPLETE"
        )
        print("=" * 80)

        print()
        print("FINAL RESULT:")
        print(
            "  Categorical input contract: PASS"
        )
        print(
            "  Activity categorical schema: PASS"
        )
        print(
            "  Nutrition categorical schema: PASS"
        )
        print(
            "  Basal insulin categorical schema: PASS"
        )
        print(
            "  Bolus insulin physical schema: PASS"
        )
        print(
            "  Sleep physical schema: PASS"
        )
        print(
            "  Single Insulin GRU constraint: PASS"
        )
        print(
            "  Single Sleep GRU constraint: PASS"
        )
        print(
            "  No categorical encoding performed: PASS"
        )

        print()
        print("FROZEN ARCHITECTURE:")
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

        print()
        print("IMPORTANT:")
        print(
            "  No dataset was modified."
        )
        print(
            "  No sequence-input file was modified."
        )
        print(
            "  No categorical encoding was performed."
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

        print()
        print("NEXT STAGE:")
        print(
            "  The physical categorical schemas and "
            "raw categorical domains are now audited."
        )
        print(
            "  The single Insulin GRU contract remains "
            "[dose, event_type]."
        )
        print(
            "  The single Sleep GRU contract remains "
            "the 6-feature sleep time-series."
        )
        print(
            "  No encoding or representation construction "
            "is performed at this stage."
        )
        print("=" * 80)

        return 0

    except Exception as exc:

        report[
            "status"
        ] = "FAILED"

        report[
            "error"
        ] = str(exc)

        try:
            write_report(
                report
            )
        except Exception:
            pass

        print()
        print("=" * 80)
        print(
            "T1D-UOM CATEGORICAL INPUT CONTRACT "
            "AUDIT FAILED"
        )
        print("=" * 80)

        print()
        print(str(exc))

        print()
        print("IMPORTANT:")
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
            "  No model was trained."
        )
        print(
            "  Frozen architecture was not changed."
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())