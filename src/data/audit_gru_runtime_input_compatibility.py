from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


# =============================================================================
# T1D-UOM GRU RUNTIME INPUT COMPATIBILITY AUDIT
# =============================================================================
#
# READ-ONLY.
#
# This script does NOT:
#   - modify any dataset
#   - modify sequence inputs
#   - encode categorical values
#   - normalize values
#   - impute values
#   - resample
#   - interpolate
#   - delete rows
#   - create windows
#   - create targets
#   - train a model
#   - implement MLP Fusion
#   - implement Digital Twin
#   - implement Prediction
#   - implement What-if
#   - implement Interactive UI
#
# PURPOSE
# -------
# Verify whether the already-frozen sequence-input schemas are directly
# compatible with tensor-based GRU execution.
#
# This is a CONTRACT AUDIT ONLY.
#
# Frozen architecture:
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
    / "gru_runtime_input_compatibility_audit.json"
)

MODEL_MANIFEST = (
    REPORT_DIR
    / "modeling_dataset_manifest.json"
)

SEQUENCE_PREPARATION_MANIFEST = (
    REPORT_DIR
    / "sequence_input_preparation_manifest.json"
)

FIVE_GRU_CONTRACT_REPORT = (
    REPORT_DIR
    / "five_gru_input_contract_audit.json"
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
# EXPECTED FILE COUNTS
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
# EXPECTED TIMESTAMP COLUMNS
# =============================================================================

EXPECTED_TIMESTAMP_COLUMNS = {
    "activity": "activity_ts",
    "glucose": "bg_ts",
    "basal_insulin": "basal_ts",
    "bolus_insulin": "bolus_ts",
    "nutrition": "meal_ts",
    "sleep_summary": "start_date_ts",
    "sleep_timeseries": "sleep_ts",
}


# =============================================================================
# FROZEN ARCHITECTURE
# =============================================================================

ARCHITECTURE = {
    "glucose": "Glucose -> GRU -> zG",
    "insulin": "Insulin -> GRU -> zI",
    "nutrition": "Nutrition -> GRU -> zN",
    "activity": "Activity -> GRU -> zA",
    "sleep": "Sleep -> GRU -> zS",
    "fusion": "zG,zI,zN,zA,zS -> MLP Fusion",
    "state": "MLP Fusion -> Unified Patient State",
    "digital_twin": "Unified Patient State -> DIGITAL TWIN",
    "downstream": "DIGITAL TWIN -> Prediction / What-if -> Interactive UI",
}


# =============================================================================
# ROBUST PARTICIPANT PARSER
# =============================================================================

PARTICIPANT_PATTERN = re.compile(
    r"UoM(?:Activity|Glucose|Basal|Bolus|Nutrition|sleep)?(\d+)",
    re.IGNORECASE,
)


def identify_participant(path: Path) -> Optional[str]:
    match = PARTICIPANT_PATTERN.search(path.name)

    if match is None:
        return None

    return f"UoM{match.group(1)}"


# =============================================================================
# FAMILY IDENTIFICATION
# =============================================================================

def identify_family(relative_path: Path) -> Optional[str]:
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
# CSV READING
# =============================================================================

def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:

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

        rows = list(reader)

    header = [
        value.strip()
        for value in header
    ]

    return header, rows


# =============================================================================
# NUMERIC TEST
# =============================================================================

def is_numeric(value: str) -> bool:

    value = value.strip()

    if value == "":
        return False

    try:
        float(value)
        return True
    except ValueError:
        return False


# =============================================================================
# VALUE CLASSIFICATION
# =============================================================================

def classify_column(
    values: list[str],
) -> dict:

    non_empty = [
        value.strip()
        for value in values
        if value.strip() != ""
    ]

    if not non_empty:
        return {
            "classification": "empty",
            "non_empty_values": 0,
            "numeric_values": 0,
            "non_numeric_values": 0,
            "unique_non_empty_values": 0,
            "sample_non_numeric_values": [],
        }

    numeric_count = sum(
        is_numeric(value)
        for value in non_empty
    )

    non_numeric = [
        value
        for value in non_empty
        if not is_numeric(value)
    ]

    if numeric_count == len(non_empty):
        classification = "numeric"

    elif numeric_count == 0:
        classification = "non_numeric"

    else:
        classification = "mixed"

    unique_values = list(
        dict.fromkeys(non_empty)
    )

    return {
        "classification": classification,
        "non_empty_values": len(non_empty),
        "numeric_values": numeric_count,
        "non_numeric_values": len(non_numeric),
        "unique_non_empty_values": len(unique_values),
        "sample_non_numeric_values": non_numeric[:10],
    }


# =============================================================================
# MANIFEST VALIDATION
# =============================================================================

def validate_manifests() -> None:

    required = [
        MODEL_MANIFEST,
        SEQUENCE_PREPARATION_MANIFEST,
        FIVE_GRU_CONTRACT_REPORT,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Required audit/manifest file(s) missing:\n"
            + "\n".join(
                f"  - {path}"
                for path in missing
            )
        )


# =============================================================================
# DISCOVER FILES
# =============================================================================

def discover_files() -> list[Path]:

    if not SEQUENCE_INPUTS.exists():
        raise RuntimeError(
            f"Sequence-input directory does not exist:\n"
            f"{SEQUENCE_INPUTS}"
        )

    return sorted(
        path
        for path in SEQUENCE_INPUTS.rglob("*.csv")
        if path.is_file()
    )


# =============================================================================
# MAIN AUDIT
# =============================================================================

def main() -> int:

    print("=" * 80)
    print("T1D-UOM GRU RUNTIME INPUT COMPATIBILITY AUDIT")
    print("=" * 80)

    print(
        """
IMPORTANT: READ-ONLY.
No dataset files will be modified.
No sequence-input files will be modified.
No values will be transformed.
No categorical encoding will be performed.
No normalization will be performed.
No imputation will be performed.
No resampling will be performed.
No interpolation will be performed.
No rows will be deleted.
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

    print(
        f"Project root:       {PROJECT_ROOT}"
    )
    print(
        f"Sequence inputs:    {SEQUENCE_INPUTS}"
    )
    print(
        f"Audit report:       {REPORT_PATH}"
    )

    try:

        # =====================================================================
        print("\n" + "-" * 80)
        print("1. DIRECTORY AND MANIFEST VALIDATION")
        print("-" * 80)

        validate_manifests()

        print(
            "Sequence-input directory: PASS"
        )
        print(
            "modeling_dataset_manifest.json: PASS"
        )
        print(
            "sequence_input_preparation_manifest.json: PASS"
        )
        print(
            "five_gru_input_contract_audit.json: PASS"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("2. SEQUENCE-INPUT INVENTORY")
        print("-" * 80)

        files = discover_files()

        print(
            f"Sequence-input CSV files discovered: {len(files)}"
        )

        if len(files) != 86:
            raise RuntimeError(
                "Expected exactly 86 frozen sequence-input CSV files.\n"
                f"Observed: {len(files)}"
            )

        print(
            "Frozen sequence-input file count: PASS"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("3. FROZEN COHORT AND FAMILY VALIDATION")
        print("-" * 80)

        records = []

        for path in files:

            relative = path.relative_to(
                SEQUENCE_INPUTS
            )

            participant = identify_participant(path)

            if participant is None:
                raise RuntimeError(
                    "Unable to identify participant:\n"
                    f"{path}"
                )

            family = identify_family(relative)

            if family is None:
                raise RuntimeError(
                    "Unable to identify modality family:\n"
                    f"{path}"
                )

            header, rows = read_csv(path)

            if len(header) == 0:
                raise RuntimeError(
                    f"Empty header:\n{path}"
                )

            if any(
                len(row) != len(header)
                for row in rows
            ):
                raise RuntimeError(
                    "CSV row width does not match header.\n"
                    f"File: {path}"
                )

            timestamp_columns = [
                column
                for column in header
                if column.lower().endswith("_ts")
            ]

            if len(timestamp_columns) != 1:
                raise RuntimeError(
                    "Expected exactly one *_ts timestamp column.\n"
                    f"File: {path}\n"
                    f"Detected: {timestamp_columns}"
                )

            records.append(
                {
                    "path": path,
                    "relative_path": relative.as_posix(),
                    "participant": participant,
                    "family": family,
                    "header": header,
                    "rows": rows,
                    "timestamp_column": timestamp_columns[0],
                }
            )

        participants = {
            record["participant"]
            for record in records
        }

        if participants != FROZEN_COHORT_SET:
            missing = (
                FROZEN_COHORT_SET
                - participants
            )

            unexpected = (
                participants
                - FROZEN_COHORT_SET
            )

            message = []

            if missing:
                message.append(
                    "Missing participants:\n"
                    + "\n".join(
                        f"  - {value}"
                        for value in sorted(missing)
                    )
                )

            if unexpected:
                message.append(
                    "Unexpected participants:\n"
                    + "\n".join(
                        f"  - {value}"
                        for value in sorted(unexpected)
                    )
                )

            raise RuntimeError(
                "\n".join(message)
            )

        family_counts = Counter(
            record["family"]
            for record in records
        )

        for family, expected in EXPECTED_FAMILY_COUNTS.items():

            actual = family_counts.get(
                family,
                0,
            )

            if actual != expected:
                raise RuntimeError(
                    f"{family}: expected {expected} files, "
                    f"observed {actual}"
                )

        print(
            "Frozen 13-participant cohort: PASS"
        )
        print(
            "Frozen modality-family file counts: PASS"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("4. ACTUAL COLUMN TYPE COMPATIBILITY")
        print("-" * 80)

        column_audit = []

        for record in records:

            path = record["path"]
            header = record["header"]
            rows = record["rows"]
            timestamp = record["timestamp_column"]

            print(
                f"\n[{record['family']}] "
                f"{record['relative_path']}"
            )

            if (
                timestamp
                != EXPECTED_TIMESTAMP_COLUMNS[
                    record["family"]
                ]
            ):
                raise RuntimeError(
                    "Unexpected timestamp column.\n"
                    f"File: {path}\n"
                    f"Expected: "
                    f"{EXPECTED_TIMESTAMP_COLUMNS[record['family']]}\n"
                    f"Observed: {timestamp}"
                )

            feature_columns = [
                column
                for column in header
                if column != timestamp
            ]

            for column in feature_columns:

                column_index = header.index(
                    column
                )

                values = [
                    row[column_index]
                    for row in rows
                ]

                classification = classify_column(
                    values
                )

                item = {
                    "family": record["family"],
                    "participant": record["participant"],
                    "relative_path": record["relative_path"],
                    "column": column,
                    **classification,
                }

                column_audit.append(item)

                print(
                    f"  {column:<35}"
                    f"{classification['classification']}"
                )

        # =====================================================================
        print("\n" + "-" * 80)
        print("5. MODALITY RUNTIME COMPATIBILITY SUMMARY")
        print("-" * 80)

        family_summary = {}

        for family in [
            "glucose",
            "nutrition",
            "activity",
            "basal_insulin",
            "bolus_insulin",
            "sleep_timeseries",
            "sleep_summary",
        ]:

            family_items = [
                item
                for item in column_audit
                if item["family"] == family
            ]

            classification_counts = Counter(
                item["classification"]
                for item in family_items
            )

            family_summary[family] = {
                "classification_counts": dict(
                    classification_counts
                ),
                "columns": sorted(
                    {
                        item["column"]
                        for item in family_items
                    }
                ),
            }

            print(
                f"\n{family}"
            )

            print(
                "  numeric:     "
                f"{classification_counts.get('numeric', 0)}"
            )

            print(
                "  non_numeric: "
                f"{classification_counts.get('non_numeric', 0)}"
            )

            print(
                "  mixed:       "
                f"{classification_counts.get('mixed', 0)}"
            )

            print(
                "  empty:       "
                f"{classification_counts.get('empty', 0)}"
            )

        # =====================================================================
        print("\n" + "-" * 80)
        print("6. FIVE-GRU RUNTIME INPUT SAFETY")
        print("-" * 80)

        non_numeric_features = [
            item
            for item in column_audit
            if item["classification"] != "numeric"
        ]

        if non_numeric_features:

            print(
                "REAL-DATA GRU RUNTIME COMPATIBILITY: "
                "REQUIRES EXPLICIT INPUT ADAPTER CONTRACT"
            )

            print(
                "\nNon-numeric or mixed feature columns detected:"
            )

            unique_problem_columns = {}

            for item in non_numeric_features:

                key = (
                    item["family"],
                    item["column"],
                )

                unique_problem_columns.setdefault(
                    key,
                    [],
                ).append(item)

            for (
                family,
                column,
            ), items in sorted(
                unique_problem_columns.items()
            ):

                classifications = sorted(
                    {
                        item["classification"]
                        for item in items
                    }
                )

                samples = []

                for item in items:
                    samples.extend(
                        item[
                            "sample_non_numeric_values"
                        ]
                    )

                samples = list(
                    dict.fromkeys(samples)
                )[:10]

                print(
                    f"  {family:<18}"
                    f"{column:<35}"
                    f"{', '.join(classifications)}"
                )

                if samples:
                    print(
                        f"    sample values: "
                        f"{samples}"
                    )

            print(
                "\nIMPORTANT:"
            )

            print(
                "  This is NOT a dataset failure."
            )

            print(
                "  This is NOT an architecture failure."
            )

            print(
                "  No encoding has been performed."
            )

            print(
                "  No values have been changed."
            )

            print(
                "  The five-GRU architecture remains unchanged."
            )

            print(
                "\nThe real-data GRU execution should NOT begin "
                "until these non-numeric inputs have an explicitly "
                "approved, architecture-consistent representation."
            )

            runtime_status = (
                "REQUIRES_INPUT_ADAPTER_CONTRACT"
            )

        else:

            print(
                "All feature columns are numerically consumable."
            )

            print(
                "Direct tensor compatibility: PASS"
            )

            runtime_status = "PASS"

        # =====================================================================
        print("\n" + "-" * 80)
        print("7. FROZEN ARCHITECTURE SAFETY CHECK")
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

        print(
            "\nAdditional Insulin GRU: NO"
        )

        print(
            "Additional Sleep GRU:   NO"
        )

        print(
            "MLP Fusion implemented:  NO"
        )

        print(
            "Digital Twin implemented: NO"
        )

        print(
            "Prediction implemented:   NO"
        )

        print(
            "What-if implemented:      NO"
        )

        print(
            "Interactive UI implemented: NO"
        )

        # =====================================================================
        print("\n" + "-" * 80)
        print("8. WRITING AUDIT REPORT")
        print("-" * 80)

        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = {
            "report_type": (
                "t1d_uom_gru_runtime_input_compatibility_audit"
            ),
            "status": runtime_status,
            "read_only": True,
            "project_root": str(
                PROJECT_ROOT
            ),
            "sequence_inputs": str(
                SEQUENCE_INPUTS
            ),
            "sequence_input_file_count": len(
                records
            ),
            "frozen_cohort": FROZEN_COHORT,
            "architecture": ARCHITECTURE,
            "family_counts": dict(
                family_counts
            ),
            "column_audit": column_audit,
            "family_summary": family_summary,
            "non_numeric_features": non_numeric_features,
            "architecture_safety": {
                "additional_insulin_gru": False,
                "additional_sleep_gru": False,
                "mlp_fusion_implemented": False,
                "digital_twin_implemented": False,
                "prediction_implemented": False,
                "what_if_implemented": False,
                "interactive_ui_implemented": False,
            },
            "data_safety": {
                "dataset_modified": False,
                "sequence_inputs_modified": False,
                "encoding_performed": False,
                "normalization_performed": False,
                "imputation_performed": False,
                "resampling_performed": False,
                "interpolation_performed": False,
                "rows_deleted": False,
            },
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

        if runtime_status == "PASS":

            print(
                "T1D-UOM GRU RUNTIME INPUT COMPATIBILITY AUDIT: PASS"
            )

            print("=" * 80)

            print(
                "\nAll verified sequence-input feature columns "
                "are numerically consumable."
            )

            print(
                "\nNEXT STAGE:"
            )

            print(
                "Controlled real-data execution of the five "
                "existing GRU branches."
            )

        else:

            print(
                "T1D-UOM GRU RUNTIME INPUT COMPATIBILITY AUDIT: "
                "REQUIRES INPUT ADAPTER CONTRACT"
            )

            print("=" * 80)

            print(
                "\nThe frozen dataset and architecture are NOT failed."
            )

            print(
                "The audit identified feature columns that cannot "
                "yet be passed directly to a numeric GRU tensor."
            )

            print(
                "\nNEXT STAGE:"
            )

            print(
                "Define only the minimal explicit input adapters "
                "required by the actual observed columns."
            )

            print(
                "Do NOT modify the frozen datasets."
            )

            print(
                "Do NOT add GRU branches."
            )

            print(
                "Do NOT implement MLP Fusion."
            )

            print(
                "Do NOT implement the Digital Twin."
            )

            print(
                "Do NOT implement Prediction."
            )

            print(
                "Do NOT implement What-if."
            )

            print(
                "Do NOT implement the Interactive UI."
            )

        print("=" * 80)

        return 0

    except Exception as exc:

        print("\n" + "=" * 80)
        print(
            "T1D-UOM GRU RUNTIME INPUT COMPATIBILITY AUDIT FAILED"
        )
        print("=" * 80)

        print(
            f"\n{exc}"
        )

        print(
            "\nIMPORTANT:"
        )

        print(
            "  No dataset files were modified."
        )

        print(
            "  No sequence-input files were modified."
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