from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "t1d_uom_v1.0.3_sequence_inputs"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"
REPORT_PATH = REPORT_DIR / "sleep_schema_audit.json"


SLEEP_SCHEMAS: Dict[str, Dict[str, object]] = {
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
}


def read_header(path: Path) -> List[str]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.reader(handle)

        try:
            return next(reader)
        except StopIteration:
            return []


def schema_columns(schema_name: str) -> List[str]:
    schema = SLEEP_SCHEMAS[schema_name]

    columns = [str(schema["timestamp"])]
    columns.extend(str(value) for value in schema["numeric"])
    columns.extend(str(value) for value in schema["categorical"])
    columns.extend(str(value) for value in schema["structured"])

    return columns


def classify_header(
    header: List[str],
) -> Tuple[str, List[str], List[str]]:
    header_set = set(header)

    exact_matches: List[str] = []

    for schema_name in SLEEP_SCHEMAS:
        expected = set(schema_columns(schema_name))

        if header_set == expected:
            exact_matches.append(schema_name)

    if len(exact_matches) == 1:
        return exact_matches[0], [], []

    scores = []

    for schema_name in SLEEP_SCHEMAS:
        expected = set(schema_columns(schema_name))

        missing = sorted(expected - header_set)
        unexpected = sorted(header_set - expected)

        score = len(missing) + len(unexpected)

        scores.append(
            (
                score,
                schema_name,
                missing,
                unexpected,
            )
        )

    scores.sort(key=lambda item: (item[0], item[1]))

    if not scores:
        return "unknown", [], sorted(header)

    _, closest_schema, missing, unexpected = scores[0]

    return (
        f"mismatch:{closest_schema}",
        missing,
        unexpected,
    )


def discover_sleep_files() -> List[Path]:
    if not SEQUENCE_INPUT_DIR.exists():
        raise FileNotFoundError(
            "Sequence-input directory does not exist:\n"
            f"{SEQUENCE_INPUT_DIR}"
        )

    return sorted(
        path
        for path in SEQUENCE_INPUT_DIR.rglob("*.csv")
        if path.is_file()
        and path.parent.name == "Sleep Data"
    )


def participant_from_name(path: Path) -> str:
    name = path.stem

    if name.startswith("UoMsleep"):
        participant_suffix = name[len("UoMsleep") :]
        return "UoM" + participant_suffix

    if name.startswith("UoM") and name.endswith("sleeptime"):
        return name[: -len("sleeptime")]

    return "UNKNOWN"


def classify_filename_role(path: Path) -> str:
    name = path.stem.lower()

    if name.endswith("sleeptime"):
        return "sleeptime_filename"

    if "sleep" in name:
        return "sleep_filename"

    return "unknown"


def main() -> int:
    print("=" * 80)
    print("T1D-UOM SLEEP PHYSICAL SCHEMA AUDIT")
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

    print(f"Project root:       {PROJECT_ROOT}")
    print(f"Sequence inputs:    {SEQUENCE_INPUT_DIR}")
    print(f"Audit report:       {REPORT_PATH}")
    print()

    if not SEQUENCE_INPUT_DIR.exists():
        print("ERROR: Sequence-input directory does not exist.")
        return 1

    files = discover_sleep_files()

    print("-" * 80)
    print("1. SLEEP FILE DISCOVERY")
    print("-" * 80)
    print(f"Sleep CSV files discovered: {len(files)}")
    print()

    records = []

    exact_schema_counts = {
        "sleep_timeseries": 0,
        "sleep_summary": 0,
        "unknown": 0,
    }

    filename_role_counts = {
        "sleeptime_filename": 0,
        "sleep_filename": 0,
        "unknown": 0,
    }

    for index, path in enumerate(files, start=1):
        header = read_header(path)

        physical_schema, missing, unexpected = classify_header(
            header
        )

        file_role = classify_filename_role(path)

        if physical_schema in exact_schema_counts:
            exact_schema_counts[physical_schema] += 1
        else:
            exact_schema_counts["unknown"] += 1

        filename_role_counts[file_role] += 1

        record = {
            "relative_path": str(
                path.relative_to(SEQUENCE_INPUT_DIR)
            ),
            "participant": participant_from_name(path),
            "filename_role": file_role,
            "column_count": len(header),
            "columns": header,
            "physical_schema": (
                physical_schema
                if not physical_schema.startswith("mismatch:")
                else "mismatch"
            ),
            "matched_schema": (
                None
                if not physical_schema.startswith("mismatch:")
                else physical_schema.split(":", 1)[1]
            ),
            "missing_columns": missing,
            "unexpected_columns": unexpected,
        }

        records.append(record)

        status = (
            "PASS"
            if physical_schema in SLEEP_SCHEMAS
            else "FAIL"
        )

        print(
            f"[{index:02d}/{len(files):02d}] "
            f"{record['relative_path']} | "
            f"filename_role={file_role} | "
            f"physical_schema={record['physical_schema']} | "
            f"{status}"
        )

    print()
    print("-" * 80)
    print("2. PHYSICAL SLEEP SCHEMA SUMMARY")
    print("-" * 80)

    print(
        "Exact sleep_timeseries schema: "
        f"{exact_schema_counts['sleep_timeseries']}"
    )

    print(
        "Exact sleep_summary schema:     "
        f"{exact_schema_counts['sleep_summary']}"
    )

    print(
        "Unknown / mismatched schema:    "
        f"{exact_schema_counts['unknown']}"
    )

    print()
    print("-" * 80)
    print("3. FILENAME VS PHYSICAL SCHEMA CROSS-CHECK")
    print("-" * 80)

    cross_check = []

    for record in records:
        file_role = record["filename_role"]
        physical_schema = record["physical_schema"]

        expected_from_filename = None

        if file_role == "sleeptime_filename":
            expected_from_filename = "sleep_timeseries"

        elif file_role == "sleep_filename":
            expected_from_filename = "sleep_summary"

        filename_schema_match = (
            expected_from_filename is not None
            and physical_schema == expected_from_filename
        )

        cross_check_record = {
            "relative_path": record["relative_path"],
            "filename_role": file_role,
            "expected_schema_from_filename": expected_from_filename,
            "actual_physical_schema": physical_schema,
            "filename_schema_match": filename_schema_match,
        }

        cross_check.append(cross_check_record)

        print(
            f"{record['relative_path']} | "
            f"filename={expected_from_filename} | "
            f"physical={physical_schema} | "
            f"{'MATCH' if filename_schema_match else 'MISMATCH'}"
        )

    filename_mismatches = [
        item
        for item in cross_check
        if not item["filename_schema_match"]
    ]

    print()
    print("-" * 80)
    print("4. FINAL RESULT")
    print("-" * 80)

    physical_schema_failures = [
        record
        for record in records
        if record["physical_schema"]
        not in {
            "sleep_timeseries",
            "sleep_summary",
        }
    ]

    physical_schema_pass = (
        len(records) == 22
        and len(physical_schema_failures) == 0
    )

    print(
        f"Files discovered:                 {len(records)}"
    )

    print(
        "Physical schema failures:         "
        f"{len(physical_schema_failures)}"
    )

    print(
        "Filename/schema mismatches:       "
        f"{len(filename_mismatches)}"
    )

    print(
        "Physical sleep schemas resolved:  "
        f"{'PASS' if physical_schema_pass else 'FAIL'}"
    )

    report = {
        "audit_name": (
            "T1D-UOM Sleep Physical Schema Audit"
        ),
        "audit_version": "1.0.1",
        "read_only": True,
        "project_root": str(PROJECT_ROOT),
        "sequence_input_directory": str(
            SEQUENCE_INPUT_DIR
        ),
        "report_path": str(REPORT_PATH),
        "files_discovered": len(records),
        "physical_schema_counts": exact_schema_counts,
        "filename_role_counts": filename_role_counts,
        "physical_schema_contract": SLEEP_SCHEMAS,
        "filename_schema_mismatches": filename_mismatches,
        "physical_schema_failures": [
            {
                "relative_path": record["relative_path"],
                "columns": record["columns"],
                "missing_columns": record[
                    "missing_columns"
                ],
                "unexpected_columns": record[
                    "unexpected_columns"
                ],
            }
            for record in physical_schema_failures
        ],
        "files": records,
        "summary": {
            "physical_schema_pass": physical_schema_pass,
            "physical_schema_failures": len(
                physical_schema_failures
            ),
            "filename_schema_mismatches": len(
                filename_mismatches
            ),
        },
        "scope_controls": {
            "dataset_modified": False,
            "sequence_inputs_modified": False,
            "categorical_encoding_performed": False,
            "event_type_constructed": False,
            "windows_created": False,
            "targets_created": False,
            "imputation_performed": False,
            "normalization_performed": False,
            "feature_engineering_performed": False,
            "model_trained": False,
            "mlp_fusion_implemented": False,
            "digital_twin_implemented": False,
            "prediction_implemented": False,
            "what_if_implemented": False,
            "interactive_ui_implemented": False,
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
    print("5. WRITING AUDIT REPORT")
    print("-" * 80)
    print("Audit report saved:")
    print(f"  {REPORT_PATH}")
    print()

    if physical_schema_pass:
        print("=" * 80)
        print(
            "T1D-UOM SLEEP PHYSICAL SCHEMA AUDIT COMPLETE"
        )
        print("=" * 80)
        print()
        print("FINAL RESULT: PASS")
        print()
        print(
            "All discovered sleep files match one of the "
            "two frozen physical sleep schemas."
        )
        print(
            "Filename/schema mismatches are reported as "
            "contract information only."
        )
        print()
        print("No dataset files were modified.")
        print("No sequence-input files were modified.")
        print("No transformations were performed.")
        print()
        return 0

    print("=" * 80)
    print(
        "T1D-UOM SLEEP PHYSICAL SCHEMA AUDIT FAILED"
    )
    print("=" * 80)
    print()
    print(
        "Unexpected physical sleep schemas were detected."
    )
    print(
        "Review the JSON report before proceeding."
    )
    print()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())