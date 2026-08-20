"""
T1D-UOM Timestamp Correction Validation

PURPOSE
-------
Independently validate that:

1. The manifest exists and is structurally correct.
2. The raw dataset still contains the original suspicious values.
3. The derived dataset contains the approved corrected values.
4. Exactly the two approved files differ from raw.
5. The other 110 CSV files are byte-identical.
6. No unexpected files were added or removed.
7. The raw dataset was not modified.
8. The correction values are unique and exact.

READ-ONLY
---------
This script does NOT modify the raw or derived datasets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

RAW_DATASET = PROJECT_ROOT / "data" / "raw" / "t1d_uom_v1.0.3"

DERIVED_DATASET = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "t1d_uom_v1.0.3_timestamp_corrected"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
    / "timestamp_corrections.json"
)


# ============================================================================
# APPROVED CORRECTIONS
# ============================================================================

APPROVED_CORRECTIONS = [
    {
        "relative_path": "Nutrition Data/UoMNutrition2320.csv",
        "participant": "UoM2320",
        "column": "meal_ts",
        "original_value": "02/12/2033 20:00",
        "corrected_value": "02/12/2023 20:00",
    },
    {
        "relative_path": "Nutrition Data/UoMNutrition2404.csv",
        "participant": "UoM2404",
        "column": "meal_ts",
        "original_value": "22/04/2204 11:45",
        "corrected_value": "22/04/2024 11:45",
    },
]


# ============================================================================
# HELPERS
# ============================================================================

def fail(message: str) -> None:
    print()
    print("=" * 80)
    print("VALIDATION FAILED")
    print("=" * 80)
    print(message)
    print("=" * 80)
    sys.exit(1)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def discover_csv_files(root: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in root.rglob("*.csv")
            if path.is_file()
        ],
        key=lambda p: p.relative_to(root).as_posix().lower(),
    )


def relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def file_map(root: Path) -> Dict[str, Path]:
    return {
        relative_path(path, root): path
        for path in discover_csv_files(root)
    }


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
            raise ValueError(f"Empty CSV file: {path}")

        rows = list(reader)

    return header, rows


def find_exact_values(
    path: Path,
    column: str,
    value: str,
) -> Tuple[List[str], List[List[str]], int, List[int]]:
    """
    Find exact value occurrences in a specified column.

    Returns:
        header,
        rows,
        column_index,
        matching data-row indices
    """
    header, rows = read_csv(path)

    if column not in header:
        raise ValueError(
            f"Column '{column}' not found in {path}."
        )

    column_index = header.index(column)

    matches = []

    for index, row in enumerate(rows):
        if len(row) != len(header):
            raise ValueError(
                f"Malformed row in {path}: "
                f"data-row index {index}; "
                f"expected {len(header)} fields, "
                f"found {len(row)}."
            )

        if row[column_index] == value:
            matches.append(index)

    return header, rows, column_index, matches


# ============================================================================
# MANIFEST VALIDATION
# ============================================================================

def validate_manifest() -> Dict:
    print("-" * 80)
    print("1. MANIFEST VALIDATION")
    print("-" * 80)

    if not MANIFEST_PATH.exists():
        fail(f"Manifest does not exist:\n{MANIFEST_PATH}")

    try:
        with MANIFEST_PATH.open(
            "r",
            encoding="utf-8",
        ) as handle:
            manifest = json.load(handle)
    except Exception as exc:
        fail(f"Manifest could not be read:\n{exc}")

    print(f"Manifest: {MANIFEST_PATH}")
    print("Manifest JSON: readable")

    required_top_level = [
        "manifest_version",
        "created_at_utc",
        "raw_dataset",
        "derived_dataset",
        "raw_dataset_modified",
        "approved_corrections",
        "correction_results",
        "final_verification",
        "status",
    ]

    for key in required_top_level:
        if key not in manifest:
            fail(f"Manifest missing required key: {key}")

    if manifest["status"] != "SUCCESS":
        fail(
            "Manifest does not report SUCCESS.\n"
            f"Status: {manifest['status']}"
        )

    if manifest["raw_dataset_modified"] is not False:
        fail(
            "Manifest does not state that the raw dataset was preserved."
        )

    if len(manifest["approved_corrections"]) != 2:
        fail(
            "Manifest does not contain exactly two approved corrections."
        )

    print("Manifest checks: PASS")

    return manifest


# ============================================================================
# DATASET STRUCTURE
# ============================================================================

def validate_structure() -> Tuple[Dict[str, Path], Dict[str, Path]]:
    print()
    print("-" * 80)
    print("2. DATASET STRUCTURE VALIDATION")
    print("-" * 80)

    if not RAW_DATASET.is_dir():
        fail(f"Raw dataset does not exist:\n{RAW_DATASET}")

    if not DERIVED_DATASET.is_dir():
        fail(f"Derived dataset does not exist:\n{DERIVED_DATASET}")

    raw_files = file_map(RAW_DATASET)
    derived_files = file_map(DERIVED_DATASET)

    print(f"Raw CSV files:     {len(raw_files)}")
    print(f"Derived CSV files: {len(derived_files)}")

    if len(raw_files) != 112:
        fail(
            f"Expected 112 raw CSV files, found {len(raw_files)}."
        )

    if len(derived_files) != 112:
        fail(
            f"Expected 112 derived CSV files, found {len(derived_files)}."
        )

    if set(raw_files) != set(derived_files):
        missing = sorted(set(raw_files) - set(derived_files))
        extra = sorted(set(derived_files) - set(raw_files))

        fail(
            "CSV file-list preservation failed.\n"
            f"Missing from derived: {missing}\n"
            f"Extra in derived: {extra}"
        )

    print("CSV file-list preservation: PASS")

    return raw_files, derived_files


# ============================================================================
# RAW PRESERVATION
# ============================================================================

def validate_raw_preservation(
    raw_files: Dict[str, Path],
    manifest: Dict,
) -> Dict[str, str]:
    print()
    print("-" * 80)
    print("3. RAW DATASET PRESERVATION")
    print("-" * 80)

    # Reconstruct expected original hashes from the manifest where possible.
    correction_results = manifest.get("correction_results", [])

    original_hashes = {}

    for result in correction_results:
        path = result["relative_path"]

        if "raw_sha256" not in result:
            fail(
                f"Manifest correction result lacks raw_sha256: {path}"
            )

        original_hashes[path] = result["raw_sha256"]

    # Verify the two known target files against the hashes captured by the
    # correction script.
    for correction in APPROVED_CORRECTIONS:
        path = correction["relative_path"]

        if path not in raw_files:
            fail(f"Raw target file missing: {path}")

        current_hash = sha256_file(raw_files[path])

        if path not in original_hashes:
            fail(
                f"No original raw hash exists in manifest for {path}."
            )

        if current_hash != original_hashes[path]:
            fail(
                "RAW DATASET APPEARS TO HAVE CHANGED.\n"
                f"File: {path}\n"
                f"Manifest hash: {original_hashes[path]}\n"
                f"Current hash:  {current_hash}"
            )

        print(
            f"PASS: raw {path} still has original SHA-256."
        )

    # Verify exact suspicious values remain in raw.
    for correction in APPROVED_CORRECTIONS:
        path = correction["relative_path"]

        (
            _header,
            _rows,
            _column_index,
            matches,
        ) = find_exact_values(
            raw_files[path],
            correction["column"],
            correction["original_value"],
        )

        if len(matches) != 1:
            fail(
                "Raw dataset target-value validation failed.\n"
                f"File: {path}\n"
                f"Column: {correction['column']}\n"
                f"Expected exactly one original value.\n"
                f"Found: {len(matches)}"
            )

        print(
            f"PASS: {path} contains exactly one original "
            f"suspicious timestamp."
        )

    print("Raw preservation: PASS")

    return original_hashes


# ============================================================================
# DERIVED CORRECTION VALIDATION
# ============================================================================

def validate_derived_corrections(
    raw_files: Dict[str, Path],
    derived_files: Dict[str, Path],
) -> None:
    print()
    print("-" * 80)
    print("4. DERIVED TIMESTAMP CORRECTIONS")
    print("-" * 80)

    for correction in APPROVED_CORRECTIONS:
        path = correction["relative_path"]

        raw_path = raw_files[path]
        derived_path = derived_files[path]

        # --------------------------------------------------------------
        # RAW: exactly one original value.
        # --------------------------------------------------------------
        (
            raw_header,
            raw_rows,
            raw_column_index,
            raw_matches,
        ) = find_exact_values(
            raw_path,
            correction["column"],
            correction["original_value"],
        )

        if len(raw_matches) != 1:
            fail(
                f"Raw target is not unique: {path}"
            )

        # --------------------------------------------------------------
        # DERIVED: exactly one corrected value.
        # --------------------------------------------------------------
        (
            derived_header,
            derived_rows,
            derived_column_index,
            corrected_matches,
        ) = find_exact_values(
            derived_path,
            correction["column"],
            correction["corrected_value"],
        )

        if len(corrected_matches) != 1:
            fail(
                "Derived timestamp correction is incorrect.\n"
                f"File: {path}\n"
                f"Expected exactly one corrected value: "
                f"{correction['corrected_value']}\n"
                f"Found: {len(corrected_matches)}"
            )

        # --------------------------------------------------------------
        # DERIVED: original suspicious value must be gone.
        # --------------------------------------------------------------
        (
            _,
            _,
            _,
            remaining_original,
        ) = find_exact_values(
            derived_path,
            correction["column"],
            correction["original_value"],
        )

        if remaining_original:
            fail(
                "Derived dataset still contains the original suspicious "
                "timestamp.\n"
                f"File: {path}\n"
                f"Data-row indices: {remaining_original}"
            )

        # --------------------------------------------------------------
        # Header must remain unchanged.
        # --------------------------------------------------------------
        if raw_header != derived_header:
            fail(
                f"Header changed unexpectedly in {path}."
            )

        print(
            f"PASS: {path}\n"
            f"      {correction['original_value']}\n"
            f"      -> {correction['corrected_value']}"
        )

    print()
    print("Derived timestamp corrections: PASS")


# ============================================================================
# BYTE-LEVEL CHANGESET VALIDATION
# ============================================================================

def validate_changeset(
    raw_files: Dict[str, Path],
    derived_files: Dict[str, Path],
) -> None:
    print()
    print("-" * 80)
    print("5. RAW / DERIVED BYTE-LEVEL CHANGESET")
    print("-" * 80)

    changed = []
    identical = []

    for path in sorted(raw_files):
        raw_hash = sha256_file(raw_files[path])
        derived_hash = sha256_file(derived_files[path])

        if raw_hash == derived_hash:
            identical.append(path)
        else:
            changed.append(path)

    approved = {
        correction["relative_path"]
        for correction in APPROVED_CORRECTIONS
    }

    if set(changed) != approved:
        unexpected = sorted(set(changed) - approved)
        missing = sorted(approved - set(changed))

        fail(
            "Derived dataset changes do not exactly match approved changes.\n"
            f"Unexpected changed files: {unexpected}\n"
            f"Approved files not changed: {missing}"
        )

    if len(changed) != 2:
        fail(
            f"Expected exactly 2 changed files; found {len(changed)}."
        )

    if len(identical) != 110:
        fail(
            f"Expected exactly 110 byte-identical files; "
            f"found {len(identical)}."
        )

    print(f"Byte-identical CSV files: {len(identical)}")
    print(f"Changed CSV files:        {len(changed)}")

    print()
    print("Changed files:")
    for path in changed:
        print(f"  - {path}")

    print()
    print("Exact changeset: PASS")


# ============================================================================
# MANIFEST CONSISTENCY
# ============================================================================

def validate_manifest_against_current_state(
    manifest: Dict,
) -> None:
    print()
    print("-" * 80)
    print("6. MANIFEST / CURRENT DATASET CONSISTENCY")
    print("-" * 80)

    manifest_changed = sorted(
        result["relative_path"]
        for result in manifest["correction_results"]
    )

    expected_changed = sorted(
        correction["relative_path"]
        for correction in APPROVED_CORRECTIONS
    )

    if manifest_changed != expected_changed:
        fail(
            "Manifest correction targets do not match the approved "
            "correction set.\n"
            f"Manifest: {manifest_changed}\n"
            f"Expected: {expected_changed}"
        )

    for result, expected in zip(
        sorted(
            manifest["correction_results"],
            key=lambda x: x["relative_path"],
        ),
        sorted(
            APPROVED_CORRECTIONS,
            key=lambda x: x["relative_path"],
        ),
    ):
        for key in (
            "relative_path",
            "participant",
            "column",
            "original_value",
            "corrected_value",
        ):
            if result.get(key) != expected.get(key):
                fail(
                    "Manifest correction metadata mismatch.\n"
                    f"Field: {key}\n"
                    f"Manifest: {result.get(key)}\n"
                    f"Expected: {expected.get(key)}"
                )

        if result.get("status") != "APPLIED_AND_VERIFIED":
            fail(
                f"Manifest result for {expected['relative_path']} "
                "is not APPLIED_AND_VERIFIED."
            )

    print("Manifest consistency: PASS")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print()
    print("=" * 80)
    print("T1D-UOM TIMESTAMP CORRECTION VALIDATION")
    print("=" * 80)
    print()
    print(f"Project root:    {PROJECT_ROOT}")
    print(f"Raw dataset:     {RAW_DATASET}")
    print(f"Derived dataset: {DERIVED_DATASET}")
    print(f"Manifest:        {MANIFEST_PATH}")
    print()
    print("IMPORTANT: READ-ONLY VALIDATION.")
    print("NO RAW OR DERIVED DATASET FILES WILL BE MODIFIED.")

    try:
        # 1. Manifest
        manifest = validate_manifest()

        # 2. Structure
        raw_files, derived_files = validate_structure()

        # 3. Raw preservation
        validate_raw_preservation(
            raw_files,
            manifest,
        )

        # 4. Corrected values
        validate_derived_corrections(
            raw_files,
            derived_files,
        )

        # 5. Exact changeset
        validate_changeset(
            raw_files,
            derived_files,
        )

        # 6. Manifest consistency
        validate_manifest_against_current_state(
            manifest,
        )

        print()
        print("=" * 80)
        print("TIMESTAMP CORRECTION VALIDATION PASSED")
        print("=" * 80)
        print()
        print("RAW DATASET:       PRESERVED")
        print("DERIVED DATASET:   VERIFIED")
        print("APPROVED CHANGES:  2")
        print("OTHER FILES:       110 BYTE-IDENTICAL")
        print()
        print("The timestamp-correction stage is validated.")
        print()
        print("NEXT STEP:")
        print("Run:")
        print()
        print("    python src\\data\\audit_dataset.py")
        print()

    except Exception as exc:
        fail(str(exc))


if __name__ == "__main__":
    main()