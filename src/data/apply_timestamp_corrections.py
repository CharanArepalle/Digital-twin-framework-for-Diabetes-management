"""
T1D-UOM Evidence-Based Timestamp Correction

PURPOSE
-------
Create a clean derived dataset from the immutable raw dataset and apply
ONLY the explicitly approved timestamp corrections.

SAFETY PRINCIPLES
-----------------
1. The raw dataset is NEVER modified.
2. The derived dataset is rebuilt from the raw dataset.
3. Corrections are located by exact file + column + original value.
4. Row numbers are NOT used to locate corrections.
5. Each correction must have exactly one matching record.
6. Unexpected duplicates cause an immediate failure.
7. The script never performs broad/fuzzy timestamp correction.
8. All 112 raw CSV files must be preserved in the derived dataset.
9. The 110 unaffected CSV files must remain byte-identical.
10. The two corrected files must contain exactly the approved change.
11. A detailed provenance manifest is written only after successful completion.
12. Verification failures cause a non-zero exit status.

APPROVED CORRECTIONS
--------------------
Nutrition Data/UoMNutrition2320.csv
    column: meal_ts
    original: 02/12/2033 20:00
    corrected: 02/12/2023 20:00

Nutrition Data/UoMNutrition2404.csv
    column: meal_ts
    original: 22/04/2204 11:45
    corrected: 22/04/2024 11:45

IMPORTANT
---------
This script intentionally does NOT attempt to discover or automatically
correct any other suspicious timestamps.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================================
# PROJECT PATHS
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
#
# IMPORTANT:
# We deliberately identify records using exact values rather than row numbers.
# This avoids the row-number convention problem encountered previously.
#
# Paths use POSIX-style separators internally so the manifest is portable.
# ============================================================================

APPROVED_CORRECTIONS = [
    {
        "relative_path": "Nutrition Data/UoMNutrition2320.csv",
        "participant": "UoM2320",
        "column": "meal_ts",
        "original_value": "02/12/2033 20:00",
        "corrected_value": "02/12/2023 20:00",
        "reason": (
            "Approved evidence-based correction of an anomalous four-digit "
            "year in the meal timestamp."
        ),
    },
    {
        "relative_path": "Nutrition Data/UoMNutrition2404.csv",
        "participant": "UoM2404",
        "column": "meal_ts",
        "original_value": "22/04/2204 11:45",
        "corrected_value": "22/04/2024 11:45",
        "reason": (
            "Approved evidence-based correction of an anomalous four-digit "
            "year in the meal timestamp."
        ),
    },
]


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def fail(message: str) -> None:
    """Print a fatal error and terminate safely."""
    print()
    print("=" * 80)
    print("TIMESTAMP CORRECTION FAILED")
    print("=" * 80)
    print(message)
    print()
    print("IMPORTANT:")
    print("The raw dataset was NOT modified by this script.")
    print("=" * 80)
    sys.exit(1)


def normalized_relative_path(path: Path, root: Path) -> str:
    """Return a portable relative path using forward slashes."""
    return path.relative_to(root).as_posix()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 without loading an entire file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def discover_csv_files(root: Path) -> List[Path]:
    """Return all CSV files recursively, sorted deterministically."""
    return sorted(
        [
            path
            for path in root.rglob("*.csv")
            if path.is_file()
        ],
        key=lambda p: p.relative_to(root).as_posix().lower(),
    )


def file_map(root: Path) -> Dict[str, Path]:
    """Map normalized relative CSV paths to absolute paths."""
    return {
        normalized_relative_path(path, root): path
        for path in discover_csv_files(root)
    }


def read_csv_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    """
    Read a CSV while preserving string values.

    newline="" is important for CSV correctness.
    UTF-8-SIG handles files that contain a UTF-8 BOM.
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
            raise ValueError("CSV file is empty.")

        rows = list(reader)

    return header, rows


def write_csv_rows(
    path: Path,
    header: List[str],
    rows: List[List[str]],
) -> None:
    """
    Write CSV using a deterministic representation.

    IMPORTANT:
    This function is used only for the two files that are intentionally
    corrected. All other files are copied byte-for-byte.
    """
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(
            handle,
            lineterminator="\n",
        )

        writer.writerow(header)
        writer.writerows(rows)


def ensure_header_column(
    header: List[str],
    column: str,
    relative_path: str,
) -> int:
    """Return column index or fail if the requested column does not exist."""
    if column not in header:
        raise ValueError(
            f"Required column '{column}' was not found in "
            f"{relative_path}. Columns={header}"
        )

    return header.index(column)


def validate_participant(
    relative_path: str,
    participant: str,
) -> None:
    """
    Validate the explicitly declared participant without making assumptions
    about how participant IDs are embedded in filenames.

    The filename itself is already an explicit correction target, so this
    function performs only a conservative sanity check:

    - the participant must be non-empty;
    - the filename must be non-empty.

    It deliberately does NOT require the participant string to appear
    literally inside the filename because the T1D-UOM naming convention does
    not use the participant identifier verbatim in every filename.
    """
    if not participant:
        raise ValueError(
            f"Approved correction for '{relative_path}' has an empty "
            "participant identifier."
        )

    if not Path(relative_path).name:
        raise ValueError(
            f"Approved correction has an invalid filename: "
            f"{relative_path}"
        )


# ============================================================================
# RAW DATASET VALIDATION
# ============================================================================

def validate_raw_dataset() -> Dict[str, Path]:
    """Validate the raw dataset and capture its complete file map."""
    print("-" * 80)
    print("RAW DATASET VALIDATION")
    print("-" * 80)

    if not RAW_DATASET.exists():
        fail(f"Raw dataset does not exist:\n{RAW_DATASET}")

    if not RAW_DATASET.is_dir():
        fail(f"Raw dataset is not a directory:\n{RAW_DATASET}")

    raw_files = file_map(RAW_DATASET)

    print(f"Raw CSV files discovered: {len(raw_files)}")

    if len(raw_files) != 112:
        fail(
            "Unexpected raw CSV count.\n"
            f"Expected: 112\n"
            f"Found:    {len(raw_files)}"
        )

    if len(raw_files) != len(set(raw_files)):
        fail("Duplicate normalized CSV relative paths detected.")

    for correction in APPROVED_CORRECTIONS:
        relative_path = correction["relative_path"]

        if relative_path not in raw_files:
            fail(
                "Approved correction target does not exist in the raw dataset:\n"
                f"{relative_path}"
            )

        validate_participant(
            relative_path,
            correction["participant"],
        )

    print("Raw dataset structure: PASS")
    print()
    print("Capturing SHA-256 hashes for all raw CSV files...")

    hashes = {}

    for relative_path, path in raw_files.items():
        hashes[relative_path] = sha256_file(path)

    print(f"Captured SHA-256 hashes for {len(hashes)} raw CSV files.")

    return raw_files, hashes


# ============================================================================
# TARGET VALIDATION
# ============================================================================

def locate_unique_target(
    path: Path,
    correction: Dict[str, str],
) -> Tuple[List[str], List[List[str]], int, int]:
    """
    Locate exactly one approved target by exact column/value matching.

    Returns:
        header
        rows
        target_row_index_zero_based
        target_column_index
    """
    header, rows = read_csv_rows(path)

    column = correction["column"]
    original_value = correction["original_value"]
    relative_path = correction["relative_path"]

    column_index = ensure_header_column(
        header,
        column,
        relative_path,
    )

    matches = []

    for row_index, row in enumerate(rows):
        if len(row) != len(header):
            raise ValueError(
                f"Malformed CSV row detected in {relative_path}: "
                f"data row index {row_index}, "
                f"expected {len(header)} fields, found {len(row)}."
            )

        if row[column_index] == original_value:
            matches.append(row_index)

    if len(matches) == 0:
        raise ValueError(
            "Approved original timestamp was not found.\n"
            f"File: {relative_path}\n"
            f"Column: {column}\n"
            f"Expected original value: {original_value}"
        )

    if len(matches) > 1:
        raise ValueError(
            "Approved original timestamp occurs more than once.\n"
            f"File: {relative_path}\n"
            f"Column: {column}\n"
            f"Original value: {original_value}\n"
            f"Number of matches: {len(matches)}\n"
            f"Matching data-row indices: {matches}"
        )

    return header, rows, matches[0], column_index


def validate_and_apply_one_correction(
    correction: Dict[str, str],
    raw_files: Dict[str, Path],
    derived_files: Dict[str, Path],
) -> Dict[str, object]:
    """Apply and verify exactly one approved correction."""
    relative_path = correction["relative_path"]

    raw_path = raw_files[relative_path]
    derived_path = derived_files[relative_path]

    print()
    print("Target:", relative_path)
    print("Participant:", correction["participant"])
    print("Column:    ", correction["column"])
    print("Original:  ", correction["original_value"])
    print("Corrected: ", correction["corrected_value"])

    # ----------------------------------------------------------------------
    # Validate raw target.
    # ----------------------------------------------------------------------
    (
        raw_header,
        raw_rows,
        raw_target_index,
        raw_column_index,
    ) = locate_unique_target(
        raw_path,
        correction,
    )

    actual_raw_value = raw_rows[raw_target_index][raw_column_index]

    if actual_raw_value != correction["original_value"]:
        raise ValueError(
            "Raw target verification failed."
        )

    # ----------------------------------------------------------------------
    # Validate derived copy before editing.
    # It must initially contain exactly the same target value.
    # ----------------------------------------------------------------------
    (
        derived_header,
        derived_rows,
        derived_target_index,
        derived_column_index,
    ) = locate_unique_target(
        derived_path,
        correction,
    )

    if derived_header != raw_header:
        raise ValueError(
            f"Derived header differs from raw header for {relative_path}."
        )

    actual_derived_value_before = (
        derived_rows[derived_target_index][derived_column_index]
    )

    if actual_derived_value_before != correction["original_value"]:
        raise ValueError(
            "Derived dataset does not contain the expected original value "
            "before correction.\n"
            f"File: {relative_path}\n"
            f"Expected: {correction['original_value']}\n"
            f"Actual:   {actual_derived_value_before}"
        )

    # ----------------------------------------------------------------------
    # Apply exactly one replacement.
    # ----------------------------------------------------------------------
    derived_rows[derived_target_index][derived_column_index] = (
        correction["corrected_value"]
    )

    write_csv_rows(
        derived_path,
        derived_header,
        derived_rows,
    )

    # ----------------------------------------------------------------------
    # Re-read and verify.
    # ----------------------------------------------------------------------
    (
        verify_header,
        verify_rows,
    ) = read_csv_rows(derived_path)

    verify_column_index = ensure_header_column(
        verify_header,
        correction["column"],
        relative_path,
    )

    matching_corrected = []

    for row_index, row in enumerate(verify_rows):
        if len(row) != len(verify_header):
            raise ValueError(
                f"Malformed derived CSV row after correction in "
                f"{relative_path}."
            )

        if row[verify_column_index] == correction["corrected_value"]:
            matching_corrected.append(row_index)

    if len(matching_corrected) != 1:
        raise ValueError(
            "Derived correction verification failed.\n"
            f"File: {relative_path}\n"
            f"Expected exactly one corrected value.\n"
            f"Found: {len(matching_corrected)}"
        )

    # The original value must no longer exist in this target column.
    remaining_original = [
        row_index
        for row_index, row in enumerate(verify_rows)
        if row[verify_column_index] == correction["original_value"]
    ]

    if remaining_original:
        raise ValueError(
            "Original suspicious value still exists after correction.\n"
            f"File: {relative_path}\n"
            f"Remaining data-row indices: {remaining_original}"
        )

    return {
        "relative_path": relative_path,
        "participant": correction["participant"],
        "column": correction["column"],
        "original_value": correction["original_value"],
        "corrected_value": correction["corrected_value"],
        "raw_sha256": sha256_file(raw_path),
        "derived_sha256": sha256_file(derived_path),
        "raw_data_row_index_zero_based": raw_target_index,
        "derived_data_row_index_zero_based": derived_target_index,
        "status": "APPLIED_AND_VERIFIED",
    }


# ============================================================================
# DERIVED DATASET CREATION
# ============================================================================

def create_fresh_derived_dataset() -> Dict[str, Path]:
    """
    Replace the existing derived dataset with a fresh copy of raw.

    The operation uses a temporary directory first, then replaces the old
    derived directory. Raw remains untouched.
    """
    print()
    print("-" * 80)
    print("CREATING CLEAN DERIVED DATASET")
    print("-" * 80)

    print(f"Raw dataset:     {RAW_DATASET}")
    print(f"Derived dataset: {DERIVED_DATASET}")
    print()

    if DERIVED_DATASET.exists():
        print("Existing derived dataset found.")
        print("It will be replaced with a fresh byte-for-byte copy of raw.")

    parent = DERIVED_DATASET.parent
    parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=".t1d_uom_derived_tmp_",
            dir=str(parent),
        )
    )

    try:
        temp_target = temp_dir / DERIVED_DATASET.name

        shutil.copytree(
            RAW_DATASET,
            temp_target,
            copy_function=shutil.copy2,
        )

        # Verify temporary copy before touching the existing derived dataset.
        raw_files = file_map(RAW_DATASET)
        temp_files = file_map(temp_target)

        if set(raw_files) != set(temp_files):
            missing = sorted(set(raw_files) - set(temp_files))
            extra = sorted(set(temp_files) - set(raw_files))

            raise RuntimeError(
                "Temporary derived copy does not preserve the raw file list.\n"
                f"Missing: {missing}\n"
                f"Extra:   {extra}"
            )

        for relative_path in raw_files:
            raw_hash = sha256_file(raw_files[relative_path])
            temp_hash = sha256_file(temp_files[relative_path])

            if raw_hash != temp_hash:
                raise RuntimeError(
                    "Temporary derived copy is not byte-identical to raw:\n"
                    f"{relative_path}"
                )

        # Only after successful verification do we replace the old derived set.
        if DERIVED_DATASET.exists():
            shutil.rmtree(DERIVED_DATASET)

        temp_target.rename(DERIVED_DATASET)

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    derived_files = file_map(DERIVED_DATASET)

    print("Fresh raw -> derived copy completed.")
    print("Initial file-list preservation: PASS")
    print("Initial byte-for-byte verification: PASS")

    return derived_files


# ============================================================================
# FINAL DATASET VERIFICATION
# ============================================================================

def verify_final_dataset(
    raw_files: Dict[str, Path],
    derived_files: Dict[str, Path],
    raw_hashes: Dict[str, str],
    correction_results: List[Dict[str, object]],
) -> Dict[str, object]:
    """Verify the complete raw/derived relationship."""
    print()
    print("-" * 80)
    print("FINAL DATASET VERIFICATION")
    print("-" * 80)

    if set(raw_files) != set(derived_files):
        missing = sorted(set(raw_files) - set(derived_files))
        extra = sorted(set(derived_files) - set(raw_files))

        raise ValueError(
            "Raw/derived CSV file lists differ.\n"
            f"Missing from derived: {missing}\n"
            f"Extra in derived: {extra}"
        )

    changed_files = []
    identical_files = []

    for relative_path in sorted(raw_files):
        raw_hash = raw_hashes[relative_path]
        derived_hash = sha256_file(derived_files[relative_path])

        if raw_hash == derived_hash:
            identical_files.append(relative_path)
        else:
            changed_files.append(relative_path)

    approved_paths = {
        correction["relative_path"]
        for correction in APPROVED_CORRECTIONS
    }

    changed_set = set(changed_files)

    if changed_set != approved_paths:
        unexpected = sorted(changed_set - approved_paths)
        missing_changes = sorted(approved_paths - changed_set)

        raise ValueError(
            "Derived dataset changed files do not exactly match the "
            "approved correction targets.\n"
            f"Unexpected changed files: {unexpected}\n"
            f"Approved files not changed: {missing_changes}"
        )

    if len(identical_files) != 110:
        raise ValueError(
            "Expected exactly 110 byte-identical files after the two "
            f"approved corrections, found {len(identical_files)}."
        )

    if len(changed_files) != 2:
        raise ValueError(
            f"Expected exactly 2 changed files, found {len(changed_files)}."
        )

    # Verify raw hashes have not changed since initial capture.
    for relative_path, original_hash in raw_hashes.items():
        current_hash = sha256_file(raw_files[relative_path])

        if current_hash != original_hash:
            raise ValueError(
                "RAW DATASET HASH CHANGED DURING CORRECTION.\n"
                f"File: {relative_path}\n"
                "This should never happen."
            )

    print(f"Raw CSV files:       {len(raw_files)}")
    print(f"Derived CSV files:   {len(derived_files)}")
    print("CSV file-list preservation: PASS")
    print(f"Byte-identical CSV files: {len(identical_files)}")
    print(f"Changed CSV files:       {len(changed_files)}")

    print()
    print("Changed files:")
    for relative_path in changed_files:
        print(f"  - {relative_path}")

    print()
    print("RAW DATASET PRESERVATION: PASS")
    print("APPROVED CHANGE SET: PASS")

    return {
        "raw_csv_files": len(raw_files),
        "derived_csv_files": len(derived_files),
        "byte_identical_csv_files": len(identical_files),
        "changed_csv_files": len(changed_files),
        "changed_files": changed_files,
        "raw_dataset_preserved": True,
        "approved_change_set_verified": True,
    }


# ============================================================================
# MANIFEST
# ============================================================================

def write_manifest(
    raw_hashes: Dict[str, str],
    correction_results: List[Dict[str, object]],
    final_verification: Dict[str, object],
) -> None:
    """Write provenance manifest only after successful verification."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "manifest_version": "2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "raw_dataset": str(RAW_DATASET),
        "derived_dataset": str(DERIVED_DATASET),
        "raw_dataset_modified": False,
        "correction_method": (
            "Exact file + exact column + exact original-value matching. "
            "No row-number-based targeting."
        ),
        "approved_corrections": APPROVED_CORRECTIONS,
        "correction_results": correction_results,
        "raw_sha256_file_count": len(raw_hashes),
        "final_verification": final_verification,
        "status": "SUCCESS",
    }

    temp_manifest = MANIFEST_PATH.with_suffix(".tmp")

    with temp_manifest.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    temp_manifest.replace(MANIFEST_PATH)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print()
    print("=" * 80)
    print("T1D-UOM EVIDENCE-BASED TIMESTAMP CORRECTION")
    print("=" * 80)
    print()
    print(f"Project root:   {PROJECT_ROOT}")
    print(f"Raw dataset:    {RAW_DATASET}")
    print(f"Derived dataset:{DERIVED_DATASET}")
    print(f"Manifest:       {MANIFEST_PATH}")
    print()
    print("SAFETY: RAW DATASET WILL NOT BE MODIFIED.")
    print()
    print("Approved corrections:")
    print()

    for correction in APPROVED_CORRECTIONS:
        print(f"  {correction['relative_path']}")
        print(f"      Column:     {correction['column']}")
        print(f"      Original:   {correction['original_value']}")
        print(f"      Corrected:  {correction['corrected_value']}")
        print()

    try:
        # 1. Validate raw and capture hashes.
        raw_files, raw_hashes = validate_raw_dataset()

        # 2. Create a completely fresh derived dataset.
        derived_files = create_fresh_derived_dataset()

        # 3. Apply ONLY the two approved corrections.
        print()
        print("-" * 80)
        print("APPLYING APPROVED CORRECTIONS")
        print("-" * 80)

        correction_results = []

        for correction in APPROVED_CORRECTIONS:
            result = validate_and_apply_one_correction(
                correction,
                raw_files,
                derived_files,
            )
            correction_results.append(result)

        # 4. Rebuild file map after edits.
        derived_files = file_map(DERIVED_DATASET)

        # 5. Full final verification.
        final_verification = verify_final_dataset(
            raw_files,
            derived_files,
            raw_hashes,
            correction_results,
        )

        # 6. Write manifest only after all verification passes.
        write_manifest(
            raw_hashes,
            correction_results,
            final_verification,
        )

        print()
        print("=" * 80)
        print("TIMESTAMP CORRECTION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        print("Raw dataset:       UNMODIFIED")
        print("Derived dataset:   CREATED AND VERIFIED")
        print("Approved changes:  2")
        print("Unaffected files:  110 byte-identical")
        print()
        print(f"Manifest saved to:")
        print(f"{MANIFEST_PATH}")
        print()
        print("NEXT STEP:")
        print("Run:")
        print()
        print("    python src\\data\\validate_timestamp_corrections.py")
        print()
        print("Then run:")
        print()
        print("    python src\\data\\audit_dataset.py")
        print()

    except Exception as exc:
        fail(str(exc))


if __name__ == "__main__":
    main()