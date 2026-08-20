from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# =============================================================================
# T1D-UOM MODELING DATASET FREEZE
# =============================================================================
#
# PURPOSE
# -------
# Create the controlled modeling dataset from the already verified,
# timestamp-corrected dataset.
#
# THIS SCRIPT DOES ONLY ONE JOB:
#     Freeze the modeling dataset.
#
# IT DOES NOT:
#     - modify raw data
#     - modify the timestamp-corrected dataset
#     - repair schemas
#     - rename columns
#     - remove rows from retained files
#     - impute missing values
#     - interpolate timestamps
#     - resample data
#     - normalize data
#     - generate sequences
#     - create training windows
#     - perform feature engineering
#     - train models
#
# FROZEN ARCHITECTURE:
#
#     Glucose   -> GRU
#     Insulin   -> GRU
#     Nutrition -> GRU
#     Activity  -> GRU
#     Sleep     -> GRU
#                      |
#                      v
#                  MLP Fusion
#
# DATASET POLICY
# --------------
# Exactly TWO files are excluded because they were explicitly identified
# as schema outliers:
#
#   1. Insulin Data/Basal Data/UoMBasal2301.csv
#   2. Sleep Data/UoM2301sleeptime.csv
#
# The previously frozen 13-participant modeling cohort remains FROZEN.
#
# Two additional participants, UoM2320 and UoM2404, become eligible for
# all five project-level modalities after the approved timestamp corrections.
# They are RECORDED as additional eligible participants but are NOT silently
# added to the frozen cohort.
#
# This distinction is deliberate and preserves experimental reproducibility.
#
# SAFETY
# ------
# Raw dataset:
#     NEVER modified.
#
# Timestamp-corrected dataset:
#     NEVER modified.
#
# Modeling dataset:
#     Created from a fresh byte-for-byte copy of the timestamp-corrected
#     dataset, followed only by the two approved file exclusions.
#
# =============================================================================


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DATASET = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "t1d_uom_v1.0.3_timestamp_corrected"
)

MODELING_DATASET = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "t1d_uom_v1.0.3_modeling"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"

EXCLUSION_MANIFEST = (
    REPORT_DIR / "modeling_dataset_exclusions.json"
)

DATASET_MANIFEST = (
    REPORT_DIR / "modeling_dataset_manifest.json"
)


# =============================================================================
# 2. APPROVED FILE EXCLUSIONS
# =============================================================================
#
# ONLY these two files may be excluded.
# =============================================================================

APPROVED_EXCLUSIONS = {
    Path("Insulin Data")
    / "Basal Data"
    / "UoMBasal2301.csv": {
        "participant": "UoM2301",
        "modality": "basal_insulin",
        "reason": (
            "Excluded from the modeling dataset because this file has a "
            "distinct basal-insulin schema from the established "
            "basal-insulin modality-family schema. The participant remains "
            "eligible because bolus-insulin data are available."
        ),
    },

    Path("Sleep Data")
    / "UoM2301sleeptime.csv": {
        "participant": "UoM2301",
        "modality": "sleep_summary",
        "reason": (
            "Excluded from the modeling dataset because this file has a "
            "distinct sleep-summary schema from the established "
            "sleep-summary modality-family schema. The participant remains "
            "eligible because sleep time-series data are available."
        ),
    },
}


# =============================================================================
# 3. FROZEN MODELING COHORT
# =============================================================================
#
# This cohort is intentionally NOT changed by newly discovered eligibility.
#
# IMPORTANT:
# UoM2320 and UoM2404 are NOT silently added.
# They are recorded separately as additional eligible participants.
# =============================================================================

FROZEN_FIVE_MODALITY_COHORT = [
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


# =============================================================================
# 4. EXPECTED COUNTS
# =============================================================================

EXPECTED_SOURCE_CSV_COUNT = 112
EXPECTED_EXCLUSION_COUNT = 2
EXPECTED_MODELING_CSV_COUNT = 110


# =============================================================================
# 5. UTILITY FUNCTIONS
# =============================================================================

def fail(message: str) -> None:
    """Abort safely without modifying source datasets."""

    print()
    print("=" * 80)
    print("MODEL DATASET FREEZE FAILED")
    print("=" * 80)
    print(message)
    print()
    print("IMPORTANT:")
    print("  Raw dataset was NOT modified.")
    print("  Timestamp-corrected dataset was NOT modified.")
    print("=" * 80)

    sys.exit(1)


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 without loading the complete file into memory."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def discover_csv_files(root: Path) -> list[Path]:
    """Return all CSV paths relative to root."""

    return sorted(
        path.relative_to(root)
        for path in root.rglob("*.csv")
        if path.is_file()
    )


def participant_from_filename(filename: str) -> str | None:
    """
    Extract participant ID from known UoM filename patterns.

    Examples:
        UoMGlucose2301.csv
        UoMBasal2301.csv
        UoMBolus2301.csv
        UoMNutrition2301.csv
        UoMActivity2301.csv
        UoM2301sleeptime.csv
        UoMsleep2301.csv
    """

    match = re.search(
        r"(2301|2302|2303|2304|2305|2306|2307|2308|2309|"
        r"2310|2312|2313|2314|2315|2320|2401|2403|2404|2405)",
        filename,
    )

    if match:
        return f"UoM{match.group(1)}"

    return None


def classify_modality(relative_path: Path) -> str:
    """Classify a known dataset path into the project modality."""

    parts = [part.lower() for part in relative_path.parts]
    filename = relative_path.name.lower()

    if "glucose data" in parts:
        return "glucose"

    if "activity data" in parts:
        return "activity"

    if "nutrition data" in parts:
        return "nutrition"

    if "insulin data" in parts:
        if "basal data" in parts:
            return "basal_insulin"

        if "bolus data" in parts:
            return "bolus_insulin"

        return "insulin"

    if "sleep data" in parts:
        if "sleeptime" in filename:
            return "sleep_summary"

        if "sleep" in filename:
            return "sleep_timeseries"

        return "sleep"

    return "unknown"


def atomic_json_write(path: Path, payload: dict) -> None:
    """Write JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    temporary.replace(path)


def replace_previous_generated_modeling_dataset() -> None:
    """
    Remove a previous modeling dataset only when its safety marker proves
    that it was generated by this script.
    """

    if not MODELING_DATASET.exists():
        return

    marker = MODELING_DATASET / ".t1d_uom_modeling_dataset"

    if not marker.is_file():
        fail(
            "A modeling dataset directory already exists, but it does not "
            "contain the expected safety marker.\n\n"
            f"Directory:\n{MODELING_DATASET}\n\n"
            "For safety, this script refuses to delete it automatically."
        )

    print("Existing generated modeling dataset found.")
    print("Safety marker verified.")
    print("Replacing it with a fresh dataset...")

    shutil.rmtree(MODELING_DATASET)

    if MODELING_DATASET.exists():
        fail(
            "The previous generated modeling dataset could not be removed."
        )


# =============================================================================
# 6. START
# =============================================================================

print()
print("=" * 80)
print("T1D-UOM MODELING DATASET FREEZE")
print("=" * 80)

print()
print(f"Project root:        {PROJECT_ROOT}")
print(f"Source dataset:      {SOURCE_DATASET}")
print(f"Modeling dataset:    {MODELING_DATASET}")
print(f"Exclusion manifest:  {EXCLUSION_MANIFEST}")
print(f"Dataset manifest:    {DATASET_MANIFEST}")

print()
print("SAFETY:")
print("  Raw dataset will NOT be modified.")
print("  Timestamp-corrected dataset will NOT be modified.")
print("  Retained CSV contents will NOT be transformed.")


# =============================================================================
# 7. SOURCE DATASET VALIDATION
# =============================================================================

print()
print("-" * 80)
print("1. SOURCE DATASET VALIDATION")
print("-" * 80)

if not SOURCE_DATASET.is_dir():
    fail(
        "Source dataset does not exist:\n"
        f"{SOURCE_DATASET}"
    )

source_files = discover_csv_files(SOURCE_DATASET)

print(f"Source CSV files discovered: {len(source_files)}")

if len(source_files) != EXPECTED_SOURCE_CSV_COUNT:
    fail(
        f"Expected {EXPECTED_SOURCE_CSV_COUNT} source CSV files, "
        f"but found {len(source_files)}."
    )

print("Source file count: PASS")


# =============================================================================
# 8. APPROVED EXCLUSION VALIDATION
# =============================================================================

print()
print("-" * 80)
print("2. APPROVED EXCLUSION VALIDATION")
print("-" * 80)

if len(APPROVED_EXCLUSIONS) != EXPECTED_EXCLUSION_COUNT:
    fail(
        "Internal exclusion configuration is inconsistent.\n"
        f"Expected {EXPECTED_EXCLUSION_COUNT} exclusions, "
        f"configured {len(APPROVED_EXCLUSIONS)}."
    )

source_set = set(source_files)
exclusion_set = set(APPROVED_EXCLUSIONS)

missing_exclusions = exclusion_set - source_set

if missing_exclusions:
    details = "\n".join(
        f"  - {path.as_posix()}"
        for path in sorted(missing_exclusions)
    )

    fail(
        "Approved exclusion file(s) were not found in the source dataset:\n"
        f"{details}"
    )

print("Approved exclusions:")

for relative_path, metadata in sorted(APPROVED_EXCLUSIONS.items()):
    print()
    print(f"  {relative_path.as_posix()}")
    print(f"      Participant: {metadata['participant']}")
    print(f"      Modality:    {metadata['modality']}")
    print(f"      Reason:      {metadata['reason']}")


# =============================================================================
# 9. CAPTURE SOURCE HASHES
# =============================================================================

print()
print("-" * 80)
print("3. SOURCE SHA-256 HASH CAPTURE")
print("-" * 80)

source_hashes: dict[str, str] = {}

for index, relative_path in enumerate(source_files, start=1):

    source_path = SOURCE_DATASET / relative_path

    print(
        f"[{index:03d}/{len(source_files)}] "
        f"{relative_path.as_posix()}"
    )

    source_hashes[relative_path.as_posix()] = sha256_file(
        source_path
    )

print()
print(
    f"Captured SHA-256 hashes for "
    f"{len(source_hashes)} source CSV files."
)


# =============================================================================
# 10. CREATE FRESH MODELING DATASET
# =============================================================================

print()
print("-" * 80)
print("4. CREATING MODELING DATASET")
print("-" * 80)

replace_previous_generated_modeling_dataset()

print("Creating byte-for-byte copy of source dataset...")

shutil.copytree(
    SOURCE_DATASET,
    MODELING_DATASET,
)

print("Source -> modeling copy: PASS")


# =============================================================================
# 11. WRITE SAFETY MARKER
# =============================================================================

marker_path = MODELING_DATASET / ".t1d_uom_modeling_dataset"

marker_payload = {
    "project": "T1D-UOM",
    "purpose": "Controlled modeling dataset",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_dataset": str(SOURCE_DATASET),
    "approved_exclusions": [
        path.as_posix()
        for path in sorted(exclusion_set)
    ],
}

with marker_path.open(
    "w",
    encoding="utf-8",
    newline="\n",
) as handle:
    json.dump(
        marker_payload,
        handle,
        indent=2,
    )
    handle.write("\n")


# =============================================================================
# 12. APPLY ONLY APPROVED EXCLUSIONS
# =============================================================================

print()
print("-" * 80)
print("5. APPLYING APPROVED FILE EXCLUSIONS")
print("-" * 80)

for relative_path, metadata in sorted(
    APPROVED_EXCLUSIONS.items()
):

    target_path = MODELING_DATASET / relative_path

    if not target_path.is_file():
        fail(
            "Expected exclusion target does not exist in the copied "
            "modeling dataset:\n"
            f"{target_path}"
        )

    print()
    print(f"Excluding: {relative_path.as_posix()}")
    print(f"Participant: {metadata['participant']}")
    print(f"Modality:    {metadata['modality']}")
    print(f"Reason:      {metadata['reason']}")

    target_path.unlink()


# =============================================================================
# 13. FINAL FILE STRUCTURE VALIDATION
# =============================================================================

print()
print("-" * 80)
print("6. MODELING DATASET STRUCTURE VALIDATION")
print("-" * 80)

modeling_files = discover_csv_files(MODELING_DATASET)

print(f"Modeling CSV files discovered: {len(modeling_files)}")
print(
    f"Expected modeling CSV files:   "
    f"{EXPECTED_MODELING_CSV_COUNT}"
)

if len(modeling_files) != EXPECTED_MODELING_CSV_COUNT:
    fail(
        f"Expected {EXPECTED_MODELING_CSV_COUNT} modeling CSV files, "
        f"but found {len(modeling_files)}."
    )

modeling_set = set(modeling_files)

expected_modeling_set = source_set - exclusion_set

missing_expected = expected_modeling_set - modeling_set
unexpected_files = modeling_set - expected_modeling_set

if missing_expected:
    details = "\n".join(
        f"  - {path.as_posix()}"
        for path in sorted(missing_expected)
    )

    fail(
        "Unexpected missing retained file(s):\n"
        f"{details}"
    )

if unexpected_files:
    details = "\n".join(
        f"  - {path.as_posix()}"
        for path in sorted(unexpected_files)
    )

    fail(
        "Unexpected additional file(s) in modeling dataset:\n"
        f"{details}"
    )

print(
    "Exact source-minus-approved-exclusions file list: PASS"
)


# =============================================================================
# 14. BYTE-LEVEL VERIFICATION
# =============================================================================

print()
print("-" * 80)
print("7. RETAINED FILE BYTE-LEVEL VERIFICATION")
print("-" * 80)

changed_retained_files: list[str] = []

for relative_path in modeling_files:

    key = relative_path.as_posix()

    source_hash = source_hashes.get(key)

    if source_hash is None:
        fail(
            "No source SHA-256 hash exists for retained file:\n"
            f"{key}"
        )

    modeling_hash = sha256_file(
        MODELING_DATASET / relative_path
    )

    if source_hash != modeling_hash:
        changed_retained_files.append(key)

if changed_retained_files:

    details = "\n".join(
        f"  - {path}"
        for path in changed_retained_files
    )

    fail(
        "Retained files were modified unexpectedly:\n"
        f"{details}"
    )

print(
    f"Byte-identical retained CSV files: "
    f"{len(modeling_files)}"
)

print("Retained-file byte identity: PASS")


# =============================================================================
# 15. VERIFY EXACT EXCLUSIONS
# =============================================================================

print()
print("-" * 80)
print("8. EXCLUSION EXACTNESS VERIFICATION")
print("-" * 80)

still_present = exclusion_set & modeling_set

if still_present:

    details = "\n".join(
        f"  - {path.as_posix()}"
        for path in sorted(still_present)
    )

    fail(
        "Approved exclusion file(s) are still present:\n"
        f"{details}"
    )

print("Excluded files absent: PASS")
print("No unexpected exclusions: PASS")


# =============================================================================
# 16. PARTICIPANT / MODALITY COVERAGE
# =============================================================================

print()
print("-" * 80)
print("9. MODELING DATASET PARTICIPANT / MODALITY COVERAGE")
print("-" * 80)

coverage: dict[str, set[str]] = defaultdict(set)

for relative_path in modeling_files:

    participant = participant_from_filename(
        relative_path.name
    )

    modality = classify_modality(
        relative_path
    )

    if participant is None:
        fail(
            "Could not identify participant from filename:\n"
            f"{relative_path.as_posix()}"
        )

    if modality == "unknown":
        fail(
            "Could not identify modality from path:\n"
            f"{relative_path.as_posix()}"
        )

    # Project-level insulin modality.
    if modality in {
        "basal_insulin",
        "bolus_insulin",
    }:
        coverage[participant].add("insulin")

    # Project-level sleep modality.
    elif modality in {
        "sleep_summary",
        "sleep_timeseries",
    }:
        coverage[participant].add("sleep")

    else:
        coverage[participant].add(modality)


required_modalities = {
    "glucose",
    "insulin",
    "nutrition",
    "activity",
    "sleep",
}


actual_eligible_cohort = sorted(
    participant
    for participant in coverage
    if required_modalities.issubset(
        coverage[participant]
    )
)

actual_eligible_set = set(actual_eligible_cohort)
frozen_cohort_set = set(
    FROZEN_FIVE_MODALITY_COHORT
)


print()
print("Frozen five-project-modality cohort:")

for participant in FROZEN_FIVE_MODALITY_COHORT:

    modalities = sorted(
        coverage.get(
            participant,
            set(),
        )
    )

    print(
        f"  {participant}: "
        f"{', '.join(modalities) if modalities else 'NONE'}"
    )


# =============================================================================
# 17. FROZEN COHORT VALIDATION
# =============================================================================
#
# IMPORTANT LOGIC:
#
# The frozen cohort must remain present.
#
# Additional eligible participants are NOT an error.
# They are explicitly recorded.
# =============================================================================

missing_from_frozen = (
    frozen_cohort_set
    - actual_eligible_set
)

additional_eligible = (
    actual_eligible_set
    - frozen_cohort_set
)

if missing_from_frozen:

    details = "\n".join(
        f"  - {participant}"
        for participant in sorted(
            missing_from_frozen
        )
    )

    fail(
        "A participant from the frozen modeling cohort "
        "no longer has all five project modalities.\n\n"
        "Affected participants:\n"
        f"{details}"
    )


print()
print(
    "Frozen five-modality cohort consistency: PASS"
)

print(
    f"Frozen cohort size: "
    f"{len(frozen_cohort_set)} participants"
)


if additional_eligible:

    print()
    print(
        "Additional five-modality participants detected:"
    )

    print(
        "These participants are eligible in the corrected "
        "dataset but are NOT added to the frozen cohort."
    )

    for participant in sorted(
        additional_eligible
    ):
        print(
            f"  - {participant}"
        )

    print()
    print(
        "Additional eligibility recorded: PASS"
    )

else:

    print()
    print(
        "Additional five-modality participants: NONE"
    )


# =============================================================================
# 18. SPECIFIC UoM2301 IMPACT CHECK
# =============================================================================

print()
print("-" * 80)
print("10. APPROVED EXCLUSION IMPACT CHECK")
print("-" * 80)

uom2301_modalities = coverage.get(
    "UoM2301",
    set(),
)

missing_uom2301 = (
    required_modalities
    - uom2301_modalities
)

if missing_uom2301:

    details = "\n".join(
        f"  - {modality}"
        for modality in sorted(
            missing_uom2301
        )
    )

    fail(
        "UoM2301 unexpectedly lost a required "
        "project-level modality:\n"
        f"{details}"
    )

print(
    "UoM2301 remains five-modality eligible: PASS"
)

print(
    "Available modalities: "
    + ", ".join(
        sorted(uom2301_modalities)
    )
)


# =============================================================================
# 19. WRITE EXCLUSION MANIFEST
# =============================================================================

print()
print("-" * 80)
print("11. WRITING EXCLUSION MANIFEST")
print("-" * 80)

exclusion_records = []

for relative_path, metadata in sorted(
    APPROVED_EXCLUSIONS.items()
):

    source_path = (
        SOURCE_DATASET
        / relative_path
    )

    exclusion_records.append(
        {
            "relative_path": (
                relative_path.as_posix()
            ),
            "participant": (
                metadata["participant"]
            ),
            "modality": (
                metadata["modality"]
            ),
            "reason": (
                metadata["reason"]
            ),
            "source_sha256": (
                source_hashes[
                    relative_path.as_posix()
                ]
            ),
            "source_file_size_bytes": (
                source_path.stat().st_size
            ),
        }
    )


exclusion_manifest = {
    "project": "T1D-UOM",
    "stage": "modeling_dataset_exclusion",
    "created_utc": (
        datetime.now(
            timezone.utc
        ).isoformat()
    ),

    "source_dataset": str(
        SOURCE_DATASET
    ),

    "modeling_dataset": str(
        MODELING_DATASET
    ),

    "raw_dataset_modified": False,
    "source_dataset_modified": False,

    "approved_exclusion_count": (
        len(exclusion_records)
    ),

    "approved_exclusions": (
        exclusion_records
    ),

    "decision": (
        "Exactly two explicitly approved schema-outlier files "
        "are excluded from the modeling dataset. No other files "
        "are excluded."
    ),
}


atomic_json_write(
    EXCLUSION_MANIFEST,
    exclusion_manifest,
)

print(
    "Exclusion manifest saved:"
)

print(
    f"  {EXCLUSION_MANIFEST}"
)


# =============================================================================
# 20. WRITE MODELING DATASET MANIFEST
# =============================================================================

print()
print("-" * 80)
print("12. WRITING MODELING DATASET MANIFEST")
print("-" * 80)

retained_records = []

for relative_path in modeling_files:

    key = relative_path.as_posix()

    modeling_path = (
        MODELING_DATASET
        / relative_path
    )

    modeling_hash = sha256_file(
        modeling_path
    )

    retained_records.append(
        {
            "relative_path": key,

            "participant": (
                participant_from_filename(
                    relative_path.name
                )
            ),

            "modality": (
                classify_modality(
                    relative_path
                )
            ),

            "source_sha256": (
                source_hashes[key]
            ),

            "modeling_sha256": (
                modeling_hash
            ),

            "byte_identical": (
                source_hashes[key]
                == modeling_hash
            ),

            "size_bytes": (
                modeling_path.stat().st_size
            ),
        }
    )


dataset_manifest = {

    "project": "T1D-UOM",

    "stage": (
        "modeling_dataset_freeze"
    ),

    "created_utc": (
        datetime.now(
            timezone.utc
        ).isoformat()
    ),

    "source_dataset": str(
        SOURCE_DATASET
    ),

    "modeling_dataset": str(
        MODELING_DATASET
    ),

    "raw_dataset_modified": False,

    "source_dataset_modified": False,

    "source_csv_count": (
        len(source_files)
    ),

    "excluded_csv_count": (
        len(exclusion_set)
    ),

    "modeling_csv_count": (
        len(modeling_files)
    ),

    "excluded_files": [
        path.as_posix()
        for path in sorted(
            exclusion_set
        )
    ],

    "frozen_five_modality_cohort": (
        FROZEN_FIVE_MODALITY_COHORT
    ),

    "additional_eligible_participants": (
        sorted(
            additional_eligible
        )
    ),

    "eligible_five_modality_participants": (
        actual_eligible_cohort
    ),

    "retained_files": (
        retained_records
    ),

    "architecture": {
        "glucose": "GRU",
        "insulin": "GRU",
        "nutrition": "GRU",
        "activity": "GRU",
        "sleep": "GRU",
        "fusion": "MLP",
    },

    "verification": {

        "source_file_count_pass": (
            len(source_files)
            == EXPECTED_SOURCE_CSV_COUNT
        ),

        "exclusion_count_pass": (
            len(exclusion_set)
            == EXPECTED_EXCLUSION_COUNT
        ),

        "modeling_file_count_pass": (
            len(modeling_files)
            == EXPECTED_MODELING_CSV_COUNT
        ),

        "exact_exclusion_set_pass": (
            modeling_set
            == expected_modeling_set
        ),

        "retained_files_byte_identical": (
            len(changed_retained_files)
            == 0
        ),

        "frozen_cohort_pass": (
            not missing_from_frozen
        ),

        "additional_eligibility_recorded": True,

        "uom2301_five_modality_pass": (
            not missing_uom2301
        ),
    },
}


atomic_json_write(
    DATASET_MANIFEST,
    dataset_manifest,
)

print(
    "Dataset manifest saved:"
)

print(
    f"  {DATASET_MANIFEST}"
)


# =============================================================================
# 21. FINAL VERIFICATION
# =============================================================================

print()
print("-" * 80)
print("13. FINAL FREEZE VERIFICATION")
print("-" * 80)

final_checks = {

    "source_count": (
        len(source_files)
        == EXPECTED_SOURCE_CSV_COUNT
    ),

    "exclusion_count": (
        len(exclusion_set)
        == EXPECTED_EXCLUSION_COUNT
    ),

    "modeling_count": (
        len(modeling_files)
        == EXPECTED_MODELING_CSV_COUNT
    ),

    "exact_file_set": (
        modeling_set
        == expected_modeling_set
    ),

    "retained_byte_identity": (
        len(changed_retained_files)
        == 0
    ),

    "frozen_cohort_preserved": (
        not missing_from_frozen
    ),

    "uom2301_preserved": (
        not missing_uom2301
    ),
}


for check_name, passed in final_checks.items():

    print(
        f"  {check_name}: "
        f"{'PASS' if passed else 'FAIL'}"
    )


if not all(final_checks.values()):

    failed_checks = [
        name
        for name, passed
        in final_checks.items()
        if not passed
    ]

    fail(
        "One or more final verification checks failed:\n"
        + "\n".join(
            f"  - {name}"
            for name in failed_checks
        )
    )


# =============================================================================
# 22. FINAL SUCCESS
# =============================================================================

print()
print("=" * 80)
print("MODEL DATASET FREEZE COMPLETED SUCCESSFULLY")
print("=" * 80)

print()
print("SOURCE DATASET")
print(
    f"  CSV files:                 "
    f"{len(source_files)}"
)
print("  Modified:                  NO")

print()
print("APPROVED EXCLUSIONS")
print(
    f"  Excluded files:            "
    f"{len(exclusion_set)}"
)

for relative_path in sorted(
    exclusion_set
):
    print(
        f"    - {relative_path.as_posix()}"
    )

print()
print("MODELING DATASET")
print(
    f"  CSV files retained:        "
    f"{len(modeling_files)}"
)
print(
    "  Retained files changed:    0"
)
print(
    "  Byte-identical retained:   PASS"
)

print()
print("FROZEN FIVE-MODALITY COHORT")
print(
    f"  Participants:              "
    f"{len(frozen_cohort_set)}"
)

for participant in FROZEN_FIVE_MODALITY_COHORT:
    print(
        f"    - {participant}"
    )

print()
print(
    "ADDITIONAL ELIGIBLE PARTICIPANTS"
)

if additional_eligible:

    for participant in sorted(
        additional_eligible
    ):
        print(
            f"    - {participant}"
        )

    print(
        "  These participants were NOT added "
        "to the frozen cohort."
    )

else:

    print(
        "    None"
    )


print()
print("ARCHITECTURE UNCHANGED")
print("  Glucose   -> GRU")
print("  Insulin   -> GRU")
print("  Nutrition -> GRU")
print("  Activity  -> GRU")
print("  Sleep     -> GRU")
print("  Fusion    -> MLP")

print()
print("MANIFESTS")
print(
    f"  Exclusions: "
    f"{EXCLUSION_MANIFEST}"
)
print(
    f"  Dataset:    "
    f"{DATASET_MANIFEST}"
)

print()
print("IMPORTANT")
print(
    "  Raw dataset:                 PRESERVED"
)
print(
    "  Timestamp-corrected dataset: PRESERVED"
)
print(
    "  Modeling dataset:            FROZEN"
)
print(
    "  Unexpected exclusions:       NONE"
)
print(
    "  Unexpected transformations:  NONE"
)

print()
print("NEXT STAGE:")
print()
print(
    "  Modeling dataset freeze is complete."
)
print(
    "  The next stage is sequence preparation."
)
print(
    "  No further dataset correction is required "
    "at this stage."
)

print()
print("=" * 80)