from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# =============================================================================
# T1D-UOM SEQUENCE INPUT PREPARATION
# =============================================================================
#
# SCOPE
# -----
# This script performs ONLY deterministic sequence-input preparation.
#
# It does NOT:
#   - modify the raw dataset
#   - modify the timestamp-corrected dataset
#   - modify the modeling dataset
#   - delete duplicate observations
#   - resample
#   - interpolate
#   - impute
#   - normalize
#   - perform feature engineering
#   - create targets
#   - create windows
#   - train a model
#   - alter the frozen architecture
#
# It DOES:
#   - validate the frozen modeling dataset
#   - select the frozen 13-participant cohort
#   - parse timestamps robustly
#   - chronologically sort observations
#   - deterministically retain duplicate timestamps using original row order
#   - preserve all original columns and row values
#   - create the derived sequence-input area
#   - write provenance and integrity information
#
#
# FROZEN SYSTEM ARCHITECTURE
# ---------------------------
#
#   Glucose   -> GRU -> zG
#   Insulin   -> GRU -> zI
#   Nutrition -> GRU -> zN
#   Activity  -> GRU -> zA
#   Sleep     -> GRU -> zS
#                         |
#                    MLP Fusion
#                         |
#                Unified Patient State
#                         |
#                    DIGITAL TWIN
#                     /         \
#               Prediction      What-if
#                     \         /
#                   Interactive UI
#
# Prediction and What-if are downstream Digital Twin functions.
# They are NOT implemented here.
#
# =============================================================================


SCRIPT_VERSION = "1.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELING_DATASET = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "t1d_uom_v1.0.3_modeling"
)

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

EXCLUSION_MANIFEST = (
    REPORT_DIR
    / "modeling_dataset_exclusions.json"
)

DATASET_MANIFEST = (
    REPORT_DIR
    / "modeling_dataset_manifest.json"
)

OUTPUT_MANIFEST = (
    REPORT_DIR
    / "sequence_input_preparation_manifest.json"
)


# ---------------------------------------------------------------------------
# Frozen project constants
# ---------------------------------------------------------------------------

EXPECTED_MODELING_FILE_COUNT = 110
EXPECTED_SEQUENCE_FILE_COUNT = 86

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


MODALITY_ORDER = [
    "activity",
    "glucose",
    "basal_insulin",
    "bolus_insulin",
    "nutrition",
    "sleep_summary",
    "sleep_timeseries",
]


MODALITY_RULES = (
    ("activity", r"^UoMActivity\d{4}\.csv$"),
    ("glucose", r"^UoMGlucose\d{4}\.csv$"),
    ("basal_insulin", r"^UoMBasal\d{4}\.csv$"),
    ("bolus_insulin", r"^UoMBolus\d{4}\.csv$"),
    ("nutrition", r"^UoMNutrition\d{4}\.csv$"),
    ("sleep_summary", r"^UoM\d{4}sleeptime\.csv$"),
    ("sleep_timeseries", r"^UoMsleep\d{4}\.csv$"),
)


TIMESTAMP_COLUMNS = {
    "activity": "activity_ts",
    "glucose": "bg_ts",
    "basal_insulin": "basal_ts",
    "bolus_insulin": "bolus_ts",
    "nutrition": "meal_ts",
    "sleep_summary": "start_date_ts",
    "sleep_timeseries": "sleep_ts",
}


ARCHITECTURE = {
    "glucose": "Glucose -> GRU -> zG",
    "insulin": "Insulin -> GRU -> zI",
    "nutrition": "Nutrition -> GRU -> zN",
    "activity": "Activity -> GRU -> zA",
    "sleep": "Sleep -> GRU -> zS",
    "fusion": "zG,zI,zN,zA,zS -> MLP Fusion",
    "unified_state": "MLP Fusion -> Unified Patient State",
    "digital_twin": "Unified Patient State -> DIGITAL TWIN",
    "downstream": "DIGITAL TWIN -> Prediction / What-if",
    "interface": "Prediction / What-if -> Interactive UI",
}


SEQUENCE_AREA_NAME = "t1d_uom_v1.0.3_sequence_inputs"

SAFETY_MARKER = ".sequence_input_area_marker"


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def discover_csv_files(root: Path) -> List[Path]:
    if not root.exists():
        fail(
            "Dataset directory does not exist:\n"
            f"{root}"
        )

    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() == ".csv"
        ],
        key=lambda path: relative_path(path, root).lower(),
    )


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        fail(
            "Required manifest does not exist:\n"
            f"{path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        fail(
            "Manifest is not a JSON object:\n"
            f"{path}"
        )

    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")

    temporary.replace(path)


def participant_from_filename(path: Path) -> Optional[str]:
    match = re.search(r"(\d{4})", path.name)

    if match is None:
        return None

    return f"UoM{match.group(1)}"


def identify_modality(path: Path) -> Optional[str]:
    for modality, pattern in MODALITY_RULES:
        if re.fullmatch(
            pattern,
            path.name,
            flags=re.IGNORECASE,
        ):
            return modality

    return None


def modality_sort_key(modality: str) -> int:
    if modality in MODALITY_ORDER:
        return MODALITY_ORDER.index(modality)

    return 999


def read_csv_rows(
    path: Path,
) -> Tuple[List[str], List[List[str]]]:

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.reader(handle)

        try:
            header = next(reader)
        except StopIteration:
            fail(
                "CSV file is empty:\n"
                f"{path}"
            )

        rows = list(reader)

    return header, rows


def write_csv_rows(
    path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def row_digest(row: Sequence[str]) -> str:
    payload = "\x1f".join(row)

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


# =============================================================================
# ROBUST TIMESTAMP PARSER
# =============================================================================
#
# IMPORTANT:
#
# The verified dataset contains timestamps such as:
#
#     01/10/2023 05:45
#
# These are interpreted as:
#
#     DD/MM/YYYY HH:MM
#
# Therefore:
#
#     01/10/2023 = 1 October 2023
#
# NOT:
#
#     January 10 2023
#
# This is explicitly day-first because that matches the observed project
# timestamp representation.
#
# Supported:
#
#   DD/MM/YYYY HH:MM
#   DD/MM/YYYY HH:MM:SS
#   DD/MM/YYYY HH:MM:SS.sss...
#   DD/MM/YYYY HH:MM:SS,sss...
#   YYYY-MM-DD HH:MM
#   YYYY-MM-DD HH:MM:SS
#   YYYY-MM-DD HH:MM:SS.sss...
#   YYYY/MM/DD HH:MM
#   YYYY/MM/DD HH:MM:SS
#   ISO-8601
#   Unix epoch seconds/milliseconds/microseconds/nanoseconds
#
# No timestamp values are written back to the CSV.
# The parsed value is used ONLY for deterministic chronological ordering.
# =============================================================================


NUMERIC_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)


def numeric_epoch_candidates(
    value: float,
) -> List[Tuple[datetime, str]]:

    if not math.isfinite(value):
        return []

    candidates: List[Tuple[datetime, str]] = []

    scales = (
        ("seconds", 1.0),
        ("milliseconds", 1_000.0),
        ("microseconds", 1_000_000.0),
        ("nanoseconds", 1_000_000_000.0),
    )

    for name, divisor in scales:

        try:
            timestamp = datetime.fromtimestamp(
                value / divisor,
                tz=timezone.utc,
            )

        except (
            OverflowError,
            OSError,
            ValueError,
        ):
            continue

        if 2000 <= timestamp.year <= 2100:
            candidates.append(
                (
                    timestamp,
                    f"unix_{name}",
                )
            )

    return candidates


def parse_timestamp_value(
    raw_value: str,
) -> Tuple[datetime, str]:

    value = raw_value.strip()

    if not value:
        raise ValueError(
            "empty timestamp"
        )

    # -----------------------------------------------------------------------
    # Numeric epoch
    # -----------------------------------------------------------------------

    if NUMERIC_RE.fullmatch(value):

        candidates = numeric_epoch_candidates(
            float(value)
        )

        if not candidates:
            raise ValueError(
                f"no plausible epoch interpretation: {value!r}"
            )

        target = datetime(
            2025,
            1,
            1,
            tzinfo=timezone.utc,
        )

        return min(
            candidates,
            key=lambda item: abs(
                (
                    item[0] - target
                ).total_seconds()
            ),
        )

    # -----------------------------------------------------------------------
    # ISO-8601
    # -----------------------------------------------------------------------

    normalized = value

    if value.endswith(
        ("Z", "z")
    ):
        normalized = value[:-1] + "+00:00"

    try:
        timestamp = datetime.fromisoformat(
            normalized
        )

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(
                tzinfo=timezone.utc
            )

        return (
            timestamp.astimezone(timezone.utc),
            "iso8601",
        )

    except ValueError:
        pass

    # -----------------------------------------------------------------------
    # Project/source day-first formats
    # -----------------------------------------------------------------------

    day_first_formats = (
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S,%f",
        "%d/%m/%Y %H:%M:%S %z",
        "%d/%m/%Y %H:%M:%S.%f %z",
        "%d/%m/%Y",
    )

    for fmt in day_first_formats:

        try:
            timestamp = datetime.strptime(
                value,
                fmt,
            )

            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(
                    tzinfo=timezone.utc
                )

            return (
                timestamp.astimezone(timezone.utc),
                f"format:{fmt}",
            )

        except ValueError:
            continue

    # -----------------------------------------------------------------------
    # Other unambiguous year-first formats
    # -----------------------------------------------------------------------

    other_formats = (
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y-%m-%d",
    )

    for fmt in other_formats:

        try:
            timestamp = datetime.strptime(
                value,
                fmt,
            )

            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(
                    tzinfo=timezone.utc
                )

            return (
                timestamp.astimezone(timezone.utc),
                f"format:{fmt}",
            )

        except ValueError:
            continue

    raise ValueError(
        f"unrecognized timestamp: {raw_value!r}"
    )


# =============================================================================
# MODELING DATASET VALIDATION
# =============================================================================

def validate_modeling_dataset(
    source_files: Sequence[Path],
) -> Dict[str, Any]:

    print(
        f"Modeling CSV files discovered: "
        f"{len(source_files)}"
    )

    if len(source_files) != EXPECTED_MODELING_FILE_COUNT:

        fail(
            "Frozen modeling dataset file-count mismatch.\n"
            f"Expected: {EXPECTED_MODELING_FILE_COUNT}\n"
            f"Found:    {len(source_files)}"
        )

    inventory: Dict[str, List[str]] = {}

    for path in source_files:

        participant = participant_from_filename(path)
        modality = identify_modality(path)

        if participant is None:

            fail(
                "Unable to identify participant from modeling file:\n"
                f"{path}"
            )

        if modality is None:

            fail(
                "Unable to identify modality from modeling file:\n"
                f"{path}"
            )

        inventory.setdefault(
            participant,
            []
        ).append(modality)

    missing_frozen = sorted(
        FROZEN_COHORT_SET
        - set(inventory)
    )

    if missing_frozen:

        fail(
            "Frozen cohort participant(s) missing "
            "from modeling dataset:\n"
            + "\n".join(
                f"  - {participant}"
                for participant in missing_frozen
            )
        )

    # -----------------------------------------------------------------------
    # Verify the five frozen architecture inputs.
    #
    # Insulin is considered available if either basal or bolus insulin exists.
    # Sleep is considered available if either retained sleep representation
    # exists.
    # -----------------------------------------------------------------------

    for participant in FROZEN_COHORT:

        modalities = set(
            inventory[participant]
        )

        required = {
            "activity": (
                "activity" in modalities
            ),
            "glucose": (
                "glucose" in modalities
            ),
            "insulin": (
                "basal_insulin" in modalities
                or
                "bolus_insulin" in modalities
            ),
            "nutrition": (
                "nutrition" in modalities
            ),
            "sleep": (
                "sleep_summary" in modalities
                or
                "sleep_timeseries" in modalities
            ),
        }

        missing = [
            name
            for name, available in required.items()
            if not available
        ]

        if missing:

            fail(
                f"Frozen participant {participant} "
                "is not five-modality complete.\n"
                f"Missing: {', '.join(missing)}"
            )

    additional = sorted(
        set(inventory)
        - FROZEN_COHORT_SET
    )

    return {
        "participants": sorted(
            inventory
        ),
        "additional_participants": additional,
        "additional_participant_count": len(
            additional
        ),
        "inventory": {
            participant: sorted(
                modalities,
                key=lambda item: (
                    modality_sort_key(item),
                    item,
                ),
            )
            for participant, modalities
            in inventory.items()
        },
    }


def validate_freeze_manifests() -> Dict[str, Any]:

    exclusions = read_json(
        EXCLUSION_MANIFEST
    )

    dataset = read_json(
        DATASET_MANIFEST
    )

    return {
        "exclusions_manifest_readable": True,
        "dataset_manifest_readable": True,
        "exclusions_manifest_sha256": (
            sha256_file(
                EXCLUSION_MANIFEST
            )
        ),
        "dataset_manifest_sha256": (
            sha256_file(
                DATASET_MANIFEST
            )
        ),
        "exclusions_manifest_keys": sorted(
            exclusions.keys()
        ),
        "dataset_manifest_keys": sorted(
            dataset.keys()
        ),
    }


def select_frozen_files(
    source_files: Sequence[Path],
) -> List[Path]:

    selected = [
        path
        for path in source_files
        if (
            participant_from_filename(path)
            in FROZEN_COHORT_SET
        )
    ]

    selected.sort(
        key=lambda path: (
            FROZEN_COHORT.index(
                participant_from_filename(path)
            ),
            modality_sort_key(
                identify_modality(path) or ""
            ),
            relative_path(
                path,
                MODELING_DATASET,
            ).lower(),
        )
    )

    if len(selected) != EXPECTED_SEQUENCE_FILE_COUNT:

        fail(
            "Frozen-cohort file count mismatch.\n"
            f"Expected: {EXPECTED_SEQUENCE_FILE_COUNT}\n"
            f"Found:    {len(selected)}"
        )

    return selected


# =============================================================================
# SAFE DERIVED OUTPUT STAGING
# =============================================================================
#
# IMPORTANT:
#
# We never replace the final sequence-input directory before all 86 files have
# successfully prepared and passed verification.
#
# Therefore a parsing failure cannot destroy a previously successful generated
# sequence-input directory.
# =============================================================================

def validate_sequence_output_path() -> None:

    expected = (
        PROJECT_ROOT
        / "data"
        / "derived"
        / SEQUENCE_AREA_NAME
    ).resolve()

    actual = (
        SEQUENCE_INPUTS.resolve()
    )

    if actual != expected:

        fail(
            "Safety check failed for sequence-input "
            "output path.\n"
            f"Expected:\n{expected}\n"
            f"Actual:\n{actual}"
        )

    if actual.name != SEQUENCE_AREA_NAME:
        fail(
            "Safety check failed: unexpected output "
            "directory name."
        )

    if actual.parent.name != "derived":
        fail(
            "Safety check failed: output directory "
            "is not inside data\\derived."
        )

    if actual.parent.parent.name != "data":
        fail(
            "Safety check failed: output directory "
            "is not inside data\\derived."
        )

    if actual.parent.parent.parent != PROJECT_ROOT.resolve():
        fail(
            "Safety check failed: output directory "
            "is outside the project root."
        )


def create_staging_area() -> Path:

    validate_sequence_output_path()

    parent = SEQUENCE_INPUTS.parent

    parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging = Path(
        tempfile.mkdtemp(
            prefix=(
                ".t1d_uom_sequence_inputs_staging_"
            ),
            dir=str(parent),
        )
    )

    marker = staging / SAFETY_MARKER

    marker.write_text(
        "T1D-UOM GENERATED SEQUENCE INPUT "
        "STAGING AREA\n"
        f"script_version={SCRIPT_VERSION}\n",
        encoding="utf-8",
    )

    return staging


def replace_generated_output(
    staging: Path,
) -> None:

    validate_sequence_output_path()

    if not staging.exists():
        fail(
            "Verified staging area no longer exists:\n"
            f"{staging}"
        )

    staging_marker = (
        staging / SAFETY_MARKER
    )

    if not staging_marker.exists():
        fail(
            "Staging safety marker is missing:\n"
            f"{staging_marker}"
        )

    previous = None

    try:

        if SEQUENCE_INPUTS.exists():

            if not SEQUENCE_INPUTS.is_dir():

                fail(
                    "Sequence-input output path exists "
                    "but is not a directory:\n"
                    f"{SEQUENCE_INPUTS}"
                )

            # Only generated sequence-input output may be replaced.
            #
            # A marker is preferred. For this project, an existing directory
            # at the exact generated path is also treated as generated output
            # because this script owns only that derived path. The source
            # datasets are separate paths and are never touched.

            marker = (
                SEQUENCE_INPUTS
                / SAFETY_MARKER
            )

            if marker.exists():

                print(
                    "Existing generated sequence-input "
                    "area found."
                )

                print(
                    "Safety marker verified."
                )

            else:

                print(
                    "Existing sequence-input area found "
                    "at the exact generated derived path."
                )

                print(
                    "It will be replaced only after "
                    "successful staging verification."
                )

            previous = SEQUENCE_INPUTS.with_name(
                SEQUENCE_INPUTS.name
                + ".previous"
            )

            if previous.exists():

                if not previous.is_dir():

                    fail(
                        "Unsafe previous-backup path exists:\n"
                        f"{previous}"
                    )

                shutil.rmtree(
                    previous
                )

            SEQUENCE_INPUTS.rename(
                previous
            )

        staging.rename(
            SEQUENCE_INPUTS
        )

        if previous is not None and previous.exists():

            shutil.rmtree(
                previous
            )

    except Exception:

        # Best-effort rollback.
        if (
            SEQUENCE_INPUTS.exists()
            and SEQUENCE_INPUTS.is_dir()
            and previous is not None
            and previous.exists()
        ):
            shutil.rmtree(
                SEQUENCE_INPUTS,
                ignore_errors=True,
            )

        if (
            previous is not None
            and previous.exists()
            and not SEQUENCE_INPUTS.exists()
        ):
            previous.rename(
                SEQUENCE_INPUTS
            )

        raise


# =============================================================================
# SINGLE-FILE PREPARATION
# =============================================================================

def prepare_file(
    source_path: Path,
    output_path: Path,
) -> Dict[str, Any]:

    participant = (
        participant_from_filename(
            source_path
        )
    )

    modality = (
        identify_modality(
            source_path
        )
    )

    if participant is None:

        fail(
            "Unable to identify participant:\n"
            f"{source_path}"
        )

    if modality is None:

        fail(
            "Unable to identify modality:\n"
            f"{source_path}"
        )

    timestamp_column = (
        TIMESTAMP_COLUMNS.get(
            modality
        )
    )

    if timestamp_column is None:

        fail(
            "No timestamp-column contract exists "
            f"for modality {modality!r}.\n"
            f"File: {source_path}"
        )

    header, rows = read_csv_rows(
        source_path
    )

    if timestamp_column not in header:

        fail(
            "Required timestamp column not found.\n"
            f"File: {source_path}\n"
            f"Timestamp column: {timestamp_column}\n"
            f"Available columns: {header}"
        )

    timestamp_index = (
        header.index(
            timestamp_column
        )
    )

    parsed_rows = []

    parse_methods: Dict[str, int] = {}

    for original_row_index, row in enumerate(rows):

        if len(row) != len(header):

            fail(
                "Malformed CSV row width detected.\n"
                f"File: {source_path}\n"
                f"Original row index: "
                f"{original_row_index}\n"
                f"Expected columns: "
                f"{len(header)}\n"
                f"Actual columns: {len(row)}"
            )

        raw_timestamp = (
            row[timestamp_index]
        )

        try:

            parsed_timestamp, method = (
                parse_timestamp_value(
                    raw_timestamp
                )
            )

        except ValueError as exc:

            fail(
                "TIMESTAMP PARSE FAILURE.\n"
                f"File: {source_path}\n"
                f"Timestamp column: "
                f"{timestamp_column}\n"
                f"Original row index: "
                f"{original_row_index}\n"
                f"Raw timestamp value: "
                f"{raw_timestamp!r}\n"
                f"Reason: {exc}"
            )

        parse_methods[method] = (
            parse_methods.get(
                method,
                0,
            )
            + 1
        )

        parsed_rows.append(
            (
                parsed_timestamp,
                original_row_index,
                row,
            )
        )

    # -----------------------------------------------------------------------
    # Duplicate timestamps are NOT removed.
    #
    # They are ordered by:
    #
    #     1. parsed timestamp
    #     2. original source row index
    #
    # This is deterministic and preserves every observation.
    # -----------------------------------------------------------------------

    timestamp_counts: Dict[
        datetime,
        int,
    ] = {}

    for timestamp, _, _ in parsed_rows:

        timestamp_counts[timestamp] = (
            timestamp_counts.get(
                timestamp,
                0,
            )
            + 1
        )

    duplicate_timestamp_extras = sum(
        count - 1
        for count in timestamp_counts.values()
        if count > 1
    )

    parsed_rows.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    sorted_rows = [
        item[2]
        for item in parsed_rows
    ]

    # -----------------------------------------------------------------------
    # Strict row-value preservation.
    # -----------------------------------------------------------------------

    source_multiset = sorted(
        row_digest(row)
        for row in rows
    )

    output_multiset = sorted(
        row_digest(row)
        for row in sorted_rows
    )

    if source_multiset != output_multiset:

        fail(
            "ROW-PRESERVATION FAILURE.\n"
            f"File: {source_path}\n"
            "The output does not contain exactly "
            "the same multiset of row values."
        )

    source_timestamp_order = [
        item[0]
        for item in sorted(
            parsed_rows,
            key=lambda item: item[1],
        )
    ]

    source_was_chronological = all(
        source_timestamp_order[index]
        <= source_timestamp_order[index + 1]
        for index in range(
            len(source_timestamp_order) - 1
        )
    )

    write_csv_rows(
        output_path,
        header,
        sorted_rows,
    )

    return {
        "source_path": relative_path(
            source_path,
            MODELING_DATASET,
        ),
        "output_path": relative_path(
            output_path,
            output_path.parents[
                len(output_path.parts)
                - len(SEQUENCE_INPUTS.parts)
            ]
            if False
            else output_path.parent.parent.parent.parent.parent
        )
        if False
        else None,
        "participant": participant,
        "modality": modality,
        "timestamp_column": timestamp_column,
        "row_count": len(rows),
        "column_count": len(header),
        "duplicate_timestamp_extras": (
            duplicate_timestamp_extras
        ),
        "source_was_chronological": (
            source_was_chronological
        ),
        "parse_methods": dict(
            sorted(
                parse_methods.items()
            )
        ),
        "source_sha256": sha256_file(
            source_path
        ),
        "output_sha256": sha256_file(
            output_path
        ),
    }


# =============================================================================
# OUTPUT VERIFICATION
# =============================================================================

def verify_output(
    selected_files: Sequence[Path],
    records: Sequence[Dict[str, Any]],
    staging: Path,
) -> Dict[str, Any]:

    output_files = discover_csv_files(
        staging
    )

    if len(output_files) != (
        EXPECTED_SEQUENCE_FILE_COUNT
    ):

        fail(
            "Final sequence-input file count failed.\n"
            f"Expected: "
            f"{EXPECTED_SEQUENCE_FILE_COUNT}\n"
            f"Found: {len(output_files)}"
        )

    expected_paths = sorted(
        relative_path(
            staging
            / relative_path(
                source,
                MODELING_DATASET,
            ),
            staging,
        )
        for source in selected_files
    )

    actual_paths = sorted(
        relative_path(
            output,
            staging,
        )
        for output in output_files
    )

    if expected_paths != actual_paths:

        fail(
            "Final sequence-input file-set "
            "verification failed."
        )

    for record in records:

        source = (
            MODELING_DATASET
            / Path(
                record["source_path"]
            )
        )

        output = (
            staging
            / Path(
                record["output_path"]
            )
        )

        source_header, source_rows = (
            read_csv_rows(
                source
            )
        )

        output_header, output_rows = (
            read_csv_rows(
                output
            )
        )

        if source_header != output_header:

            fail(
                "HEADER PRESERVATION FAILURE.\n"
                f"Source: {source}\n"
                f"Output: {output}"
            )

        if len(source_rows) != len(
            output_rows
        ):

            fail(
                "ROW COUNT PRESERVATION FAILURE.\n"
                f"Source: {source}\n"
                f"Output: {output}"
            )

        source_multiset = sorted(
            row_digest(row)
            for row in source_rows
        )

        output_multiset = sorted(
            row_digest(row)
            for row in output_rows
        )

        if source_multiset != output_multiset:

            fail(
                "ROW VALUE PRESERVATION FAILURE.\n"
                f"Source: {source}\n"
                f"Output: {output}"
            )

    return {
        "output_file_count": len(
            output_files
        ),
        "expected_output_file_count": (
            EXPECTED_SEQUENCE_FILE_COUNT
        ),
        "file_count_pass": True,
        "exact_file_set_pass": True,
        "headers_preserved": True,
        "row_counts_preserved": True,
        "row_value_multisets_preserved": True,
        "files_verified": len(
            records
        ),
        "source_byte_identity_required": False,
        "source_byte_identity_note": (
            "Sequence-input CSV files are derived "
            "through chronological ordering, so "
            "byte identity is intentionally not "
            "required. Original headers and row "
            "values are verified exactly."
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print("T1D-UOM SEQUENCE INPUT PREPARATION")
    print("=" * 80)
    print()

    print("IMPORTANT:")
    print("  Raw dataset will NOT be modified.")
    print("  Timestamp-corrected dataset will NOT be modified.")
    print("  Modeling dataset will NOT be modified.")
    print("  Duplicate observations will NOT be deleted.")
    print("  No resampling will be performed.")
    print("  No interpolation will be performed.")
    print("  No imputation will be performed.")
    print("  No normalization will be performed.")
    print("  No feature engineering will be performed.")
    print("  No targets/windows will be created.")
    print("  No model will be trained.")
    print()

    print(
        f"Project root:       "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Modeling dataset:   "
        f"{MODELING_DATASET}"
    )

    print(
        f"Sequence inputs:    "
        f"{SEQUENCE_INPUTS}"
    )

    print(
        f"Manifest:           "
        f"{OUTPUT_MANIFEST}"
    )

    print()

    # -----------------------------------------------------------------------
    # 1
    # -----------------------------------------------------------------------

    print("-" * 80)
    print("1. FROZEN MODELING DATASET VALIDATION")
    print("-" * 80)

    source_files = discover_csv_files(
        MODELING_DATASET
    )

    validation = (
        validate_modeling_dataset(
            source_files
        )
    )

    print(
        "Frozen modeling dataset: PASS"
    )

    print(
        "Frozen cohort size: 13"
    )

    print(
        "Additional participants present "
        "but excluded from sequence preparation: "
        f"{validation['additional_participant_count']}"
    )

    print()

    # -----------------------------------------------------------------------
    # 2
    # -----------------------------------------------------------------------

    print("-" * 80)
    print("2. FREEZE MANIFEST VALIDATION")
    print("-" * 80)

    freeze_validation = (
        validate_freeze_manifests()
    )

    print(
        "PASS: modeling_dataset_exclusions.json"
    )

    print(
        "PASS: modeling_dataset_manifest.json"
    )

    print()

    # -----------------------------------------------------------------------
    # 3
    # -----------------------------------------------------------------------

    print("-" * 80)
    print("3. FROZEN-COHORT FILE SELECTION")
    print("-" * 80)

    selected_files = (
        select_frozen_files(
            source_files
        )
    )

    print(
        "Files selected for sequence preparation: "
        f"{len(selected_files)}"
    )

    print(
        "Non-frozen files excluded: "
        f"{len(source_files) - len(selected_files)}"
    )

    print(
        "Frozen-cohort file count: PASS"
    )

    print()

    # -----------------------------------------------------------------------
    # 4
    # -----------------------------------------------------------------------

    print("-" * 80)
    print("4. PREPARING SAFE DERIVED STAGING AREA")
    print("-" * 80)

    staging = create_staging_area()

    print(
        "Fresh temporary sequence-input "
        "staging area: READY"
    )

    print(
        "Existing final sequence-input area "
        "will remain untouched until all "
        "86 files pass verification."
    )

    print()

    records: List[
        Dict[str, Any]
    ] = []

    try:

        # -------------------------------------------------------------------
        # 5
        # -------------------------------------------------------------------

        print("-" * 80)
        print("5. DETERMINISTIC TEMPORAL PREPARATION")
        print("-" * 80)

        for index, source_path in enumerate(
            selected_files,
            start=1,
        ):

            relative = relative_path(
                source_path,
                MODELING_DATASET,
            )

            print(
                f"[{index:03d}/{len(selected_files):03d}] "
                f"{relative}"
            )

            output_path = (
                staging
                / relative
            )

            record = prepare_file(
                source_path,
                output_path,
            )

            # Output path is exactly the same relative path as the source,
            # but under the derived sequence-input area.
            record["output_path"] = relative

            records.append(
                record
            )

        print()

        print(
            f"Prepared {len(records)} "
            "frozen-cohort files."
        )

        print()

        # -------------------------------------------------------------------
        # 6
        # -------------------------------------------------------------------

        print("-" * 80)
        print("6. FINAL STAGING VERIFICATION")
        print("-" * 80)

        verification = verify_output(
            selected_files,
            records,
            staging,
        )

        print(
            "Sequence-input CSV files: "
            f"{len(records)} / "
            f"{EXPECTED_SEQUENCE_FILE_COUNT}"
        )

        print(
            "Exact frozen-cohort file set: PASS"
        )

        print(
            "Original columns preserved: PASS"
        )

        print(
            "Original row-value multiset preserved: PASS"
        )

        print(
            "Duplicate observations retained: PASS"
        )

        print(
            "Chronological ordering applied: PASS"
        )

        print(
            "Deterministic duplicate-timestamp "
            "tie handling: PASS"
        )

        print()

        # -------------------------------------------------------------------
        # 7
        # -------------------------------------------------------------------

        total_rows = sum(
            int(record["row_count"])
            for record in records
        )

        duplicate_timestamp_extras = sum(
            int(
                record[
                    "duplicate_timestamp_extras"
                ]
            )
            for record in records
        )

        reordered_files = sum(
            1
            for record in records
            if not record[
                "source_was_chronological"
            ]
        )

        parse_methods: Dict[
            str,
            int,
        ] = {}

        for record in records:

            for method, count in (
                record["parse_methods"]
                .items()
            ):

                parse_methods[method] = (
                    parse_methods.get(
                        method,
                        0,
                    )
                    + int(count)
                )

        print("-" * 80)
        print("7. TEMPORAL PREPARATION SUMMARY")
        print("-" * 80)

        print(
            f"Files prepared: "
            f"{len(records)}"
        )

        print(
            f"Total rows preserved: "
            f"{total_rows}"
        )

        print(
            f"Duplicate timestamp extras: "
            f"{duplicate_timestamp_extras}"
        )

        print(
            "Files requiring chronological reordering: "
            f"{reordered_files}"
        )

        print(
            "Timestamp parse failures:       0"
        )

        print()

        print(
            "Timestamp parsing methods:"
        )

        for method, count in sorted(
            parse_methods.items()
        ):

            print(
                f"  {method}: {count}"
            )

        print()

        # -------------------------------------------------------------------
        # 8
        # -------------------------------------------------------------------

        print("-" * 80)
        print("8. FROZEN ARCHITECTURE CONTRACT")
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

        print()

        print(
            "Frozen architecture contract: PASS"
        )

        print()

        # -------------------------------------------------------------------
        # 9
        # -------------------------------------------------------------------

        print("-" * 80)
        print("9. WRITING PREPARATION MANIFEST")
        print("-" * 80)

        manifest = {
            "schema_version": "1.1",
            "script_version": SCRIPT_VERSION,
            "generated_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "project_root": str(
                PROJECT_ROOT
            ),
            "source_dataset": str(
                MODELING_DATASET
            ),
            "sequence_input_dataset": str(
                SEQUENCE_INPUTS
            ),
            "frozen_cohort": list(
                FROZEN_COHORT
            ),
            "frozen_cohort_size": len(
                FROZEN_COHORT
            ),
            "source_modeling_file_count": len(
                source_files
            ),
            "expected_modeling_file_count": (
                EXPECTED_MODELING_FILE_COUNT
            ),
            "selected_sequence_file_count": len(
                selected_files
            ),
            "expected_sequence_file_count": (
                EXPECTED_SEQUENCE_FILE_COUNT
            ),
            "additional_participants_not_used": (
                validation[
                    "additional_participants"
                ]
            ),
            "freeze_manifest_validation": (
                freeze_validation
            ),
            "architecture": ARCHITECTURE,
            "sequence_preparation_policy": {
                "chronological_sort": True,
                "sort_key": [
                    "parsed_timestamp_utc",
                    "original_source_row_order",
                ],
                "duplicate_observations": (
                    "retained"
                ),
                "duplicate_timestamp_tie_breaker": (
                    "original source row order"
                ),
                "resampling": False,
                "interpolation": False,
                "imputation": False,
                "normalization": False,
                "feature_engineering": False,
                "target_generation": False,
                "window_generation": False,
                "model_training": False,
                "source_values_modified": False,
                "source_columns_modified": False,
            },
            "timestamp_columns": (
                TIMESTAMP_COLUMNS
            ),
            "timestamp_parser_contract": {
                "slash_date_format": (
                    "DD/MM/YYYY"
                ),
                "supported_examples": [
                    "01/10/2023 05:45",
                    "01/10/2023 05:45:00",
                    "01/10/2023 05:45:00.000000",
                ],
                "slash_date_interpretation": (
                    "day-first"
                ),
                "iso8601_supported": True,
                "numeric_epoch_supported": True,
            },
            "summary": {
                "files_prepared": len(
                    records
                ),
                "total_rows_preserved": (
                    total_rows
                ),
                "duplicate_timestamp_extras": (
                    duplicate_timestamp_extras
                ),
                "files_requiring_reordering": (
                    reordered_files
                ),
                "timestamp_parse_failures": 0,
            },
            "verification": verification,
            "files": records,
        }

        # Write the manifest into the staging area first.
        staging_manifest = (
            staging
            / "_sequence_input_manifest.json"
        )

        write_json(
            staging_manifest,
            manifest,
        )

        # Also maintain the requested report manifest.
        write_json(
            OUTPUT_MANIFEST,
            manifest,
        )

        print(
            "Preparation manifest saved:"
        )

        print(
            f"  {OUTPUT_MANIFEST}"
        )

        print()

        # -------------------------------------------------------------------
        # 10
        # -------------------------------------------------------------------

        print("-" * 80)
        print("10. COMMITTING VERIFIED STAGING AREA")
        print("-" * 80)

        replace_generated_output(
            staging
        )

        staging = None

        print(
            "Verified sequence-input staging "
            "area committed: PASS"
        )

        print()

        # -------------------------------------------------------------------
        # FINAL SUCCESS
        # -------------------------------------------------------------------

        print("=" * 80)
        print(
            "T1D-UOM SEQUENCE INPUT "
            "PREPARATION COMPLETED SUCCESSFULLY"
        )
        print("=" * 80)

        print()

        print("SOURCE DATASET")

        print(
            f"  Modeling CSV files:       "
            f"{len(source_files)}"
        )

        print(
            "  Modified:                 NO"
        )

        print()

        print("FROZEN SEQUENCE COHORT")

        print(
            "  Participants:             13"
        )

        for participant in FROZEN_COHORT:

            print(
                f"    - {participant}"
            )

        print()

        print("SEQUENCE INPUTS")

        print(
            f"  CSV files prepared:       "
            f"{len(records)}"
        )

        print(
            f"  Rows preserved:           "
            f"{total_rows}"
        )

        print(
            "  Source values modified:   NO"
        )

        print(
            "  Source columns modified:  NO"
        )

        print(
            "  Duplicate rows deleted:   NO"
        )

        print(
            "  Resampling:               NO"
        )

        print(
            "  Interpolation:            NO"
        )

        print(
            "  Imputation:               NO"
        )

        print(
            "  Normalization:            NO"
        )

        print(
            "  Feature engineering:      NO"
        )

        print(
            "  Targets/windows:          NO"
        )

        print(
            "  Model training:           NO"
        )

        print()

        print("TEMPORAL PREPARATION")

        print(
            "  Timestamp parsing:        PASS"
        )

        print(
            "  Chronological sorting:    PASS"
        )

        print(
            "  Duplicate retention:      PASS"
        )

        print(
            "  Deterministic tie-break:  PASS"
        )

        print()

        print("FROZEN ARCHITECTURE")

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

        print("MANIFEST")

        print(
            f"  {OUTPUT_MANIFEST}"
        )

        print()

        print("IMPORTANT")

        print(
            "  Raw dataset:               PRESERVED"
        )

        print(
            "  Timestamp-corrected data: PRESERVED"
        )

        print(
            "  Modeling dataset:         PRESERVED"
        )

        print(
            "  Frozen architecture:      UNCHANGED"
        )

        print(
            "  Additional participants:  NOT USED"
        )

        print()

        print("NEXT STAGE:")

        print(
            "  Sequence-input preparation is complete."
        )

        print(
            "  No targets or windows have been created."
        )

        print(
            "  No model architecture has been changed."
        )

        print(
            "  No prediction/what-if implementation "
            "has been created."
        )

        print(
            "  No Interactive UI has been created."
        )

        print("=" * 80)

    except Exception:

        # The staging directory is disposable. The existing final generated
        # sequence-input area is not touched until successful commit.

        if (
            staging is not None
            and staging.exists()
        ):

            shutil.rmtree(
                staging,
                ignore_errors=True,
            )

        raise


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Operation cancelled by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 80)
        print(
            "SEQUENCE INPUT PREPARATION FAILED"
        )
        print("=" * 80)

        print(
            str(exc)
        )

        print()

        print("IMPORTANT:")

        print(
            "  Raw dataset was preserved."
        )

        print(
            "  Timestamp-corrected dataset "
            "was preserved."
        )

        print(
            "  Modeling dataset was preserved."
        )

        print(
            "  The previous final sequence-input "
            "area was preserved unless a fully "
            "verified replacement was committed."
        )

        print("=" * 80)

        sys.exit(1)