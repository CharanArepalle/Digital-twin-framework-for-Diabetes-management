from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


# ============================================================================
# T1D-UOM TIMESTAMP ANOMALY INVESTIGATION
# ============================================================================
#
# READ-ONLY.
#
# This script investigates the two suspicious nutrition timestamps found by
# the dataset audit.
#
# It DOES NOT modify any raw CSV file.
# ============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "t1d_uom_v1.0.3"
)


TARGETS = [
    {
        "filename": "UoMNutrition2320.csv",
        "participant": "UoM2320",
        "column": "meal_ts",
        "row_index": 19,
        "suspected_year": 2023,
    },
    {
        "filename": "UoMNutrition2404.csv",
        "participant": "UoM2404",
        "column": "meal_ts",
        "row_index": 166,
        "suspected_year": 2024,
    },
]


# ============================================================================
# TIMESTAMP PARSER
# ============================================================================

def parse_timestamp(value):
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()

    if not text:
        return pd.NaT

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
    ]

    for fmt in formats:
        parsed = pd.to_datetime(
            text,
            format=fmt,
            errors="coerce",
        )

        if not pd.isna(parsed):
            return parsed

    return pd.NaT


# ============================================================================
# PARTICIPANT EXTRACTION
# ============================================================================

def extract_participant_id(filename: str):
    match = re.search(
        r"UoM(?:[A-Za-z]+)?(2[34]\d{2})",
        filename,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return f"UoM{match.group(1)}"


# ============================================================================
# MODALITY IDENTIFICATION
# ============================================================================

def identify_modality(path: Path):

    text = str(path).lower()
    filename = path.name.lower()

    if "activity data" in text:
        return "activity"

    if "glucose data" in text:
        return "glucose"

    if "basal data" in text:
        return "basal_insulin"

    if "bolus data" in text:
        return "bolus_insulin"

    if "nutrition data" in text:
        return "nutrition"

    if "sleeptime" in filename:
        return "sleep_summary"

    if "sleep data" in text:
        return "sleep_timeseries"

    return "unknown"


# ============================================================================
# FILE SEARCH
# ============================================================================

def find_participant_files(
    participant: str,
):
    matches = []

    for path in DATASET_ROOT.rglob("*.csv"):

        extracted = extract_participant_id(
            path.name
        )

        if extracted == participant:
            matches.append(path)

    return sorted(
        matches,
        key=lambda p: str(p).lower(),
    )


# ============================================================================
# INVESTIGATE TARGET
# ============================================================================

def investigate_target(target):

    filename = target["filename"]
    participant = target["participant"]
    column = target["column"]
    row_index = target["row_index"]
    suspected_year = target["suspected_year"]

    path = DATASET_ROOT / "Nutrition Data" / filename

    print()
    print("=" * 80)
    print(f"FILE: {filename}")
    print(f"PARTICIPANT: {participant}")
    print(f"TARGET ROW INDEX: {row_index}")
    print("=" * 80)

    if not path.exists():
        print(f"ERROR: file does not exist: {path}")
        return

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df["_parsed_ts"] = df[column].map(
        parse_timestamp
    )

    if row_index not in df.index:
        print(
            f"ERROR: row index {row_index} "
            f"is not present."
        )
        return

    target_row = df.loc[row_index]

    raw_value = target_row[column]

    parsed_value = target_row["_parsed_ts"]

    print()
    print("TARGET VALUE")
    print("-" * 80)

    print(
        f"Original timestamp : {raw_value}"
    )

    print(
        f"Parsed timestamp   : {parsed_value}"
    )

    print(
        f"Suspected year     : {suspected_year}"
    )

    # ------------------------------------------------------------------------
    # Neighbouring rows
    # ------------------------------------------------------------------------

    print()
    print("NEIGHBOURING ROWS")
    print("-" * 80)

    start = max(
        0,
        row_index - 5,
    )

    end = min(
        len(df),
        row_index + 6,
    )

    columns_to_show = [
        column,
        "_parsed_ts",
    ]

    for extra in [
        "meal_type",
        "meal_tag",
        "carbs_g",
        "prot_g",
        "fat_g",
        "fibre_g",
    ]:
        if extra in df.columns:
            columns_to_show.append(
                extra
            )

    print(
        df.loc[
            start:end - 1,
            columns_to_show,
        ].to_string()
    )

    # ------------------------------------------------------------------------
    # Previous and next valid timestamps
    # ------------------------------------------------------------------------

    valid_ts = (
        df["_parsed_ts"]
        .dropna()
        .sort_values()
    )

    previous = valid_ts[
        valid_ts.index < row_index
    ]

    following = valid_ts[
        valid_ts.index > row_index
    ]

    print()
    print("TEMPORAL NEIGHBOURS")
    print("-" * 80)

    if not previous.empty:

        prev_index = previous.index[-1]

        prev_ts = previous.iloc[-1]

        print(
            f"Previous row timestamp: "
            f"index={prev_index}, "
            f"{prev_ts}"
        )

    else:

        print(
            "No previous valid timestamp."
        )

    if not following.empty:

        next_index = following.index[0]

        next_ts = following.iloc[0]

        print(
            f"Next row timestamp: "
            f"index={next_index}, "
            f"{next_ts}"
        )

    else:

        print(
            "No next valid timestamp."
        )

    # ------------------------------------------------------------------------
    # Candidate corrected timestamp
    # ------------------------------------------------------------------------

    candidate = parsed_value.replace(
        year=suspected_year
    )

    print()
    print("CANDIDATE YEAR CORRECTION")
    print("-" * 80)

    print(
        f"Original parsed timestamp : "
        f"{parsed_value}"
    )

    print(
        f"Candidate corrected value  : "
        f"{candidate}"
    )

    collision = (
        df["_parsed_ts"] == candidate
    ).sum()

    print(
        f"Existing rows with candidate "
        f"timestamp: {int(collision)}"
    )

    # ------------------------------------------------------------------------
    # Difference to neighbouring rows
    # ------------------------------------------------------------------------

    if not previous.empty:

        previous_ts = previous.iloc[-1]

        print(
            f"Difference from previous "
            f"timestamp: "
            f"{parsed_value - previous_ts}"
        )

        print(
            f"Difference from previous "
            f"using candidate year: "
            f"{candidate - previous_ts}"
        )

    if not following.empty:

        following_ts = following.iloc[0]

        print(
            f"Difference to next timestamp: "
            f"{following_ts - parsed_value}"
        )

        print(
            f"Difference to next timestamp "
            f"using candidate year: "
            f"{following_ts - candidate}"
        )


# ============================================================================
# PARTICIPANT-WIDE DATE RANGES
# ============================================================================

def print_participant_ranges(
    participant: str,
):

    print()
    print("=" * 80)
    print(
        f"PARTICIPANT-WIDE TIMESTAMP RANGES: "
        f"{participant}"
    )
    print("=" * 80)

    files = find_participant_files(
        participant
    )

    if not files:

        print(
            "No participant files found."
        )

        return

    for path in files:

        modality = identify_modality(
            path
        )

        try:

            df = pd.read_csv(
                path,
                low_memory=False,
            )

        except Exception as exc:

            print()
            print(
                f"{modality:<18} "
                f"{path.name}: READ ERROR: "
                f"{exc}"
            )

            continue

        timestamp_columns = []

        for column in df.columns:

            lower = str(
                column
            ).lower()

            if (
                lower.endswith("_ts")
                or lower in {
                    "calendar_date",
                    "start_date_ts",
                }
            ):
                timestamp_columns.append(
                    column
                )

        print()

        print(
            f"{modality:<18} "
            f"{path.name}"
        )

        for column in timestamp_columns:

            parsed = df[column].map(
                parse_timestamp
            )

            valid = parsed.dropna()

            if valid.empty:

                print(
                    f"  {column}: "
                    f"no valid timestamps"
                )

                continue

            print(
                f"  {column}: "
                f"{valid.min()} "
                f"-> "
                f"{valid.max()} "
                f"({len(valid)} valid)"
            )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print(
        "T1D-UOM TIMESTAMP ANOMALY INVESTIGATION"
    )
    print("=" * 80)

    print()
    print(
        f"Dataset root: {DATASET_ROOT}"
    )

    print()
    print(
        "IMPORTANT: READ-ONLY. "
        "NO RAW FILES WILL BE MODIFIED."
    )

    for target in TARGETS:

        investigate_target(
            target
        )

        print_participant_ranges(
            target["participant"]
        )

    print()
    print("=" * 80)
    print(
        "INVESTIGATION COMPLETE"
    )
    print("=" * 80)

    print()
    print(
        "No raw dataset files were modified."
    )


if __name__ == "__main__":
    main()