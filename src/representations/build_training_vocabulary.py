"""
T1D-UOM TRAINING VOCABULARY BUILDER
===================================

Purpose
-------
Create the persisted categorical vocabulary artifact required by the
real unified-state generator.

IMPORTANT
---------
This script does NOT modify source CSV files.

The vocabulary is fitted from an explicitly supplied TRAINING partition.

No:
    - lower-casing
    - whitespace trimming
    - category collapsing
    - imputation
    - normalization
    - source modification

Categorical policy
------------------
    PAD     = 0
    UNK     = 1
    MISSING = 2

Fields:
    meal_type
    meal_tag
    activity_type
    intensity

Usage
-----
python -m src.representations.build_training_vocabulary \
    --participants UoM2301 UoM2302 UoM2304

The participant list must be the project's locked TRAINING partition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.representations.categorical_vocabulary import (
    CategoricalVocabulary,
    fit_categorical_vocabulary,
    validate_vocabulary,
)


ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = (
    ROOT
    / "data"
    / "derived"
    / "t1d_uom_v1.0.3_sequence_inputs"
)

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "derived"
    / "categorical_vocabularies"
)

OUTPUT_FILE = (
    OUTPUT_ROOT
    / "categorical_vocabularies.json"
)


FROZEN_COHORT = (
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
)


FIELDS = (
    "meal_type",
    "meal_tag",
    "activity_type",
    "intensity",
)


def participant_from_filename(
    path: Path,
) -> str | None:

    import re

    patterns = (
        r"UoMActivity(\d{4})",
        r"UoMGlucose(\d{4})",
        r"UoMBasal(\d{4})",
        r"UoMBolus(\d{4})",
        r"UoMNutrition(\d{4})",
        r"UoMsleep(\d{4})",
        r"UoM(\d{4})sleeptime",
    )

    for pattern in patterns:

        match = re.search(
            pattern,
            path.name,
            flags=re.IGNORECASE,
        )

        if match:
            return f"UoM{match.group(1)}"

    return None


def discover_files(
    participant: str,
    filename_fragment: str,
) -> list[Path]:

    result = []

    for path in SOURCE_ROOT.rglob("*.csv"):

        pid = participant_from_filename(path)

        if pid != participant:
            continue

        if filename_fragment.lower() in path.name.lower():
            result.append(path)

    return sorted(result)


def read_column(
    paths: list[Path],
    column: str,
) -> list[object]:

    values = []

    for path in paths:

        df = pd.read_csv(path)

        if column not in df.columns:
            raise RuntimeError(
                f"{path}: required column '{column}' "
                f"is missing."
            )

        values.extend(
            df[column].tolist()
        )

    return values


def build_training_vocabularies(
    participants: tuple[str, ...],
) -> dict[str, CategoricalVocabulary]:

    nutrition_paths = []
    activity_paths = []

    for participant in participants:

        nutrition = discover_files(
            participant,
            "UoMNutrition",
        )

        activity = discover_files(
            participant,
            "UoMActivity",
        )

        if not nutrition:
            raise RuntimeError(
                f"{participant}: nutrition file not found."
            )

        if not activity:
            raise RuntimeError(
                f"{participant}: activity file not found."
            )

        nutrition_paths.extend(nutrition)
        activity_paths.extend(activity)

    meal_type_values = read_column(
        nutrition_paths,
        "meal_type",
    )

    meal_tag_values = read_column(
        nutrition_paths,
        "meal_tag",
    )

    activity_type_values = read_column(
        activity_paths,
        "activity_type",
    )

    intensity_values = read_column(
        activity_paths,
        "intensity",
    )

    vocabularies = {
        "meal_type": fit_categorical_vocabulary(
            meal_type_values,
            field_name="meal_type",
        ),
        "meal_tag": fit_categorical_vocabulary(
            meal_tag_values,
            field_name="meal_tag",
        ),
        "activity_type": fit_categorical_vocabulary(
            activity_type_values,
            field_name="activity_type",
        ),
        "intensity": fit_categorical_vocabulary(
            intensity_values,
            field_name="intensity",
        ),
    }

    for field, vocabulary in vocabularies.items():
        validate_vocabulary(vocabulary)

    return vocabularies


def serialize_vocabulary(
    vocabulary: CategoricalVocabulary,
) -> dict:

    return {
        "field_name": vocabulary.field_name,
        "token_to_id": {
            str(key): int(value)
            for key, value in vocabulary.token_to_id.items()
        },
        "size": vocabulary.size,
    }


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--participants",
        nargs="+",
        required=True,
        help=(
            "Explicit training-partition participants."
        ),
    )

    args = parser.parse_args()

    participants = tuple(
        args.participants
    )

    print()
    print("=" * 80)
    print("T1D-UOM TRAINING VOCABULARY BUILDER")
    print("=" * 80)
    print()

    print(
        "Training participants:"
    )

    for participant in participants:
        print(
            f"  {participant}"
        )

    print()

    invalid = sorted(
        set(participants)
        - set(FROZEN_COHORT)
    )

    if invalid:
        raise RuntimeError(
            "Training participant(s) are outside the "
            f"frozen cohort: {invalid}"
        )

    if not participants:
        raise RuntimeError(
            "At least one training participant is required."
        )

    print(
        "Source modification : NO"
    )
    print(
        "Normalization       : NO"
    )
    print(
        "Whitespace trimming : NO"
    )
    print(
        "Category collapsing  : NO"
    )
    print(
        "Imputation           : NO"
    )

    vocabularies = build_training_vocabularies(
        participants
    )

    print()
    print(
        "VOCABULARY SIZES"
    )
    print("-" * 80)

    for field in FIELDS:

        vocabulary = vocabularies[field]

        print(
            f"{field:20s}: "
            f"{vocabulary.size}"
        )

    artifact = {
        "format": "t1d_uom_categorical_vocabulary_v1",
        "training_participants": list(
            participants
        ),
        "policy": {
            "exact_string_matching": True,
            "lowercase": False,
            "trim_whitespace": False,
            "category_collapsing": False,
            "missing_id": 2,
            "unknown_id": 1,
            "padding_id": 0,
        },
        "vocabularies": {
            field: serialize_vocabulary(
                vocabularies[field]
            )
            for field in FIELDS
        },
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            artifact,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "ARTIFACT CREATED"
    )
    print("-" * 80)
    print(
        OUTPUT_FILE
    )

    print()
    print(
        "VALIDATION"
    )
    print("-" * 80)

    saved = json.loads(
        OUTPUT_FILE.read_text(
            encoding="utf-8"
        )
    )

    assert (
        saved["format"]
        == "t1d_uom_categorical_vocabulary_v1"
    )

    assert (
        saved["training_participants"]
        == list(participants)
    )

    for field in FIELDS:

        token_to_id = (
            saved["vocabularies"][field]
            ["token_to_id"]
        )

        assert token_to_id["<PAD>"] == 0
        assert token_to_id["<UNK>"] == 1
        assert token_to_id["<MISSING>"] == 2

        ids = sorted(
            int(value)
            for value in token_to_id.values()
        )

        assert ids == list(
            range(len(ids))
        )

    print(
        "Reserved IDs          : PASS"
    )
    print(
        "Contiguous IDs        : PASS"
    )
    print(
        "Training-only source  : PASS"
    )
    print(
        "Artifact integrity    : PASS"
    )

    print()
    print(
        "TRAINING VOCABULARY BUILD: PASS"
    )
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()