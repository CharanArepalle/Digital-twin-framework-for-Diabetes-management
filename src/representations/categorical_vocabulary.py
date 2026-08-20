"""
T1D-UOM CATEGORICAL VOCABULARY REPRESENTATION
=============================================

Purpose
-------
Provide the minimal categorical representation layer required before
the five frozen modality-specific GRU branches.

Frozen architecture
-------------------
    Glucose   -> GRU -> zG
    Insulin   -> GRU -> zI
    Nutrition -> GRU -> zN
    Activity  -> GRU -> zA
    Sleep     -> GRU -> zS

Important
---------
This module does NOT modify any CSV file.

Categorical values are represented in memory only.

Policy
------
1. Vocabulary is fitted ONLY from the supplied training partition.
2. Raw strings are preserved exactly.
3. No lower-casing.
4. No whitespace trimming.
5. No category collapsing.
6. Empty / missing categorical observations -> MISSING.
7. Values unseen during training -> UNK.
8. PAD is reserved for sequence padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor


PAD_ID = 0
UNK_ID = 1
MISSING_ID = 2


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MISSING_TOKEN = "<MISSING>"


@dataclass(frozen=True)
class CategoricalVocabulary:
    """
    Immutable categorical vocabulary.

    IDs:
        0 -> PAD
        1 -> UNK
        2 -> MISSING
        3+ -> observed training values
    """

    field_name: str
    token_to_id: Mapping[str, int]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def contains(self, value: str) -> bool:
        return value in self.token_to_id

    def id_for(self, value: object) -> int:
        """
        Convert one raw value to its runtime categorical ID.

        IMPORTANT:
        The original value is not modified.

        None / empty string -> MISSING
        unseen value        -> UNK
        known value         -> learned ID
        """

        if value is None:
            return MISSING_ID

        text = str(value)

        if text == "":
            return MISSING_ID

        return self.token_to_id.get(text, UNK_ID)


def fit_categorical_vocabulary(
    values: Iterable[object],
    *,
    field_name: str,
) -> CategoricalVocabulary:
    """
    Fit an exact-string vocabulary from TRAINING-PARTITION values only.

    This function does not normalize source values.

    The caller is responsible for ensuring that `values` originates only
    from the training partition.
    """

    if not field_name:
        raise ValueError("field_name must not be empty.")

    observed: set[str] = set()

    for value in values:
        if value is None:
            continue

        text = str(value)

        if text == "":
            continue

        observed.add(text)

    token_to_id: dict[str, int] = {
        PAD_TOKEN: PAD_ID,
        UNK_TOKEN: UNK_ID,
        MISSING_TOKEN: MISSING_ID,
    }

    # Deterministic ordering.
    for token in sorted(observed):
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)

    return CategoricalVocabulary(
        field_name=field_name,
        token_to_id=token_to_id,
    )


def encode_categorical_values(
    values: Sequence[object],
    vocabulary: CategoricalVocabulary,
) -> Tensor:
    """
    Encode raw categorical values into a torch.long tensor.

    No source data is modified.
    """

    ids = [
        vocabulary.id_for(value)
        for value in values
    ]

    return torch.tensor(
        ids,
        dtype=torch.long,
    )


def validate_vocabulary(vocabulary: CategoricalVocabulary) -> None:
    """
    Validate the structural integrity of a vocabulary.
    """

    required = {
        PAD_TOKEN: PAD_ID,
        UNK_TOKEN: UNK_ID,
        MISSING_TOKEN: MISSING_ID,
    }

    for token, expected_id in required.items():
        actual_id = vocabulary.token_to_id.get(token)

        if actual_id != expected_id:
            raise ValueError(
                f"Invalid vocabulary for '{vocabulary.field_name}': "
                f"{token!r} must have ID {expected_id}, "
                f"but has {actual_id!r}."
            )

    ids = list(vocabulary.token_to_id.values())

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"Vocabulary '{vocabulary.field_name}' contains duplicate IDs."
        )

    if sorted(ids) != list(range(len(ids))):
        raise ValueError(
            f"Vocabulary '{vocabulary.field_name}' IDs must be contiguous "
            f"from 0 to size-1."
        )


def validate_encoded_tensor(
    encoded: Tensor,
    vocabulary: CategoricalVocabulary,
) -> None:
    """
    Validate a categorical tensor before it is passed to an embedding.
    """

    if encoded.dtype != torch.long:
        raise TypeError(
            f"Categorical tensor for '{vocabulary.field_name}' must have "
            f"dtype torch.long; got {encoded.dtype}."
        )

    if encoded.numel() == 0:
        return

    minimum = int(encoded.min().item())
    maximum = int(encoded.max().item())

    if minimum < 0:
        raise ValueError(
            f"Categorical tensor for '{vocabulary.field_name}' contains "
            f"negative ID {minimum}."
        )

    if maximum >= vocabulary.size:
        raise ValueError(
            f"Categorical tensor for '{vocabulary.field_name}' contains "
            f"ID {maximum}, but vocabulary size is {vocabulary.size}."
        )