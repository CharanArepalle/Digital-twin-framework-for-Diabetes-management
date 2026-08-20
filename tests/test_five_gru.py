"""
T1D-UOM — Five-GRU architecture tests.
"""

from __future__ import annotations

import torch

from src.models.five_gru import (
    ACTIVITY_DIM,
    GLUCOSE_DIM,
    INSULIN_DIM,
    NUTRITION_DIM,
    SLEEP_DIM,
    FiveGRU,
    validate_five_gru,
    validate_output_contract,
    validate_runtime_dimensions,
)


def test_exactly_five_gru_branches() -> None:
    model = FiveGRU(hidden_dim=32)

    validate_five_gru(model)
    validate_runtime_dimensions(model)


def test_runtime_input_dimensions() -> None:
    model = FiveGRU(hidden_dim=32)

    assert model.glucose_gru.gru.input_size == GLUCOSE_DIM
    assert model.insulin_gru.gru.input_size == INSULIN_DIM
    assert model.nutrition_gru.gru.input_size == NUTRITION_DIM
    assert model.activity_gru.gru.input_size == ACTIVITY_DIM
    assert model.sleep_gru.gru.input_size == SLEEP_DIM


def test_five_latent_outputs() -> None:
    batch = 4
    sequence_length = 12
    hidden_dim = 32

    model = FiveGRU(hidden_dim=hidden_dim)

    glucose = torch.randn(
        batch,
        sequence_length,
        GLUCOSE_DIM,
    )

    insulin = torch.randn(
        batch,
        sequence_length,
        INSULIN_DIM,
    )

    nutrition = torch.randn(
        batch,
        sequence_length,
        NUTRITION_DIM,
    )

    activity = torch.randn(
        batch,
        sequence_length,
        ACTIVITY_DIM,
    )

    sleep = torch.randn(
        batch,
        sequence_length,
        SLEEP_DIM,
    )

    outputs = model(
        glucose=glucose,
        insulin=insulin,
        nutrition=nutrition,
        activity=activity,
        sleep=sleep,
    )

    validate_output_contract(
        outputs,
        batch_size=batch,
        hidden_dim=hidden_dim,
    )

    assert set(outputs) == {
        "zG",
        "zI",
        "zN",
        "zA",
        "zS",
    }


def test_variable_sequence_lengths() -> None:
    batch = 3
    sequence_length = 10
    hidden_dim = 16

    model = FiveGRU(hidden_dim=hidden_dim)

    inputs = {
        "glucose": torch.randn(
            batch,
            sequence_length,
            GLUCOSE_DIM,
        ),
        "insulin": torch.randn(
            batch,
            sequence_length,
            INSULIN_DIM,
        ),
        "nutrition": torch.randn(
            batch,
            sequence_length,
            NUTRITION_DIM,
        ),
        "activity": torch.randn(
            batch,
            sequence_length,
            ACTIVITY_DIM,
        ),
        "sleep": torch.randn(
            batch,
            sequence_length,
            SLEEP_DIM,
        ),
    }

    lengths = {
        "glucose": torch.tensor([10, 7, 4]),
        "insulin": torch.tensor([9, 6, 3]),
        "nutrition": torch.tensor([8, 5, 2]),
        "activity": torch.tensor([10, 8, 6]),
        "sleep": torch.tensor([7, 7, 5]),
    }

    outputs = model(
        **inputs,
        lengths=lengths,
    )

    validate_output_contract(
        outputs,
        batch_size=batch,
        hidden_dim=hidden_dim,
    )