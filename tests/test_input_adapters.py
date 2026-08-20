from __future__ import annotations

import torch
import pytest

from src.models.input_adapters import (
    ActivityAdapter,
    IdentityNumericAdapter,
    InsulinAdapter,
    NutritionAdapter,
)


def test_glucose_identity_adapter() -> None:
    adapter = IdentityNumericAdapter(1)

    x = torch.randn(2, 5, 1)

    output = adapter(x)

    assert output.shape == (2, 5, 1)
    assert torch.equal(output, x)


def test_sleep_identity_adapter() -> None:
    adapter = IdentityNumericAdapter(6)

    x = torch.randn(2, 5, 6)

    output = adapter(x)

    assert output.shape == (2, 5, 6)
    assert torch.equal(output, x)


def test_nutrition_adapter_produces_24_features() -> None:
    torch.manual_seed(42)

    adapter = NutritionAdapter(
        meal_type_cardinality=15,
        meal_tag_cardinality=2006,
    )

    numeric = torch.randn(2, 5, 4)

    meal_type_ids = torch.randint(
        0,
        15,
        (2, 5),
    )

    meal_tag_ids = torch.randint(
        0,
        2006,
        (2, 5),
    )

    output = adapter(
        numeric,
        meal_type_ids,
        meal_tag_ids,
    )

    assert output.shape == (2, 5, 24)
    assert torch.isfinite(output).all()


def test_activity_adapter_produces_17_features() -> None:
    torch.manual_seed(42)

    adapter = ActivityAdapter(
        activity_type_cardinality=6,
        intensity_cardinality=3,
    )

    numeric = torch.randn(2, 5, 10)

    activity_type_ids = torch.randint(
        0,
        6,
        (2, 5),
    )

    intensity_ids = torch.randint(
        0,
        3,
        (2, 5),
    )

    output = adapter(
        numeric,
        activity_type_ids,
        intensity_ids,
    )

    assert output.shape == (2, 5, 17)
    assert torch.isfinite(output).all()


def test_insulin_adapter_produces_two_features() -> None:
    adapter = InsulinAdapter()

    dose = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
        ]
    )

    event_type = torch.tensor(
        [
            [0, 1, 0],
        ],
        dtype=torch.long,
    )

    output = adapter(
        dose,
        event_type,
    )

    expected = torch.tensor(
        [
            [
                [1.0, 0.0],
                [2.0, 1.0],
                [3.0, 0.0],
            ]
        ]
    )

    assert output.shape == (1, 3, 2)
    assert torch.equal(output, expected)


def test_nutrition_rejects_wrong_numeric_dimension() -> None:
    adapter = NutritionAdapter(
        meal_type_cardinality=15,
        meal_tag_cardinality=2006,
    )

    numeric = torch.randn(2, 5, 3)

    with pytest.raises(
        ValueError,
        match="expected 4 features",
    ):
        adapter(
            numeric,
            torch.zeros(2, 5, dtype=torch.long),
            torch.zeros(2, 5, dtype=torch.long),
        )


def test_activity_rejects_wrong_numeric_dimension() -> None:
    adapter = ActivityAdapter(
        activity_type_cardinality=6,
        intensity_cardinality=3,
    )

    numeric = torch.randn(2, 5, 9)

    with pytest.raises(
        ValueError,
        match="expected 10 features",
    ):
        adapter(
            numeric,
            torch.zeros(2, 5, dtype=torch.long),
            torch.zeros(2, 5, dtype=torch.long),
        )


def test_insulin_rejects_invalid_event_type() -> None:
    adapter = InsulinAdapter()

    dose = torch.ones(1, 3, 1)

    event_type = torch.tensor(
        [[0, 2, 1]],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="only 0.*or 1",
    ):
        adapter(
            dose,
            event_type,
        )


def test_category_indices_are_validated() -> None:
    adapter = NutritionAdapter(
        meal_type_cardinality=3,
        meal_tag_cardinality=4,
    )

    numeric = torch.randn(1, 2, 4)

    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        adapter(
            numeric,
            torch.tensor([[0, 3]]),
            torch.tensor([[0, 1]]),
        )