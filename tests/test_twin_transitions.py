"""
Tests for the T1D-UOM Digital Twin temporal transition contract.
"""

from __future__ import annotations

import pytest
import torch

from src.twin.transitions import (
    TwinTransition,
    TwinTransitionDataset,
    build_transition_dataset,
)


def _state(
    value: float,
    state_dim: int = 4,
) -> torch.Tensor:
    return torch.full(
        (state_dim,),
        value,
        dtype=torch.float32,
    )


def test_transition_accepts_matching_states() -> None:
    transition = TwinTransition(
        current_state=_state(1.0),
        next_state=_state(2.0),
        delta_t=300.0,
        participant_id="UoM2301",
    )

    assert transition.current_state.shape == (4,)
    assert transition.next_state.shape == (4,)
    assert transition.delta_t == 300.0
    assert transition.participant_id == "UoM2301"


def test_transition_rejects_state_shape_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="exactly the same shape",
    ):
        TwinTransition(
            current_state=_state(1.0, state_dim=4),
            next_state=_state(2.0, state_dim=5),
        )


def test_transition_rejects_non_vector_state() -> None:
    with pytest.raises(
        ValueError,
        match=r"shape \[state_dim\]",
    ):
        TwinTransition(
            current_state=torch.randn(2, 4),
            next_state=_state(2.0),
        )


def test_transition_rejects_non_finite_state() -> None:
    current = _state(1.0)
    current[0] = float("nan")

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        TwinTransition(
            current_state=current,
            next_state=_state(2.0),
        )


def test_transition_rejects_non_positive_delta_t() -> None:
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        TwinTransition(
            current_state=_state(1.0),
            next_state=_state(2.0),
            delta_t=0.0,
        )


def test_dataset_preserves_transition_contract() -> None:
    transitions = [
        TwinTransition(
            current_state=_state(1.0),
            next_state=_state(2.0),
            delta_t=60.0,
            participant_id="UoM2301",
        ),
        TwinTransition(
            current_state=_state(2.0),
            next_state=_state(3.0),
            delta_t=60.0,
            participant_id="UoM2301",
        ),
    ]

    dataset = TwinTransitionDataset(
        transitions
    )

    assert len(dataset) == 2
    assert dataset.state_dim == 4

    item = dataset[0]

    assert set(item.keys()) == {
        "current_state",
        "next_state",
        "delta_t",
    }

    assert item["current_state"].shape == (4,)
    assert item["next_state"].shape == (4,)
    assert item["delta_t"].item() == pytest.approx(60.0)


def test_dataset_returns_copies() -> None:
    transition = TwinTransition(
        current_state=_state(1.0),
        next_state=_state(2.0),
    )

    dataset = TwinTransitionDataset(
        [transition]
    )

    item = dataset[0]
    item["current_state"][0] = 999.0

    original = dataset[0]["current_state"]

    assert original[0].item() == pytest.approx(1.0)


def test_build_creates_adjacent_same_participant_pairs() -> None:
    states = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ]
    )

    dataset = build_transition_dataset(
        states=states,
        timestamps=[0.0, 60.0, 120.0],
        participant_ids=[
            "UoM2301",
            "UoM2301",
            "UoM2301",
        ],
    )

    assert len(dataset) == 2

    assert dataset[0]["current_state"].tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]

    assert dataset[0]["next_state"].tolist() == [
        2.0,
        2.0,
        2.0,
        2.0,
    ]

    assert dataset[1]["current_state"].tolist() == [
        2.0,
        2.0,
        2.0,
        2.0,
    ]


def test_build_never_crosses_participant_boundary() -> None:
    states = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ]
    )

    dataset = build_transition_dataset(
        states=states,
        timestamps=[0.0, 60.0, 120.0],
        participant_ids=[
            "UoM2301",
            "UoM2301",
            "UoM2302",
        ],
    )

    assert len(dataset) == 1

    assert (
        dataset.transitions[0].participant_id
        == "UoM2301"
    )


def test_build_rejects_backward_timestamp() -> None:
    states = torch.randn(3, 4)

    dataset = build_transition_dataset(
        states=states,
        timestamps=[0.0, 60.0, 30.0],
        participant_ids=[
            "UoM2301",
            "UoM2301",
            "UoM2301",
        ],
    )

    assert len(dataset) == 1


def test_build_rejects_equal_timestamp() -> None:
    states = torch.randn(3, 4)

    dataset = build_transition_dataset(
        states=states,
        timestamps=[0.0, 60.0, 60.0],
        participant_ids=[
            "UoM2301",
            "UoM2301",
            "UoM2301",
        ],
    )

    assert len(dataset) == 1


def test_build_requires_matching_metadata_lengths() -> None:
    states = torch.randn(3, 4)

    with pytest.raises(
        ValueError,
        match="timestamps length",
    ):
        build_transition_dataset(
            states=states,
            timestamps=[0.0, 60.0],
            participant_ids=[
                "UoM2301",
                "UoM2301",
                "UoM2301",
            ],
        )

    with pytest.raises(
        ValueError,
        match="participant_ids length",
    ):
        build_transition_dataset(
            states=states,
            timestamps=[0.0, 60.0, 120.0],
            participant_ids=[
                "UoM2301",
                "UoM2301",
            ],
        )


def test_build_rejects_raw_timestamp_strings() -> None:
    states = torch.randn(2, 4)

    with pytest.raises(
        TypeError,
        match="already-resolved",
    ):
        build_transition_dataset(
            states=states,
            timestamps=[
                "2024-01-01 00:00:00",
                "2024-01-01 00:05:00",
            ],
            participant_ids=[
                "UoM2301",
                "UoM2301",
            ],
        )


def test_empty_transition_dataset_is_supported() -> None:
    dataset = TwinTransitionDataset(())

    assert len(dataset) == 0
    assert dataset.state_dim is None


def test_build_single_observation_has_no_transition() -> None:
    states = torch.randn(1, 4)

    dataset = build_transition_dataset(
        states=states,
        timestamps=[0.0],
        participant_ids=["UoM2301"],
    )

    assert len(dataset) == 0