"""
Tests for the Unified Patient State -> transition bridge.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import torch

from src.twin.build_transitions import (
    StateRecord,
    build_transitions,
)
from src.twin.transitions import (
    TwinTransitionDataset,
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


def _record(
    value: float,
    timestamp: object,
    participant_id: str = "UoM2301",
) -> StateRecord:
    return StateRecord(
        state=_state(value),
        timestamp=timestamp,
        participant_id=participant_id,
    )


def test_builds_adjacent_transitions() -> None:
    records = [
        _record(1.0, 0.0),
        _record(2.0, 60.0),
        _record(3.0, 120.0),
    ]

    dataset = build_transitions(
        records
    )

    assert isinstance(
        dataset,
        TwinTransitionDataset,
    )

    assert len(dataset) == 2
    assert dataset.state_dim == 4

    first = dataset[0]

    assert first["current_state"].tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]

    assert first["next_state"].tolist() == [
        2.0,
        2.0,
        2.0,
        2.0,
    ]

    assert first["delta_t"].item() == pytest.approx(
        60.0
    )


def test_does_not_cross_participant_boundary() -> None:
    records = [
        _record(1.0, 0.0, "UoM2301"),
        _record(2.0, 60.0, "UoM2301"),
        _record(3.0, 120.0, "UoM2302"),
        _record(4.0, 180.0, "UoM2302"),
    ]

    dataset = build_transitions(
        records
    )

    assert len(dataset) == 2

    assert (
        dataset.transitions[0].participant_id
        == "UoM2301"
    )

    assert (
        dataset.transitions[1].participant_id
        == "UoM2302"
    )


def test_backward_timestamp_does_not_create_transition() -> None:
    records = [
        _record(1.0, 0.0),
        _record(2.0, 60.0),
        _record(3.0, 30.0),
    ]

    dataset = build_transitions(
        records
    )

    assert len(dataset) == 1


def test_equal_timestamp_does_not_create_transition() -> None:
    records = [
        _record(1.0, 0.0),
        _record(2.0, 60.0),
        _record(3.0, 60.0),
    ]

    dataset = build_transitions(
        records
    )

    assert len(dataset) == 1


def test_datetime_timestamps_are_supported() -> None:
    base = datetime(
        2024,
        1,
        1,
        tzinfo=timezone.utc,
    )

    records = [
        _record(
            1.0,
            base,
        ),
        _record(
            2.0,
            base.replace(
                minute=5
            ),
        ),
    ]

    dataset = build_transitions(
        records
    )

    assert len(dataset) == 1

    assert dataset[0]["delta_t"].item() == pytest.approx(
        300.0
    )


def test_raw_timestamp_strings_are_rejected() -> None:
    records = [
        _record(
            1.0,
            "2024-01-01 00:00:00",
        ),
        _record(
            2.0,
            "2024-01-01 00:05:00",
        ),
    ]

    with pytest.raises(
        TypeError,
        match="already-resolved",
    ):
        build_transitions(
            records
        )


def test_empty_records_return_empty_dataset() -> None:
    dataset = build_transitions(
        []
    )

    assert isinstance(
        dataset,
        TwinTransitionDataset,
    )

    assert len(dataset) == 0
    assert dataset.state_dim is None


def test_single_record_has_no_transition() -> None:
    dataset = build_transitions(
        [
            _record(1.0, 0.0),
        ]
    )

    assert len(dataset) == 0


def test_state_dimension_mismatch_is_rejected() -> None:
    records = [
        StateRecord(
            state=torch.ones(
                4,
                dtype=torch.float32,
            ),
            timestamp=0.0,
            participant_id="UoM2301",
        ),
        StateRecord(
            state=torch.ones(
                5,
                dtype=torch.float32,
            ),
            timestamp=60.0,
            participant_id="UoM2301",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="same state dimension",
    ):
        build_transitions(
            records
        )


def test_non_finite_state_is_rejected() -> None:
    state = _state(1.0)
    state[0] = float("nan")

    records = [
        StateRecord(
            state=state,
            timestamp=0.0,
            participant_id="UoM2301",
        ),
        _record(
            2.0,
            60.0,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        build_transitions(
            records
        )


def test_integer_state_is_rejected() -> None:
    records = [
        StateRecord(
            state=torch.ones(
                4,
                dtype=torch.int64,
            ),
            timestamp=0.0,
            participant_id="UoM2301",
        ),
        _record(
            2.0,
            60.0,
        ),
    ]

    with pytest.raises(
        TypeError,
        match="floating-point",
    ):
        build_transitions(
            records
        )


def test_empty_participant_id_is_rejected() -> None:
    records = [
        StateRecord(
            state=_state(1.0),
            timestamp=0.0,
            participant_id="   ",
        ),
        _record(
            2.0,
            60.0,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        build_transitions(
            records
        )


def test_input_states_are_not_modified() -> None:
    first = _state(1.0)
    second = _state(2.0)

    original_first = first.clone()
    original_second = second.clone()

    records = [
        StateRecord(
            state=first,
            timestamp=0.0,
            participant_id="UoM2301",
        ),
        StateRecord(
            state=second,
            timestamp=60.0,
            participant_id="UoM2301",
        ),
    ]

    build_transitions(
        records
    )

    assert torch.equal(
        first,
        original_first,
    )

    assert torch.equal(
        second,
        original_second,
    )


def test_records_are_not_sorted_by_bridge() -> None:
    """
    The bridge must preserve the existing temporal contract:
    callers are responsible for supplying intended order.
    """

    records = [
        _record(2.0, 60.0),
        _record(1.0, 0.0),
    ]

    dataset = build_transitions(
        records
    )

    assert len(dataset) == 0


def test_transition_metadata_is_preserved() -> None:
    records = [
        _record(
            1.0,
            100.0,
            "UoM2301",
        ),
        _record(
            2.0,
            250.0,
            "UoM2301",
        ),
    ]

    dataset = build_transitions(
        records
    )

    assert len(dataset) == 1

    transition = dataset.transitions[0]

    assert transition.participant_id == "UoM2301"
    assert transition.current_time == 100.0
    assert transition.next_time == 250.0
    assert transition.delta_t == pytest.approx(
        150.0
    )