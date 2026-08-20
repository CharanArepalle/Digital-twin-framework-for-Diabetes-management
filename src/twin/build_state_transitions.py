from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import json
import torch

from src.twin.transitions import TwinTransitionDataset, build_transition_dataset


@dataclass(frozen=True)
class StateTrajectory:
    """One participant's already-generated Unified Patient State trajectory."""

    participant_id: str
    timestamps: tuple[object, ...]
    states: torch.Tensor

    def __post_init__(self) -> None:
        if not self.participant_id.strip():
            raise ValueError("participant_id must not be empty.")

        if not isinstance(self.states, torch.Tensor):
            raise TypeError("states must be a torch.Tensor.")

        if self.states.ndim != 2:
            raise ValueError(
                "states must have shape [num_observations, state_dim]."
            )

        if self.states.shape[0] != len(self.timestamps):
            raise ValueError(
                "states and timestamps must contain the same number "
                "of observations."
            )

        if self.states.shape[0] == 0:
            raise ValueError("trajectory must contain at least one observation.")

        if not torch.is_floating_point(self.states):
            raise TypeError("states must use a floating-point dtype.")

        if not torch.isfinite(self.states).all():
            raise ValueError("states must contain only finite values.")


def build_participant_transitions(
    trajectory: StateTrajectory,
) -> TwinTransitionDataset:
    """
    Build temporal transitions for one participant.

    Ordering is deliberately NOT changed here.

    The trajectory must already be chronologically ordered.
    """

    return build_transition_dataset(
        states=trajectory.states,
        timestamps=trajectory.timestamps,
        participant_ids=[
            trajectory.participant_id
        ] * trajectory.states.shape[0],
    )


def build_cohort_transitions(
    trajectories: Sequence[StateTrajectory],
) -> TwinTransitionDataset:
    """
    Build transitions across the supplied participant trajectories.

    Each participant is processed independently, preventing transitions
    from ever crossing participant boundaries.
    """

    if not trajectories:
        raise ValueError("trajectories must not be empty.")

    all_transitions = []

    for trajectory in trajectories:
        dataset = build_participant_transitions(trajectory)

        all_transitions.extend(dataset.transitions)

    return TwinTransitionDataset(all_transitions)


def save_trajectory(
    trajectory: StateTrajectory,
    output_path: str | Path,
) -> Path:
    """Persist one validated state trajectory."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": "t1d_uom_unified_patient_state_trajectory",
        "version": 1,
        "participant_id": trajectory.participant_id,
        "timestamps": list(trajectory.timestamps),
        "state_dim": int(trajectory.states.shape[1]),
        "states": trajectory.states.detach().cpu(),
    }

    torch.save(payload, destination)

    return destination


def load_trajectory(
    input_path: str | Path,
) -> StateTrajectory:
    """Load and validate one state trajectory."""

    source = Path(input_path)

    if not source.exists():
        raise FileNotFoundError(source)

    payload = torch.load(
        source,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise ValueError("Invalid trajectory artifact.")

    if payload.get("format") != (
        "t1d_uom_unified_patient_state_trajectory"
    ):
        raise ValueError("Unexpected trajectory artifact format.")

    participant_id = payload.get("participant_id")
    timestamps = payload.get("timestamps")
    states = payload.get("states")

    if not isinstance(participant_id, str):
        raise ValueError("Invalid participant_id.")

    if not isinstance(timestamps, list):
        raise ValueError("Invalid timestamps.")

    if not isinstance(states, torch.Tensor):
        raise ValueError("Invalid states tensor.")

    return StateTrajectory(
        participant_id=participant_id,
        timestamps=tuple(timestamps),
        states=states.float(),
    )


def save_transition_summary(
    dataset: TwinTransitionDataset,
    output_path: str | Path,
) -> Path:
    """Save a lightweight JSON summary of generated transitions."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    participants = {}

    for transition in dataset.transitions:
        participant = transition.participant_id or "UNKNOWN"
        participants[participant] = (
            participants.get(participant, 0) + 1
        )

    summary = {
        "format": "t1d_uom_twin_transition_summary",
        "version": 1,
        "num_transitions": len(dataset),
        "state_dim": dataset.state_dim,
        "participants": participants,
    }

    destination.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return destination


def _self_test() -> None:
    """Contract test for the transition artifact boundary."""

    states_a = torch.tensor(
        [
            [1.0] * 64,
            [2.0] * 64,
            [3.0] * 64,
        ],
        dtype=torch.float32,
    )

    states_b = torch.tensor(
        [
            [10.0] * 64,
            [11.0] * 64,
        ],
        dtype=torch.float32,
    )

    trajectory_a = StateTrajectory(
        participant_id="UoM2301",
        timestamps=(0.0, 60.0, 120.0),
        states=states_a,
    )

    trajectory_b = StateTrajectory(
        participant_id="UoM2302",
        timestamps=(0.0, 300.0),
        states=states_b,
    )

    dataset = build_cohort_transitions(
        [trajectory_a, trajectory_b]
    )

    assert len(dataset) == 3
    assert dataset.state_dim == 64

    for transition in dataset.transitions:
        assert transition.participant_id in {
            "UoM2301",
            "UoM2302",
        }
        assert transition.delta_t is not None
        assert transition.delta_t > 0

    print("STATE → TRANSITION ARTIFACT SELF-TEST")
    print("=" * 72)
    print("State dimension :", dataset.state_dim)
    print("Transitions     :", len(dataset))
    print("Participants    :", 2)
    print("Boundary safety : PASS")
    print("Temporal safety : PASS")
    print("SELF-TEST       : PASS")


if __name__ == "__main__":
    _self_test()