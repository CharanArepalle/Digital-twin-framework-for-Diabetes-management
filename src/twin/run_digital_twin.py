"""
T1D-UOM — Production Digital Twin Prediction + What-if Runner.

Architecture
------------
Unified Patient State
        |
        v
   DigitalTwin
        |
        v
   TwinDynamics
        |
        v
 Simulated State
     /       \
Prediction   What-if

This module is the application layer above the already validated
DigitalTwin state-management layer.

Responsibilities
----------------
- Load the already-trained TwinDynamics checkpoint.
- Load held-out Unified Patient State trajectories.
- Initialize a DigitalTwin from observed states.
- Produce recursive baseline predictions.
- Produce isolated latent-state what-if simulations.
- Compare simulations against observed future states where available.
- Save reproducible machine-readable artifacts.

Important
---------
The 64-dimensional Unified Patient State is a learned latent representation.
The current repository does not define individual latent dimensions as
specific physiological interventions.

Therefore the what-if experiment is explicitly reported as:

    latent-state perturbation / counterfactual simulation

It must NOT be interpreted as a direct insulin, meal, exercise, or
clinical intervention.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .digital_twin import DigitalTwin, DigitalTwinState
from .evaluate_dynamics import load_state_trajectory


# ============================================================================
# LOCKED PROJECT CONTRACT
# ============================================================================

STATE_DIM = 64
HIDDEN_DIM = 64

HELD_OUT_PARTICIPANTS = (
    "UoM2401",
    "UoM2405",
)

HORIZONS = (
    1,
    5,
    10,
    30,
    60,
)

# Fractional latent-state perturbation used for the What-if branch.
#
# Example:
#
#     scenario_state = baseline_state * 1.05
#
# This is intentionally a latent-state perturbation, NOT a claim about
# a specific physiological intervention.
SCENARIO_SCALE = 0.05

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "unified_state_trajectories"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "twin_models"
)

CHECKPOINT = (
    MODEL_ROOT
    / "twin_dynamics_best.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "digital_twin"
)


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # We do not force deterministic CUDA algorithms because some
    # PyTorch/CUDA operations may not have deterministic implementations.
    # The model is run in eval/no-grad mode and the seed is fixed.


# ============================================================================
# VALIDATION
# ============================================================================

def validate_state(
    state: torch.Tensor,
    *,
    name: str,
) -> None:
    if not isinstance(state, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if state.ndim != 2:
        raise ValueError(
            f"{name} must have shape [batch, {STATE_DIM}]. "
            f"Received {tuple(state.shape)}."
        )

    if state.shape[1] != STATE_DIM:
        raise ValueError(
            f"{name} has dimension {state.shape[1]}; "
            f"expected {STATE_DIM}."
        )

    if not torch.is_floating_point(state):
        raise TypeError(
            f"{name} must be floating point."
        )

    if not torch.isfinite(state).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )


def validate_twin_state(
    state: DigitalTwinState,
    *,
    name: str,
) -> None:
    if not isinstance(state, DigitalTwinState):
        raise TypeError(
            f"{name} must be a DigitalTwinState."
        )

    validate_state(
        state.value,
        name=f"{name}.value",
    )


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """
    Extract a model state_dict from common checkpoint layouts.

    The training pipeline may save either:
        1. a raw state_dict
        2. a dictionary containing model_state_dict
        3. a dictionary containing state_dict
    """

    if isinstance(checkpoint, dict):
        if checkpoint:
            if all(
                isinstance(k, str)
                and isinstance(v, torch.Tensor)
                for k, v in checkpoint.items()
            ):
                return checkpoint

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            candidate = checkpoint.get(key)

            if isinstance(candidate, dict):
                if all(
                    isinstance(k, str)
                    and isinstance(v, torch.Tensor)
                    for k, v in candidate.items()
                ):
                    return candidate

    raise RuntimeError(
        "Unable to identify a PyTorch model state_dict in checkpoint:\n"
        f"{CHECKPOINT}"
    )


def load_trained_twin(
    device: torch.device,
) -> DigitalTwin:
    if not CHECKPOINT.exists():
        raise RuntimeError(
            "Missing trained TwinDynamics checkpoint:\n"
            f"{CHECKPOINT}"
        )

    twin = DigitalTwin(
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        context_dim=None,
        dropout=0.0,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    state_dict = _extract_state_dict(
        checkpoint
    )

    # Training saved TwinDynamics directly, whereas DigitalTwin owns
    # TwinDynamics under the "dynamics" attribute.
    #
    # Therefore accept either:
    #
    #   network.0.weight
    #
    # or:
    #
    #   dynamics.network.0.weight
    #
    # depending on the checkpoint layout.
    if any(
        key.startswith("dynamics.")
        for key in state_dict
    ):
        twin_state_dict = state_dict
    else:
        twin_state_dict = {
            f"dynamics.{key}": value
            for key, value in state_dict.items()
        }

    missing, unexpected = twin.load_state_dict(
        twin_state_dict,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "TwinDynamics checkpoint is incompatible with the "
            "DigitalTwin architecture.\n"
            f"Missing keys    : {missing}\n"
            f"Unexpected keys : {unexpected}"
        )

    twin.to(device)
    twin.eval()

    return twin


# ============================================================================
# TRAJECTORY LOADING
# ============================================================================

def load_participant(
    participant: str,
):
    path = (
        STATE_ROOT
        / f"{participant}_unified_state.csv"
    )

    trajectory = load_state_trajectory(
        path,
        expected_participant=participant,
    )

    if trajectory.states.ndim != 2:
        raise RuntimeError(
            f"{participant}: invalid state shape "
            f"{tuple(trajectory.states.shape)}"
        )

    if trajectory.states.shape[1] != STATE_DIM:
        raise RuntimeError(
            f"{participant}: expected state dimension "
            f"{STATE_DIM}; received "
            f"{trajectory.states.shape[1]}"
        )

    if not torch.isfinite(
        trajectory.states
    ).all():
        raise RuntimeError(
            f"{participant}: trajectory contains non-finite states."
        )

    if len(trajectory.timestamps) != trajectory.states.shape[0]:
        raise RuntimeError(
            f"{participant}: timestamp/state length mismatch."
        )

    if len(trajectory.timestamps) < max(HORIZONS) + 1:
        raise RuntimeError(
            f"{participant}: trajectory is too short for "
            f"H={max(HORIZONS)}."
        )

    return trajectory


# ============================================================================
# RECURSIVE ROLLOUT
# ============================================================================

def recursive_rollout(
    twin: DigitalTwin,
    initial_state: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    """
    Recursively evolve a DigitalTwin.

    Returns
    -------
    Tensor
        Shape [steps, STATE_DIM].
    """

    validate_state(
        initial_state,
        name="initial_state",
    )

    if initial_state.shape[0] != 1:
        raise ValueError(
            "recursive_rollout currently expects batch size 1."
        )

    if steps <= 0:
        raise ValueError(
            "steps must be positive."
        )

    current = twin.initialize(
        initial_state.clone()
    )

    predictions = []

    with torch.no_grad():
        for _ in range(steps):
            current = twin.evolve(
                current
            )

            validate_twin_state(
                current,
                name="predicted_state",
            )

            predictions.append(
                current.value[0].detach().clone()
            )

    return torch.stack(
        predictions,
        dim=0,
    )


# ============================================================================
# WHAT-IF ROLLOUT
# ============================================================================

def build_scenario_delta(
    baseline_state: torch.Tensor,
) -> torch.Tensor:
    """
    Construct a small latent-state perturbation.

    scenario = baseline + delta

    Because the latent dimensions do not have an established physiological
    semantic mapping, this is deliberately an amplitude perturbation of
    the complete latent state.
    """

    validate_state(
        baseline_state,
        name="baseline_state",
    )

    delta = (
        baseline_state
        * SCENARIO_SCALE
    )

    validate_state(
        delta,
        name="scenario_delta",
    )

    return delta


def recursive_what_if_rollout(
    twin: DigitalTwin,
    baseline_state: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    validate_state(
        baseline_state,
        name="baseline_state",
    )

    if baseline_state.shape[0] != 1:
        raise ValueError(
            "recursive_what_if_rollout expects batch size 1."
        )

    baseline_twin = twin.initialize(
        baseline_state.clone()
    )

    delta = build_scenario_delta(
        baseline_state
    )

    scenario_twin = twin.scenario(
        baseline_twin,
        delta,
    )

    # Confirm the baseline was not modified.
    if not torch.equal(
        baseline_twin.value,
        baseline_state,
    ):
        raise RuntimeError(
            "DigitalTwin baseline state was unexpectedly modified "
            "during scenario construction."
        )

    predictions = []

    with torch.no_grad():
        for _ in range(steps):
            scenario_twin = twin.evolve(
                scenario_twin
            )

            validate_twin_state(
                scenario_twin,
                name="scenario_state",
            )

            predictions.append(
                scenario_twin.value[0]
                .detach()
                .clone()
            )

    return torch.stack(
        predictions,
        dim=0,
    )


# ============================================================================
# METRICS
# ============================================================================

def tensor_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    return float(
        torch.mean(
            (prediction - target) ** 2
        ).item()
    )


def tensor_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    return float(
        torch.sqrt(
            torch.mean(
                (prediction - target) ** 2
            )
        ).item()
    )


def tensor_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    return float(
        torch.mean(
            torch.abs(
                prediction - target
            )
        ).item()
    )


def tensor_r2(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    target_mean = torch.mean(
        target
    )

    ss_res = torch.sum(
        (target - prediction) ** 2
    )

    ss_tot = torch.sum(
        (target - target_mean) ** 2
    )

    if float(ss_tot.item()) <= 0.0:
        return float("nan")

    return float(
        (1.0 - ss_res / ss_tot).item()
    )


# ============================================================================
# ARTIFACT WRITING
# ============================================================================

def write_prediction_csv(
    participant: str,
    timestamps: list[pd.Timestamp],
    predictions: torch.Tensor,
) -> Path:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for step, timestamp in enumerate(
        timestamps,
        start=1,
    ):
        row = {
            "participant_id": participant,
            "step": step,
            "timestamp": timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

        values = predictions[
            step - 1
        ].detach().cpu().numpy()

        for dimension, value in enumerate(values):
            row[
                f"state_{dimension:02d}"
            ] = float(value)

        rows.append(row)

    path = (
        OUTPUT_ROOT
        / f"{participant}_prediction.csv"
    )

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )

    return path


def write_what_if_csv(
    participant: str,
    timestamps: list[pd.Timestamp],
    baseline: torch.Tensor,
    scenario: torch.Tensor,
) -> Path:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for step, timestamp in enumerate(
        timestamps,
        start=1,
    ):
        baseline_values = (
            baseline[step - 1]
            .detach()
            .cpu()
            .numpy()
        )

        scenario_values = (
            scenario[step - 1]
            .detach()
            .cpu()
            .numpy()
        )

        row = {
            "participant_id": participant,
            "step": step,
            "timestamp": timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

        for dimension in range(STATE_DIM):
            row[
                f"baseline_state_{dimension:02d}"
            ] = float(
                baseline_values[dimension]
            )

            row[
                f"what_if_state_{dimension:02d}"
            ] = float(
                scenario_values[dimension]
            )

            row[
                f"difference_{dimension:02d}"
            ] = float(
                scenario_values[dimension]
                - baseline_values[dimension]
            )

        rows.append(row)

    path = (
        OUTPUT_ROOT
        / f"{participant}_what_if.csv"
    )

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )

    return path


# ============================================================================
# PARTICIPANT EXPERIMENT
# ============================================================================

def run_participant(
    participant: str,
    twin: DigitalTwin,
    *,
    device: torch.device,
) -> dict[str, Any]:

    print()
    print(
        f"[{participant}] Loading held-out Unified Patient State..."
    )

    trajectory = load_participant(
        participant
    )

    states = trajectory.states.to(
        device=device,
        dtype=torch.float32,
    )

    timestamps = list(
        trajectory.timestamps
    )

    # Use a starting point with enough future observations for H=60.
    #
    # Choosing the middle of the trajectory gives us a genuine observed
    # future against which recursive predictions can be compared.
    start_index = (
        len(timestamps)
        // 2
    )

    max_horizon = max(
        HORIZONS
    )

    if (
        start_index + max_horizon
        >= len(timestamps)
    ):
        start_index = (
            len(timestamps)
            - max_horizon
            - 1
        )

    initial_state = (
        states[start_index]
        .unsqueeze(0)
    )

    validate_state(
        initial_state,
        name="initial_state",
    )

    print(
        f"  states          : {len(states)}"
    )
    print(
        f"  state dimension  : {states.shape[1]}"
    )
    print(
        f"  start index      : {start_index}"
    )
    print(
        f"  start timestamp  : "
        f"{timestamps[start_index]}"
    )

    # ------------------------------------------------------------------
    # Generate the maximum baseline and What-if rollouts once.
    # ------------------------------------------------------------------

    baseline_rollout = recursive_rollout(
        twin,
        initial_state,
        steps=max_horizon,
    )

    what_if_rollout = recursive_what_if_rollout(
        twin,
        initial_state,
        steps=max_horizon,
    )

    validate_state(
        baseline_rollout,
        name="baseline_rollout",
    )

    validate_state(
        what_if_rollout,
        name="what_if_rollout",
    )

    # ------------------------------------------------------------------
    # Baseline / What-if comparison.
    # ------------------------------------------------------------------

    scenario_difference = (
        what_if_rollout
        - baseline_rollout
    )

    if not torch.isfinite(
        scenario_difference
    ).all():
        raise RuntimeError(
            f"{participant}: non-finite scenario difference."
        )

    future_timestamps = [
        timestamps[
            start_index + horizon
        ]
        for horizon in range(
            1,
            max_horizon + 1,
        )
    ]

    prediction_path = write_prediction_csv(
        participant,
        future_timestamps,
        baseline_rollout,
    )

    what_if_path = write_what_if_csv(
        participant,
        future_timestamps,
        baseline_rollout,
        what_if_rollout,
    )

    # ------------------------------------------------------------------
    # Quantitative validation against real future states.
    # ------------------------------------------------------------------

    metrics = {}

    for horizon in HORIZONS:
        prediction = baseline_rollout[
            horizon - 1
        ]

        target = states[
            start_index + horizon
        ]

        metrics[str(horizon)] = {
            "mse": tensor_mse(
                prediction,
                target,
            ),
            "rmse": tensor_rmse(
                prediction,
                target,
            ),
            "mae": tensor_mae(
                prediction,
                target,
            ),
            "r2": tensor_r2(
                prediction,
                target,
            ),
        }

    # ------------------------------------------------------------------
    # What-if effect magnitude.
    # ------------------------------------------------------------------

    what_if_metrics = {}

    for horizon in HORIZONS:
        difference = scenario_difference[
            horizon - 1
        ]

        what_if_metrics[str(horizon)] = {
            "mean_absolute_latent_difference": float(
                torch.mean(
                    torch.abs(difference)
                ).item()
            ),
            "rmse_latent_difference": float(
                torch.sqrt(
                    torch.mean(
                        difference ** 2
                    )
                ).item()
            ),
            "max_absolute_latent_difference": float(
                torch.max(
                    torch.abs(difference)
                ).item()
            ),
        }

    comparison = {
        "participant_id": participant,
        "state_dimension": STATE_DIM,
        "start_index": start_index,
        "start_timestamp": str(
            timestamps[start_index]
        ),
        "prediction_horizons": list(
            HORIZONS
        ),
        "scenario_type": (
            "latent-state perturbation / "
            "counterfactual simulation"
        ),
        "scenario_scale": SCENARIO_SCALE,
        "baseline_prediction_metrics": metrics,
        "what_if_difference_metrics": what_if_metrics,
        "artifacts": {
            "prediction": str(
                prediction_path
            ),
            "what_if": str(
                what_if_path
            ),
        },
    }

    comparison_path = (
        OUTPUT_ROOT
        / f"{participant}_comparison.json"
    )

    with comparison_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            comparison,
            handle,
            indent=2,
        )

    print(
        f"  prediction artifact : {prediction_path}"
    )
    print(
        f"  what-if artifact    : {what_if_path}"
    )
    print(
        f"  comparison artifact : {comparison_path}"
    )

    print()
    print(
        f"  Prediction metrics:"
    )

    for horizon in HORIZONS:
        item = metrics[
            str(horizon)
        ]

        print(
            f"    H={horizon:2d} "
            f"MSE={item['mse']:.8e} "
            f"RMSE={item['rmse']:.8e} "
            f"MAE={item['mae']:.8e} "
            f"R2={item['r2']:.6f}"
        )

    print()
    print(
        "  What-if latent-state effect:"
    )

    for horizon in HORIZONS:
        item = what_if_metrics[
            str(horizon)
        ]

        print(
            f"    H={horizon:2d} "
            f"mean_abs={item['mean_absolute_latent_difference']:.8e} "
            f"RMSE={item['rmse_latent_difference']:.8e}"
        )

    return comparison


# ============================================================================
# SELF TEST
# ============================================================================

def self_test() -> None:
    print()
    print("=" * 80)
    print("DIGITAL TWIN APPLICATION SELF-TEST")
    print("=" * 80)

    set_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"State dimension : {STATE_DIM}"
    )
    print(
        f"Device          : {device}"
    )

    twin = DigitalTwin(
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        context_dim=None,
        dropout=0.0,
    ).to(device)

    twin.eval()

    state = torch.randn(
        1,
        STATE_DIM,
        device=device,
        dtype=torch.float32,
    )

    validate_state(
        state,
        name="test_state",
    )

    # Initialize.
    baseline = twin.initialize(
        state
    )

    validate_twin_state(
        baseline,
        name="baseline",
    )

    print(
        "Initialization contract : PASS"
    )

    # Evolution.
    with torch.no_grad():
        evolved = twin.evolve(
            baseline
        )

    validate_twin_state(
        evolved,
        name="evolved",
    )

    print(
        "Prediction evolution    : PASS"
    )

    # Scenario.
    delta = (
        state
        * SCENARIO_SCALE
    )

    scenario = twin.scenario(
        baseline,
        delta,
    )

    validate_twin_state(
        scenario,
        name="scenario",
    )

    if not torch.equal(
        baseline.value,
        state,
    ):
        raise RuntimeError(
            "Scenario construction modified the baseline state."
        )

    expected_scenario = (
        state + delta
    )

    if not torch.allclose(
        scenario.value,
        expected_scenario,
    ):
        raise RuntimeError(
            "Scenario state does not equal state + delta."
        )

    print(
        "What-if isolation        : PASS"
    )

    # Forward alias.
    with torch.no_grad():
        forwarded = twin.forward(
            baseline
        )

    validate_twin_state(
        forwarded,
        name="forwarded",
    )

    print(
        "Forward interface        : PASS"
    )

    # Finite outputs.
    if not torch.isfinite(
        evolved.value
    ).all():
        raise RuntimeError(
            "Evolution produced non-finite values."
        )

    if not torch.isfinite(
        scenario.value
    ).all():
        raise RuntimeError(
            "Scenario produced non-finite values."
        )

    print(
        "Finite-state contract    : PASS"
    )

    print(
        "Architecture             : "
        "Unified State -> DigitalTwin -> TwinDynamics"
    )

    print()
    print(
        "SELF-TEST                : PASS"
    )
    print()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the T1D-UOM Digital Twin prediction and "
            "latent-state What-if experiment."
        )
    )

    parser.add_argument(
        "--participant",
        type=str,
        default=None,
        help="Run one held-out participant.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all held-out participants.",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run Digital Twin application self-test.",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.participant is not None:
        if args.participant not in HELD_OUT_PARTICIPANTS:
            parser.error(
                "Only held-out participants are allowed: "
                f"{', '.join(HELD_OUT_PARTICIPANTS)}"
            )

        participants = [
            args.participant
        ]

    elif args.all:
        participants = list(
            HELD_OUT_PARTICIPANTS
        )

    else:
        parser.error(
            "Specify --participant UoM2401, "
            "--participant UoM2405, or --all."
        )

    set_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 80)
    print("T1D-UOM DIGITAL TWIN PREDICTION + WHAT-IF")
    print("=" * 80)
    print()
    print("Architecture:")
    print(
        "  Unified Patient State -> DigitalTwin "
        "-> TwinDynamics"
    )
    print(
        "  Simulated State -> Prediction / What-if"
    )
    print()
    print(
        f"State dimension       : {STATE_DIM}"
    )
    print(
        f"TwinDynamics hidden   : {HIDDEN_DIM}"
    )
    print(
        f"Device                : {device}"
    )
    print(
        f"Prediction horizons   : {list(HORIZONS)}"
    )
    print(
        f"What-if scale         : {SCENARIO_SCALE}"
    )
    print()
    print(
        "Checkpoint:"
    )
    print(
        f"  {CHECKPOINT}"
    )
    print()
    print(
        "Held-out participants:"
    )

    for participant in participants:
        print(
            f"  {participant}"
        )

    print()
    print(
        "Source policy:"
    )
    print(
        "  Source modification : NO"
    )
    print(
        "  Retraining          : NO"
    )
    print(
        "  Resampling          : NO"
    )
    print(
        "  Interpolation       : NO"
    )
    print(
        "  Imputation          : NO"
    )
    print(
        "  Normalization       : NO"
    )

    print()
    print(
        "Loading trained Digital Twin..."
    )

    twin = load_trained_twin(
        device
    )

    print(
        "  checkpoint           : PASS"
    )
    print(
        "  model state          : PASS"
    )
    print(
        "  evaluation mode      : PASS"
    )

    results = []

    for participant in participants:
        print()
        print("-" * 80)

        try:
            result = run_participant(
                participant,
                twin,
                device=device,
            )

            results.append(
                result
            )

            print()
            print(
                f"[{participant}] PASS"
            )

        except Exception as exc:
            print()
            print(
                "=" * 80
            )
            print(
                f"DIGITAL TWIN FAILED: {participant}"
            )
            print(
                "=" * 80
            )
            print(
                str(exc)
            )
            raise

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "architecture": (
            "Unified Patient State -> DigitalTwin "
            "-> TwinDynamics -> Simulated State"
        ),
        "state_dimension": STATE_DIM,
        "hidden_dimension": HIDDEN_DIM,
        "checkpoint": str(
            CHECKPOINT
        ),
        "device": str(device),
        "held_out_participants": list(
            participants
        ),
        "prediction_horizons": list(
            HORIZONS
        ),
        "scenario_type": (
            "latent-state perturbation / "
            "counterfactual simulation"
        ),
        "scenario_scale": SCENARIO_SCALE,
        "retraining_performed": False,
        "source_data_modified": False,
        "results": results,
    }

    summary_path = (
        OUTPUT_ROOT
        / "digital_twin_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print()
    print("=" * 80)
    print(
        "DIGITAL TWIN EXPERIMENT COMPLETED SUCCESSFULLY"
    )
    print("=" * 80)
    print()
    print(
        f"Participants processed : {len(results)}"
    )
    print(
        f"Output directory       : {OUTPUT_ROOT}"
    )
    print(
        f"Summary                : {summary_path}"
    )
    print()
    print(
        "Prediction             : COMPLETE"
    )
    print(
        "What-if simulation     : COMPLETE"
    )
    print(
        "Held-out evaluation    : COMPLETE"
    )
    print()


if __name__ == "__main__":
    main()