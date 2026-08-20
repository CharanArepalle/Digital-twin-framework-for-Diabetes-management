"""
T1D-UOM Digital Twin Dynamics Evaluation
-----------------------------------------
Consolidated independent evaluation of the already-trained TwinDynamics model.

Evaluates:
  1. One-step prediction on held-out participants.
  2. Multi-step recursive rollout on held-out participants.
  3. Finite/stability checks.
  4. Per-participant and aggregate metrics.
  5. Reproducible JSON + CSV evaluation artifacts.

Locked architecture:
    Unified Patient State (64)
             |
             v
        TwinDynamics
             |
             v
       Predicted Next State

No retraining.
No source-data modification.
No normalization.
No interpolation.
No imputation.
delta_t is NOT a model input.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .dynamics import TwinDynamics
from .build_transition_artifacts import (
    load_state_trajectory,
    build_transitions,
)


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

EVAL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "twin_evaluation"
)

BEST_CHECKPOINT = MODEL_ROOT / "twin_dynamics_best.pt"

STATE_DIM = 64
HORIZONS = (1, 5, 10, 30, 60)
DEFAULT_BATCH_SIZE = 4096

HELD_OUT = (
    "UoM2401",
    "UoM2405",
)


def finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).item())


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return math.sqrt(mse(pred, target))


def r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    target_mean = torch.mean(target)
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target_mean) ** 2)

    denom = float(ss_tot.item())
    if denom <= 0.0:
        return float("nan")

    return float((1.0 - ss_res / ss_tot).item())


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing TwinDynamics checkpoint: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    required = {
        "format",
        "state_dim",
        "hidden_dim",
        "context_dim",
        "dropout",
        "model_state_dict",
        "training_participants",
        "held_out_participants",
        "delta_t_as_model_input",
    }

    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(
            "TwinDynamics checkpoint is missing required fields: "
            + ", ".join(missing)
        )

    if int(checkpoint["state_dim"]) != STATE_DIM:
        raise RuntimeError(
            f"Checkpoint state dimension is {checkpoint['state_dim']}; "
            f"expected {STATE_DIM}."
        )

    if checkpoint["context_dim"] is not None:
        raise RuntimeError(
            "Checkpoint unexpectedly contains a context dimension: "
            f"{checkpoint['context_dim']}"
        )

    if bool(checkpoint["delta_t_as_model_input"]):
        raise RuntimeError(
            "Checkpoint violates the locked contract: "
            "delta_t must not be a model input."
        )

    held_out = tuple(checkpoint["held_out_participants"])
    if held_out != HELD_OUT:
        raise RuntimeError(
            "Checkpoint held-out partition does not match the locked "
            f"partition. Expected {HELD_OUT}, received {held_out}."
        )

    return checkpoint


def build_model(checkpoint: dict[str, Any], device: torch.device) -> TwinDynamics:
    model = TwinDynamics(
        state_dim=int(checkpoint["state_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        context_dim=checkpoint["context_dim"],
        dropout=float(checkpoint["dropout"]),
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    return model


def transition_tensors(
    participant: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    path = STATE_ROOT / f"{participant}_unified_state.csv"

    if not path.exists():
        raise RuntimeError(
            f"Missing held-out Unified Patient State trajectory: {path}"
        )

    trajectory = load_state_trajectory(
        path,
        expected_participant=participant,
    )

    transitions, _ = build_transitions(trajectory)

    if not transitions:
        raise RuntimeError(
            f"{participant}: no transitions were produced."
        )

    current = torch.stack(
        [item["current_state"] for item in transitions],
        dim=0,
    ).to(device=device, dtype=torch.float32)

    target = torch.stack(
        [item["next_state"] for item in transitions],
        dim=0,
    ).to(device=device, dtype=torch.float32)

    if current.ndim != 2 or current.shape[1] != STATE_DIM:
        raise RuntimeError(
            f"{participant}: unexpected current-state shape "
            f"{tuple(current.shape)}"
        )

    if target.shape != current.shape:
        raise RuntimeError(
            f"{participant}: current/next state shape mismatch: "
            f"{tuple(current.shape)} vs {tuple(target.shape)}"
        )

    if not finite_tensor(current):
        raise RuntimeError(f"{participant}: current states contain non-finite values.")

    if not finite_tensor(target):
        raise RuntimeError(f"{participant}: next states contain non-finite values.")

    timestamps = trajectory.timestamps

    if len(timestamps) != len(transitions) + 1:
        raise RuntimeError(
            f"{participant}: trajectory/transition count mismatch."
        )

    rows = []
    for i in range(len(transitions)):
        rows.append(
            {
                "participant_id": participant,
                "current_timestamp": timestamps[i],
                "next_timestamp": timestamps[i + 1],
                "delta_t_seconds": (
                    timestamps[i + 1] - timestamps[i]
                ).total_seconds(),
            }
        )

    transition_meta = pd.DataFrame(rows)

    return current, target, transition_meta


@torch.no_grad()
def evaluate_one_step(
    model: TwinDynamics,
    current: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    predictions = []

    for start in range(0, len(current), batch_size):
        batch = current[start : start + batch_size]
        pred = model(batch)

        if not finite_tensor(pred):
            raise RuntimeError(
                "TwinDynamics produced non-finite one-step predictions."
            )

        predictions.append(pred)

    pred = torch.cat(predictions, dim=0)

    return {
        "mse": mse(pred, target),
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
        "r2": r2(pred, target),
        "prediction_abs_max": float(pred.abs().max().item()),
        "prediction_abs_mean": float(pred.abs().mean().item()),
    }


@torch.no_grad()
def evaluate_rollouts(
    model: TwinDynamics,
    states: torch.Tensor,
    horizons: tuple[int, ...],
    batch_size: int,
) -> list[dict[str, float | int]]:
    """
    Recursive rollout evaluation.

    For each horizon H:
        state[t] -> model -> state_hat[t+1]
                   -> model -> state_hat[t+2]
                   ...
                   -> state_hat[t+H]

    Each horizon is compared with the actual state at t+H.
    """

    results = []

    for horizon in horizons:
        if len(states) <= horizon:
            continue

        starts = states[:-horizon]
        targets = states[horizon:]

        predictions = []

        for start in range(0, len(starts), batch_size):
            batch = starts[start : start + batch_size]
            simulated = batch

            for _ in range(horizon):
                simulated = model(simulated)

                if not finite_tensor(simulated):
                    raise RuntimeError(
                        f"Non-finite recursive rollout at horizon {horizon}."
                    )

            predictions.append(simulated)

        pred = torch.cat(predictions, dim=0)

        results.append(
            {
                "horizon_steps": horizon,
                "samples": int(len(pred)),
                "mse": mse(pred, targets),
                "rmse": rmse(pred, targets),
                "mae": mae(pred, targets),
                "r2": r2(pred, targets),
                "prediction_abs_max": float(pred.abs().max().item()),
                "prediction_abs_mean": float(pred.abs().mean().item()),
            }
        )

    return results


def summarize_delta_t(meta: pd.DataFrame) -> dict[str, float]:
    delta = pd.to_numeric(
        meta["delta_t_seconds"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(delta).all():
        raise RuntimeError("Held-out transition delta_t contains non-finite values.")

    if (delta <= 0).any():
        raise RuntimeError("Held-out transition delta_t contains non-positive values.")

    return {
        "delta_t_seconds_min": float(delta.min()),
        "delta_t_seconds_median": float(np.median(delta)),
        "delta_t_seconds_mean": float(delta.mean()),
        "delta_t_seconds_max": float(delta.max()),
    }


def evaluate_participant(
    model: TwinDynamics,
    participant: str,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print()
    print(f"[{participant}] Loading held-out trajectory...")

    current, target, meta = transition_tensors(
        participant,
        device,
    )

    print(f"  transitions : {len(current)}")
    print(f"  state dim   : {current.shape[1]}")

    dt_summary = summarize_delta_t(meta)

    one_step = evaluate_one_step(
        model,
        current,
        target,
        batch_size,
    )

    rollout = evaluate_rollouts(
        model,
        torch.cat(
            (
                current[:1],
                target,
            ),
            dim=0,
        ),
        HORIZONS,
        batch_size,
    )

    participant_summary = {
        "participant_id": participant,
        "states_evaluated": int(len(current) + 1),
        "transitions_evaluated": int(len(current)),
        **dt_summary,
        "one_step": one_step,
    }

    for item in rollout:
        item["participant_id"] = participant

    print(
        f"  one-step MSE  : {one_step['mse']:.10e}"
    )
    print(
        f"  one-step RMSE : {one_step['rmse']:.10e}"
    )
    print(
        f"  one-step MAE  : {one_step['mae']:.10e}"
    )

    for item in rollout:
        print(
            f"  rollout H={item['horizon_steps']:>2d} "
            f"MSE={item['mse']:.10e} "
            f"RMSE={item['rmse']:.10e} "
            f"MAE={item['mae']:.10e}"
        )

    return participant_summary, rollout


def aggregate_one_step(
    summaries: list[dict[str, Any]],
) -> dict[str, float]:
    values = [s["one_step"] for s in summaries]

    return {
        "mean_mse": float(np.mean([x["mse"] for x in values])),
        "mean_rmse": float(np.mean([x["rmse"] for x in values])),
        "mean_mae": float(np.mean([x["mae"] for x in values])),
        "mean_r2": float(np.nanmean([x["r2"] for x in values])),
        "max_mse": float(np.max([x["mse"] for x in values])),
        "max_mae": float(np.max([x["mae"] for x in values])),
    }


def aggregate_rollouts(
    rollout_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rollout_rows:
        return []

    frame = pd.DataFrame(rollout_rows)

    result = []

    for horizon, group in frame.groupby("horizon_steps", sort=True):
        result.append(
            {
                "horizon_steps": int(horizon),
                "participants": int(group["participant_id"].nunique()),
                "total_samples": int(group["samples"].sum()),
                "mean_mse": float(group["mse"].mean()),
                "mean_rmse": float(group["rmse"].mean()),
                "mean_mae": float(group["mae"].mean()),
                "mean_r2": float(group["r2"].mean()),
                "max_mse": float(group["mse"].max()),
                "max_mae": float(group["mae"].max()),
            }
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--participant",
        action="append",
        dest="participants",
        help="Held-out participant. May be repeated.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--device",
        default=None,
        help="cpu or cuda. Defaults to cuda when available.",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")

    participants = (
        tuple(args.participants)
        if args.participants
        else HELD_OUT
    )

    invalid = [p for p in participants if p not in HELD_OUT]
    if invalid:
        parser.error(
            "Only locked held-out participants are permitted: "
            + ", ".join(HELD_OUT)
        )

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print()
    print("=" * 80)
    print("T1D-UOM DIGITAL TWIN HELD-OUT EVALUATION")
    print("=" * 80)
    print()
    print("Architecture:")
    print("  Unified Patient State -> TwinDynamics -> Predicted Next State")
    print()
    print(f"State dimension : {STATE_DIM}")
    print(f"Device          : {device}")
    print(f"Batch size      : {args.batch_size}")
    print()
    print("Held-out participants:")
    for p in participants:
        print(f"  {p}")

    checkpoint = load_checkpoint(
        BEST_CHECKPOINT,
        device,
    )

    model = build_model(
        checkpoint,
        device,
    )

    print()
    print("Checkpoint:")
    print(f"  {BEST_CHECKPOINT}")
    print(f"  best epoch       : {checkpoint.get('epoch')}")
    print(f"  validation MSE   : {checkpoint.get('validation_loss'):.10e}")
    print()
    print("Training participants were not evaluated.")
    print("Held-out participants were not used for optimization.")
    print("Frozen source data is not modified.")

    summaries = []
    rollout_rows = []

    for participant in participants:
        summary, rollout = evaluate_participant(
            model,
            participant,
            device,
            args.batch_size,
        )

        summaries.append(summary)
        rollout_rows.extend(rollout)

    aggregate = {
        "one_step": aggregate_one_step(summaries),
        "rollout": aggregate_rollouts(rollout_rows),
    }

    EVAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = EVAL_ROOT / "held_out_evaluation_summary.json"
    participant_path = EVAL_ROOT / "held_out_participant_metrics.json"
    rollout_path = EVAL_ROOT / "held_out_rollout_metrics.csv"

    report = {
        "format": "t1d_uom_twin_dynamics_held_out_evaluation_v1",
        "architecture": {
            "state_dim": STATE_DIM,
            "hidden_dim": int(checkpoint["hidden_dim"]),
            "context_dim": checkpoint["context_dim"],
            "dropout": float(checkpoint["dropout"]),
            "delta_t_as_model_input": False,
        },
        "checkpoint": {
            "path": str(BEST_CHECKPOINT),
            "epoch": int(checkpoint["epoch"]),
            "validation_loss": float(checkpoint["validation_loss"]),
        },
        "held_out_participants": list(participants),
        "horizons": list(HORIZONS),
        "aggregate": aggregate,
        "source_policy": {
            "source_modification": False,
            "resampling": False,
            "interpolation": False,
            "imputation": False,
            "normalization": False,
        },
    }

    summary_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    participant_path.write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(rollout_rows).to_csv(
        rollout_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("HELD-OUT DIGITAL TWIN EVALUATION COMPLETED")
    print("=" * 80)
    print()
    print("Aggregate one-step metrics:")
    for key, value in aggregate["one_step"].items():
        print(f"  {key:16s}: {value:.10e}")

    print()
    print("Aggregate recursive rollout:")
    for row in aggregate["rollout"]:
        print(
            f"  H={row['horizon_steps']:>2d} "
            f"MSE={row['mean_mse']:.10e} "
            f"RMSE={row['mean_rmse']:.10e} "
            f"MAE={row['mean_mae']:.10e}"
        )

    print()
    print("Artifacts:")
    print(f"  summary       : {summary_path}")
    print(f"  participants  : {participant_path}")
    print(f"  rollouts      : {rollout_path}")
    print()
    print("No training was performed.")
    print("No held-out data was used for optimization.")
    print("Frozen source data was not modified.")


if __name__ == "__main__":
    main()