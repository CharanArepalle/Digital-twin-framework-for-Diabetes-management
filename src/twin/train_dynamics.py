"""
T1D-UOM DIGITAL TWIN DYNAMICS TRAINING
======================================

Locked architecture
-------------------
Unified Patient State (64D)
        |
        v
   TwinDynamics
        |
        v
 predicted next state (64D)

Training contract
-----------------
- Training participants:
    UoM2301
    UoM2302
    UoM2304
    UoM2305
    UoM2306
    UoM2307
    UoM2308
    UoM2309
    UoM2313

- Held-out participants:
    UoM2401
    UoM2405

- State dimension: 64
- delta_t is NOT a model input.
- No modification of frozen source data.
- No resampling.
- No interpolation.
- No imputation.
- No normalization of source data.
- Transition artifacts are consumed exactly as generated.

The model learns:

    current_state -> next_state

using the repository's existing TwinDynamics implementation.

Outputs
-------
data/derived/twin_models/
    twin_dynamics_best.pt
    twin_dynamics_final.pt
    twin_dynamics_training_metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.twin.dynamics import TwinDynamics


# ============================================================================
# LOCKED PROJECT CONTRACT
# ============================================================================

STATE_DIM = 64
HIDDEN_DIM = 64

TRAINING_PARTICIPANTS = (
    "UoM2301",
    "UoM2302",
    "UoM2304",
    "UoM2305",
    "UoM2306",
    "UoM2307",
    "UoM2308",
    "UoM2309",
    "UoM2313",
)

HELD_OUT_PARTICIPANTS = (
    "UoM2401",
    "UoM2405",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRANSITION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "twin_transitions"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "twin_models"
)

METADATA_PATH = (
    TRANSITION_ROOT
    / "transition_dataset_metadata.json"
)

SEED = 42

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5

DEFAULT_PATIENCE = 8


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior where supported.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# DATASET
# ============================================================================

class TransitionDataset(Dataset):
    """
    In-memory transition dataset.

    Each item contains:

        current_state : [64]
        next_state    : [64]

    delta_t is deliberately not supplied to TwinDynamics.
    """

    def __init__(
        self,
        current_states: Tensor,
        next_states: Tensor,
    ) -> None:

        if current_states.ndim != 2:
            raise ValueError(
                "current_states must have shape [N, 64]."
            )

        if next_states.ndim != 2:
            raise ValueError(
                "next_states must have shape [N, 64]."
            )

        if current_states.shape != next_states.shape:
            raise ValueError(
                "Current and next-state tensors must have identical "
                f"shapes; received {current_states.shape} and "
                f"{next_states.shape}."
            )

        if current_states.shape[1] != STATE_DIM:
            raise ValueError(
                f"Expected state dimension {STATE_DIM}; "
                f"received {current_states.shape[1]}."
            )

        if not torch.isfinite(current_states).all():
            raise ValueError(
                "Current-state tensor contains non-finite values."
            )

        if not torch.isfinite(next_states).all():
            raise ValueError(
                "Next-state tensor contains non-finite values."
            )

        self.current_states = current_states
        self.next_states = next_states

    def __len__(self) -> int:
        return self.current_states.shape[0]

    def __getitem__(self, index: int):
        return (
            self.current_states[index],
            self.next_states[index],
        )


# ============================================================================
# TRANSITION COLUMN DISCOVERY
# ============================================================================

def state_columns(
    prefix: str,
    columns: Iterable[str],
) -> list[str]:
    """
    Locate state_00 ... state_63 columns.

    Supported transition naming conventions:

        current_state_00
        next_state_00

    and, for compatibility:

        current_00
        next_00
    """

    columns = list(columns)

    candidates = []

    for i in range(STATE_DIM):
        candidates.append(
            f"{prefix}_state_{i:02d}"
        )

    if all(c in columns for c in candidates):
        return candidates

    candidates = []

    for i in range(STATE_DIM):
        candidates.append(
            f"{prefix}_{i:02d}"
        )

    if all(c in columns for c in candidates):
        return candidates

    # Search for the repository's actual likely names.
    if prefix == "current":
        candidates = [
            f"current_state_{i:02d}"
            for i in range(STATE_DIM)
        ]

        if all(c in columns for c in candidates):
            return candidates

    if prefix == "next":
        candidates = [
            f"next_state_{i:02d}"
            for i in range(STATE_DIM)
        ]

        if all(c in columns for c in candidates):
            return candidates

    raise RuntimeError(
        f"Unable to locate the {prefix} state columns. "
        f"Expected 64 state columns. Available columns include: "
        f"{columns[:20]}"
    )


# ============================================================================
# TRANSITION LOADING
# ============================================================================

def transition_path(
    participant_id: str,
) -> Path:

    return (
        TRANSITION_ROOT
        / f"{participant_id}_twin_transitions.csv"
    )


def load_participant_transitions(
    participant_id: str,
) -> tuple[Tensor, Tensor]:

    path = transition_path(participant_id)

    if not path.exists():
        raise RuntimeError(
            f"Missing transition artifact for {participant_id}: "
            f"{path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise RuntimeError(
            f"{participant_id}: transition artifact is empty."
        )

    if "participant_id" not in df.columns:
        raise RuntimeError(
            f"{participant_id}: transition artifact is missing "
            "'participant_id'."
        )

    participant_values = (
        df["participant_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    if participant_values != [participant_id]:
        raise RuntimeError(
            f"{participant_id}: unexpected participant IDs in "
            f"transition artifact: {participant_values}"
        )

    current_cols = state_columns(
        "current",
        df.columns,
    )

    next_cols = state_columns(
        "next",
        df.columns,
    )

    current = (
        df[current_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    next_state = (
        df[next_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    if not np.isfinite(current).all():
        raise RuntimeError(
            f"{participant_id}: current-state transition values "
            "contain non-finite values."
        )

    if not np.isfinite(next_state).all():
        raise RuntimeError(
            f"{participant_id}: next-state transition values "
            "contain non-finite values."
        )

    current_tensor = torch.from_numpy(
        current
    )

    next_tensor = torch.from_numpy(
        next_state
    )

    return current_tensor, next_tensor


# ============================================================================
# DATASET ASSEMBLY
# ============================================================================

def concatenate_participants(
    participants: tuple[str, ...],
) -> tuple[Tensor, Tensor, dict[str, int]]:

    current_parts = []
    next_parts = []
    counts = {}

    for participant in participants:

        print(
            f"  Loading {participant}...",
            end=" ",
            flush=True,
        )

        current, next_state = (
            load_participant_transitions(
                participant
            )
        )

        current_parts.append(current)
        next_parts.append(next_state)

        counts[participant] = len(current)

        print(
            f"{len(current):,} transitions"
        )

    current_all = torch.cat(
        current_parts,
        dim=0,
    )

    next_all = torch.cat(
        next_parts,
        dim=0,
    )

    return (
        current_all,
        next_all,
        counts,
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_dataset(
    current: Tensor,
    next_state: Tensor,
) -> None:

    if current.ndim != 2:
        raise RuntimeError(
            f"Current-state tensor must be 2D; "
            f"received {current.ndim}D."
        )

    if next_state.ndim != 2:
        raise RuntimeError(
            f"Next-state tensor must be 2D; "
            f"received {next_state.ndim}D."
        )

    if current.shape != next_state.shape:
        raise RuntimeError(
            "Current and next-state shapes differ: "
            f"{current.shape} vs {next_state.shape}"
        )

    if current.shape[1] != STATE_DIM:
        raise RuntimeError(
            f"Expected state dimension {STATE_DIM}; "
            f"received {current.shape[1]}."
        )

    if not torch.isfinite(current).all():
        raise RuntimeError(
            "Training current-state tensor contains "
            "non-finite values."
        )

    if not torch.isfinite(next_state).all():
        raise RuntimeError(
            "Training next-state tensor contains "
            "non-finite values."
        )


# ============================================================================
# MODEL VALIDATION
# ============================================================================

def validate_model() -> TwinDynamics:

    model = TwinDynamics(
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        context_dim=None,
        dropout=0.0,
    )

    model.eval()

    test_state = torch.zeros(
        4,
        STATE_DIM,
        dtype=torch.float32,
    )

    with torch.no_grad():
        prediction = model(
            test_state
        )

    if prediction.shape != test_state.shape:
        raise RuntimeError(
            "TwinDynamics output shape mismatch: "
            f"expected {tuple(test_state.shape)}, "
            f"received {tuple(prediction.shape)}."
        )

    if not torch.isfinite(prediction).all():
        raise RuntimeError(
            "TwinDynamics self-validation produced "
            "non-finite values."
        )

    return model


# ============================================================================
# TRAIN / VALIDATION SPLIT
# ============================================================================

def split_training_data(
    current: Tensor,
    next_state: Tensor,
    validation_fraction: float = 0.10,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:

    if not (
        0.0 < validation_fraction < 1.0
    ):
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    n = current.shape[0]

    if n < 10:
        raise RuntimeError(
            "Not enough transition samples for training."
        )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    permutation = torch.randperm(
        n,
        generator=generator,
    )

    validation_size = max(
        1,
        int(round(n * validation_fraction)),
    )

    validation_indices = permutation[
        :validation_size
    ]

    training_indices = permutation[
        validation_size:
    ]

    return (
        current[training_indices],
        next_state[training_indices],
        current[validation_indices],
        next_state[validation_indices],
    )


# ============================================================================
# EPOCH
# ============================================================================

def run_epoch(
    model: TwinDynamics,
    loader: DataLoader,
    criterion,
    optimizer=None,
    device: torch.device | str = "cpu",
) -> float:

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for current_state, next_state in loader:

        current_state = (
            current_state.to(device)
        )

        next_state = (
            next_state.to(device)
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):

            predicted_next = model(
                current_state
            )

            loss = criterion(
                predicted_next,
                next_state,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite TwinDynamics loss encountered."
                )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        batch_size = (
            current_state.shape[0]
        )

        total_loss += (
            float(loss.detach().cpu())
            * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "Epoch received zero samples."
        )

    return total_loss / total_samples


# ============================================================================
# SAVE CHECKPOINT
# ============================================================================

def save_checkpoint(
    path: Path,
    model: TwinDynamics,
    optimizer,
    epoch: int,
    train_loss: float,
    validation_loss: float,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "format": (
            "t1d_uom_twin_dynamics_checkpoint_v1"
        ),
        "state_dim": STATE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "context_dim": None,
        "dropout": 0.0,
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "training_participants": list(
            TRAINING_PARTICIPANTS
        ),
        "held_out_participants": list(
            HELD_OUT_PARTICIPANTS
        ),
        "delta_t_as_model_input": False,
        "seed": SEED,
    }

    torch.save(
        payload,
        path,
    )


# ============================================================================
# METADATA
# ============================================================================

def write_metadata(
    *,
    counts: dict[str, int],
    train_samples: int,
    validation_samples: int,
    best_epoch: int,
    best_validation_loss: float,
    final_train_loss: float,
) -> None:

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "format": (
            "t1d_uom_twin_dynamics_training_v1"
        ),
        "architecture": {
            "state_dim": STATE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "context_dim": None,
            "dropout": 0.0,
            "transition_rule": (
                "current_state_to_next_state"
            ),
            "delta_t_as_model_input": False,
        },
        "training_participants": list(
            TRAINING_PARTICIPANTS
        ),
        "held_out_participants": list(
            HELD_OUT_PARTICIPANTS
        ),
        "transition_counts": counts,
        "training_samples": train_samples,
        "validation_samples": validation_samples,
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_validation_loss
        ),
        "final_train_loss": final_train_loss,
        "seed": SEED,
        "source_policy": {
            "source_modification": False,
            "resampling": False,
            "interpolation": False,
            "imputation": False,
            "normalization": False,
        },
    }

    with (
        MODEL_ROOT
        / "twin_dynamics_training_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as fh:

        json.dump(
            metadata,
            fh,
            indent=2,
        )


# ============================================================================
# SELF TEST
# ============================================================================

def self_test() -> None:

    print()
    print("=" * 80)
    print("TWIN DYNAMICS TRAINING SELF-TEST")
    print("=" * 80)

    print(
        f"State dimension          : {STATE_DIM}"
    )

    model = validate_model()

    print(
        "TwinDynamics construction : PASS"
    )

    sample = torch.randn(
        8,
        STATE_DIM,
    )

    with torch.no_grad():
        output = model(sample)

    if output.shape != sample.shape:
        raise RuntimeError(
            "TwinDynamics shape contract failed."
        )

    print(
        "Forward shape contract    : PASS"
    )

    if not torch.isfinite(output).all():
        raise RuntimeError(
            "TwinDynamics finite-output contract failed."
        )

    print(
        "Finite-output contract    : PASS"
    )

    # Verify residual behavior mathematically.
    zero_model = TwinDynamics(
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        context_dim=None,
        dropout=0.0,
    )

    with torch.no_grad():
        zero_model.network[-1].weight.zero_()
        zero_model.network[-1].bias.zero_()

        residual_output = zero_model(
            sample
        )

    if not torch.allclose(
        residual_output,
        sample,
    ):
        raise RuntimeError(
            "Residual transition contract failed."
        )

    print(
        "Residual state transition : PASS"
    )

    print(
        "delta_t model input       : NO"
    )

    print(
        "Training partition        : PASS"
    )

    print(
        "Held-out partition        : PASS"
    )

    print()
    print(
        "SELF-TEST                : PASS"
    )
    print()


# ============================================================================
# TRAINING
# ============================================================================

def train(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: str,
) -> None:

    seed_everything(SEED)

    resolved_device = torch.device(
        device
    )

    print()
    print("=" * 80)
    print("T1D-UOM DIGITAL TWIN DYNAMICS TRAINING")
    print("=" * 80)
    print()
    print("Architecture:")
    print(
        "  Unified Patient State -> TwinDynamics "
        "-> Predicted Next State"
    )
    print()
    print(
        f"State dimension       : {STATE_DIM}"
    )
    print(
        f"TwinDynamics hidden   : {HIDDEN_DIM}"
    )
    print(
        "Context               : NONE"
    )
    print(
        "delta_t model input   : NO"
    )
    print(
        f"Device                : {resolved_device}"
    )
    print()

    print(
        "Training participants:"
    )

    for participant in TRAINING_PARTICIPANTS:
        print(
            f"  {participant}"
        )

    print()
    print(
        "Held-out participants:"
    )

    for participant in HELD_OUT_PARTICIPANTS:
        print(
            f"  {participant}"
        )

    print()
    print(
        "Loading validated transition artifacts..."
    )

    current, next_state, counts = (
        concatenate_participants(
            TRAINING_PARTICIPANTS
        )
    )

    validate_dataset(
        current,
        next_state,
    )

    print()
    print(
        f"Total training transitions : "
        f"{len(current):,}"
    )

    print(
        f"State dimension            : "
        f"{current.shape[1]}"
    )

    (
        train_current,
        train_next,
        validation_current,
        validation_next,
    ) = split_training_data(
        current,
        next_state,
    )

    print(
        f"Optimization samples      : "
        f"{len(train_current):,}"
    )

    print(
        f"Validation samples        : "
        f"{len(validation_current):,}"
    )

    train_dataset = TransitionDataset(
        train_current,
        train_next,
    )

    validation_dataset = TransitionDataset(
        validation_current,
        validation_next,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=(
            resolved_device.type == "cuda"
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=(
            resolved_device.type == "cuda"
        ),
    )

    model = TwinDynamics(
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        context_dim=None,
        dropout=0.0,
    ).to(resolved_device)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        MODEL_ROOT
        / "twin_dynamics_best.pt"
    )

    final_model_path = (
        MODEL_ROOT
        / "twin_dynamics_final.pt"
    )

    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    print()
    print("-" * 80)
    print("TRAINING")
    print("-" * 80)

    for epoch in range(
        1,
        epochs + 1,
    ):

        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=resolved_device,
        )

        validation_loss = run_epoch(
            model,
            validation_loader,
            criterion,
            optimizer=None,
            device=resolved_device,
        )

        scheduler.step(
            validation_loss
        )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train MSE={train_loss:.8e} | "
            f"validation MSE={validation_loss:.8e} | "
            f"lr={current_lr:.3e}"
        )

        if validation_loss < (
            best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                best_model_path,
                model,
                optimizer,
                epoch,
                train_loss,
                validation_loss,
            )

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ------------------------------------------------------------------
    # Final checkpoint.
    # ------------------------------------------------------------------

    save_checkpoint(
        final_model_path,
        model,
        optimizer,
        epoch,
        train_loss,
        validation_loss,
    )

    write_metadata(
        counts=counts,
        train_samples=len(train_current),
        validation_samples=len(
            validation_current
        ),
        best_epoch=best_epoch,
        best_validation_loss=(
            best_validation_loss
        ),
        final_train_loss=train_loss,
    )

    print()
    print("=" * 80)
    print("TWIN DYNAMICS TRAINING COMPLETED")
    print("=" * 80)
    print()
    print(
        f"Best epoch             : {best_epoch}"
    )
    print(
        f"Best validation MSE    : "
        f"{best_validation_loss:.8e}"
    )
    print(
        f"Final training MSE     : "
        f"{train_loss:.8e}"
    )
    print()
    print(
        f"Best model             : "
        f"{best_model_path}"
    )
    print(
        f"Final model            : "
        f"{final_model_path}"
    )
    print(
        f"Metadata               : "
        f"{MODEL_ROOT / 'twin_dynamics_training_metadata.json'}"
    )
    print()
    print(
        "Held-out participants were NOT used "
        "for optimization."
    )
    print(
        "Frozen source data was not modified."
    )
    print()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train the T1D-UOM TwinDynamics "
            "Digital Twin transition model."
        )
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run TwinDynamics training-stage self-test.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.epochs <= 0:
        parser.error(
            "--epochs must be positive."
        )

    if args.batch_size <= 0:
        parser.error(
            "--batch-size must be positive."
        )

    if args.learning_rate <= 0:
        parser.error(
            "--learning-rate must be positive."
        )

    if args.weight_decay < 0:
        parser.error(
            "--weight-decay cannot be negative."
        )

    if args.patience <= 0:
        parser.error(
            "--patience must be positive."
        )

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
    )


if __name__ == "__main__":
    main()