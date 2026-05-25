"""PyTorch sequence-model classifiers for ICU outcome prediction.

Unit U2 of Phase 5 (plan:
``docs/plans/2026-05-25-001-feat-phase5-sequence-model-plan.md``).

This module exposes the LSTM baseline classifier that consumes the tensors
produced by :mod:`icumodelstream.sequences`. The transformer variant is
intentionally deferred per the plan's "Alternative Approaches Considered"
section; a stub class is provided so callers and tests can refer to the
symbol without being able to silently fall back to an unimplemented model.

CLAUDE.md rule 9 (baselines before deep learning): the LSTM is the FIRST
sequence model in the project, and it sits AFTER the LightGBM / logistic
flat-feature baselines have been implemented and reproduced. CLAUDE.md
rule 2 (simplicity first): one LSTM module + one frozen result dataclass,
no early-stopping logic, no learning-rate schedulers, no AMP wrappers --
those live in the training module (U3). CLAUDE.md rule 8 (no silent
patient leakage): this module is pure ``nn.Module`` arithmetic on already
shape-validated tensors; the leakage-relevant work (per-patient splitting,
half-open windowing) is upstream in :mod:`icumodelstream.sequences` and
:mod:`icumodelstream.splits`.

The forward pass returns RAW LOGITS (no sigmoid). The training loop in U3
will wrap predictions with :class:`torch.nn.BCEWithLogitsLoss` which
applies the sigmoid in a numerically stable way; inference-time probability
conversion (``torch.sigmoid(logits)``) belongs at the call site, not in the
model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch import nn


@dataclass(frozen=True)
class SequenceResult:
    """Container for a fitted sequence model's test-set predictions and metrics.

    Mirrors :class:`icumodelstream.models.BaselineResult` field-for-field
    where the two overlap, so the CLI's JSON / Markdown writers can serialize
    either type without branching. The two additional fields
    (``epochs_trained`` and ``early_stopped_at_epoch``) carry sequence-model
    training metadata that the flat baselines do not need.

    Attributes
    ----------
    model_name:
        Either ``"lstm"`` or ``"transformer"``. Lets downstream code branch
        without inspecting the model object.
    y_true:
        Ground-truth labels on the test set, shape ``(n_test,)``, int 0/1.
    y_pred_proba:
        Predicted probabilities of the positive class, shape ``(n_test,)``,
        floats in ``[0, 1]``.
    metrics:
        Dict with six keys: ``auroc``, ``auprc``, ``brier_score``,
        ``prevalence``, ``calibration_intercept``, ``calibration_slope``.
        Same key set as :class:`BaselineResult` so dashboards aggregate
        cleanly across model families.
    calibration_table:
        Polars DataFrame with fixed-decile bins matching the layout
        produced by :mod:`icumodelstream.models` (``bin``, ``mean_pred``,
        ``mean_actual``, ``count``).
    epochs_trained:
        Total number of epochs the optimizer actually ran.
    early_stopped_at_epoch:
        1-based index of the epoch at which validation loss stopped
        improving, or ``None`` if the training loop ran to ``max_epochs``
        without triggering early stopping.
    """

    model_name: str
    y_true: np.ndarray
    y_pred_proba: np.ndarray
    metrics: dict[str, float]
    calibration_table: pl.DataFrame
    epochs_trained: int
    early_stopped_at_epoch: int | None


class LSTMBaseline(nn.Module):
    """Two-layer LSTM classifier over per-hospitalization sequence tensors.

    Architecture:

    .. code-block:: text

        (batch, timesteps, input_dim)
              |
              v
        nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                num_layers=n_layers, batch_first=True,
                dropout=dropout, bidirectional=bidirectional)
              |
              v   take output of LAST timestep
              v   shape: (batch, hidden_dim * (2 if bidirectional else 1))
              |
        nn.Dropout(dropout)   # nn.LSTM only applies dropout BETWEEN layers
              |
              v
        nn.Linear(hidden_dim * D, 1)   # D = 2 if bidirectional else 1
              |
              v
        .squeeze(-1)  ->  (batch,) RAW LOGITS

    The explicit :class:`nn.Dropout` between the LSTM output and the linear
    head is the standard fix for ``nn.LSTM``'s "dropout applies between
    layers, never before the head" behavior.

    No custom weight initialization: PyTorch's default LSTM init (uniform
    over ``[-1/sqrt(hidden), 1/sqrt(hidden)]``) is the standard starting
    point and is good enough for a baseline; deviating here would be
    premature optimization per CLAUDE.md rule 2.

    Parameters
    ----------
    input_dim:
        Number of channels per timestep (must match
        ``SequenceTensors.X.shape[-1]``).
    hidden_dim:
        LSTM hidden state size. 128 is a sensible baseline for ~30 channels
        and ~24 timesteps -- big enough to fit, small enough to train on
        CPU/MPS.
    n_layers:
        Number of stacked LSTM layers. With ``dropout > 0`` and
        ``n_layers >= 2``, dropout is applied between layers.
    dropout:
        Dropout probability used both inside the LSTM (between layers) and
        just before the linear head.
    bidirectional:
        If True, the LSTM reads the sequence in both directions and the
        final hidden representation has size ``2 * hidden_dim``. The linear
        head is sized accordingly.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # nn.LSTM emits a UserWarning when num_layers == 1 AND dropout > 0
        # because there are no inter-layer gaps to apply dropout to. Pass 0.0
        # in that case rather than silently letting torch warn on every call.
        lstm_dropout = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        directions = 2 if bidirectional else 1
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * directions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of sequence tensors.

        Parameters
        ----------
        x:
            Float tensor of shape ``(batch, timesteps, input_dim)``. This is
            the layout that :func:`icumodelstream.sequences.build_sequence_tensors`
            produces in ``SequenceTensors.X``.

        Returns
        -------
        torch.Tensor
            1-D tensor of RAW LOGITS, shape ``(batch,)``. The loss function
            (``BCEWithLogitsLoss``) applies the sigmoid for numerical
            stability; convert to probabilities at the call site with
            ``torch.sigmoid(logits)``.
        """
        # outputs: (batch, timesteps, hidden_dim * directions)
        # We discard (h_n, c_n) and take outputs[:, -1, :] so the same code
        # path works regardless of whether the LSTM is unidirectional or
        # bidirectional. For a bidirectional LSTM the forward direction's
        # final hidden AND the backward direction's hidden at t=0 are both
        # already concatenated in outputs[:, -1, :], so this is the correct
        # representation in both cases.
        outputs, _ = self.lstm(x)
        last = outputs[:, -1, :]
        last = self.head_dropout(last)
        logits = self.classifier(last)
        return logits.squeeze(-1)


class TransformerBaseline(nn.Module):
    """Deferred transformer sequence classifier.

    The forward signature mirrors :class:`LSTMBaseline` so a future
    implementer can swap models without rewiring callers. The constructor
    raises immediately to prevent silent fallback or partially-trained
    "transformer" runs that are actually default-initialized noise.

    See the plan's "Alternative Approaches Considered" section for the
    rationale: LSTM is characterized first because its training dynamics are
    well-understood on small clinical sequence cohorts, and the transformer
    follow-up is gated on having LSTM curves and calibration baselines to
    compare against.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "TransformerBaseline is intentionally deferred per "
            "docs/plans/2026-05-25-001-feat-phase5-sequence-model-plan.md "
            "(Alternative Approaches Considered). Implement after LSTM "
            "characterization is complete."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - stub
        """Same shape contract as :meth:`LSTMBaseline.forward`."""
        raise NotImplementedError
