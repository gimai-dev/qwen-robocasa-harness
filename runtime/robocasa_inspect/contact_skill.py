"""Reusable public-proprioception contact confirmation for bounded push skills."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PUBLIC_CONTACT_STALL_RATIO = 0.15
PUBLIC_CONTACT_ENTRY_TOLERANCE_M = 0.005
REQUIRED_CONTACT_CONFIRMATIONS = 2


@dataclass(frozen=True)
class PublicContactEvidence:
    progress_ratio: float
    eligible: bool
    stalled: bool
    confirmations: int
    confirmed: bool


def confirm_public_motion_contact(
    *,
    realized_m: float,
    requested_m: float,
    realized_forward_m: float,
    contact_entry_m: float,
    previous_confirmations: int,
) -> PublicContactEvidence:
    """Confirm contact from repeated motion stall after a public approach gate.

    This consumes only commanded motion and public end-effector displacement. It
    deliberately does not read simulator contacts or task state.
    """

    values = np.asarray(
        [realized_m, requested_m, realized_forward_m, contact_entry_m],
        dtype=np.float64,
    )
    if (
        not np.isfinite(values).all()
        or realized_m < 0.0
        or requested_m <= 0.0
        or contact_entry_m < 0.0
        or isinstance(previous_confirmations, bool)
        or not isinstance(previous_confirmations, int)
        or not 0 <= previous_confirmations <= REQUIRED_CONTACT_CONFIRMATIONS
    ):
        raise ValueError("public contact evidence is invalid")
    ratio = float(realized_m / requested_m)
    eligible = (
        realized_forward_m
        >= contact_entry_m - PUBLIC_CONTACT_ENTRY_TOLERANCE_M
    )
    stalled = eligible and ratio <= PUBLIC_CONTACT_STALL_RATIO
    confirmations = (
        min(REQUIRED_CONTACT_CONFIRMATIONS, previous_confirmations + 1)
        if stalled
        else 0
    )
    return PublicContactEvidence(
        progress_ratio=ratio,
        eligible=eligible,
        stalled=stalled,
        confirmations=confirmations,
        confirmed=confirmations >= REQUIRED_CONTACT_CONFIRMATIONS,
    )
