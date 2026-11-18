"""Compare observations with caller-supplied spectra in the same power convention."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .spectra import Spectrogram, _frequency_grid


@dataclass(frozen=True)
class FitResult:
    """Winning ID, RMSE, bin residuals and runner-up margin for each window."""

    candidate_ids: tuple[str | None, ...]
    rmse_db: NDArray[np.float64]
    residual_db: NDArray[np.float64]
    margin_db: NDArray[np.float64]


def fit_catalogue(
    observed: Spectrogram,
    *,
    frequency_hz: ArrayLike,
    candidate_ids: Sequence[str],
    candidate_power: ArrayLike,
) -> FitResult:
    """Find the minimum dB RMSE on a shared, explicitly supplied frequency grid.

    Supply linear, positive catalogue PSDs with rows in candidate order and
    units matching the observations. The first row wins an exact tie.
    Windows with gaps or nonpositive power have no winning ID and NaN scores.
    A catalogue with one candidate has an infinite runner-up margin.
    """
    grid = _frequency_grid(frequency_hz)
    if not np.array_equal(grid, observed.frequency_hz):
        raise ValueError("catalogue and observed frequency grids must match")
    if isinstance(candidate_ids, str):
        raise ValueError("candidate_ids must be a sequence of unique, non-empty strings")
    ids = tuple(candidate_ids)
    if not ids or any(not isinstance(name, str) or not name for name in ids):
        raise ValueError("candidate_ids must be a sequence of unique, non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_ids must be unique")
    if np.iscomplexobj(candidate_power) or np.ma.is_masked(candidate_power):
        raise ValueError("candidate_power must be real and unmasked")
    catalogue = np.asarray(candidate_power, dtype=np.float64)
    if catalogue.shape != (len(ids), grid.size):
        raise ValueError("candidate_power must have shape (candidate, frequency)")
    if not np.isfinite(catalogue).all() or np.any(catalogue <= 0):
        raise ValueError("candidate_power must be finite and strictly positive")
    shape = (observed.start_samples.size, grid.size)
    if observed.power.shape != shape or observed.valid.shape != (shape[0],):
        raise ValueError("observed power and validity must match the window and frequency axes")

    reference_db = 10 * np.log10(catalogue)
    selected: list[str | None] = [None] * shape[0]
    rmse = np.full(shape[0], np.nan)
    margin = np.full(shape[0], np.nan)
    residual = np.full(shape, np.nan)
    for row in np.flatnonzero(observed.valid):
        power = observed.power[row]
        if np.any(power <= 0) or not np.isfinite(power).all():
            continue
        differences = reference_db - 10 * np.log10(power)
        scores = np.sqrt(np.mean(differences**2, axis=1))
        best = int(np.argmin(scores))
        selected[row] = ids[best]
        rmse[row] = scores[best]
        residual[row] = np.abs(differences[best])
        margin[row] = np.partition(scores, 1)[1] - scores[best] if len(ids) > 1 else np.inf
    return FitResult(tuple(selected), rmse, residual, margin)
