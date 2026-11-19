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
    ids, catalogue = _catalogue(candidate_ids, candidate_power, grid.size)
    if not np.isfinite(catalogue).all() or np.any(catalogue <= 0):
        raise ValueError("candidate_power must be finite and strictly positive")
    shape = (observed.start_samples.size, grid.size)
    if observed.power.shape != shape or observed.valid.shape != (shape[0],):
        raise ValueError("observed power and validity must match the window and frequency axes")

    observed_db = np.full(shape, np.nan)
    positive = (observed.power > 0) & np.isfinite(observed.power)
    np.log10(observed.power, out=observed_db, where=positive)
    return _fit_db(10 * observed_db, 10 * np.log10(catalogue), ids, observed.valid)


def fit_eseis_catalogue(
    observed_power_db: ArrayLike,
    *,
    candidate_ids: Sequence[str],
    candidate_power_db: ArrayLike,
) -> FitResult:
    """Match already aligned spectra using the eseis unweighted dB RMSE.

    Supply observations in (window, frequency) order and references in
    (candidate, frequency) order, with identical frequencies in each column.
    The first candidate wins an exact tie. Windows with nonfinite values are
    skipped. Bin residuals are absolute differences in dB.
    """
    if np.iscomplexobj(observed_power_db):
        raise ValueError("observed_power_db must be real")
    observed = np.ma.asarray(observed_power_db, dtype=np.float64).filled(np.nan)
    if observed.ndim != 2 or observed.shape[1] == 0:
        raise ValueError("observed_power_db must have shape (window, frequency)")
    ids, catalogue = _catalogue(candidate_ids, candidate_power_db, observed.shape[1])
    if not np.isfinite(catalogue).all():
        raise ValueError("candidate_power_db must be finite")
    return _fit_db(observed, catalogue, ids, np.ones(observed.shape[0], dtype=bool))


def _catalogue(
    candidate_ids: Sequence[str], candidate_power: ArrayLike, frequency_count: int
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
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
    if catalogue.shape != (len(ids), frequency_count):
        raise ValueError("candidate_power must have shape (candidate, frequency)")
    return ids, catalogue


def _fit_db(
    observed_db: NDArray[np.float64],
    reference_db: NDArray[np.float64],
    ids: tuple[str, ...],
    valid: NDArray[np.bool_],
) -> FitResult:
    window_count = observed_db.shape[0]
    selected: list[str | None] = [None] * window_count
    rmse = np.full(window_count, np.nan)
    margin = np.full(window_count, np.nan)
    residual = np.full(observed_db.shape, np.nan)
    for row in np.flatnonzero(valid & np.isfinite(observed_db).all(axis=1)):
        differences = reference_db - observed_db[row]
        scores = np.sqrt(np.mean(differences**2, axis=1))
        best = int(np.argmin(scores))
        selected[row] = ids[best]
        rmse[row] = scores[best]
        residual[row] = np.abs(differences[best])
        margin[row] = np.partition(scores, 1)[1] - scores[best] if len(ids) > 1 else np.inf
    return FitResult(tuple(selected), rmse, residual, margin)
