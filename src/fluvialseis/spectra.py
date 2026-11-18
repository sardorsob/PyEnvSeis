"""One-sided velocity power spectral densities on physical FFT frequency grids."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal

Detrend = Literal["constant", "linear", False]


@dataclass(frozen=True)
class Spectrum:
    """Frequency in Hz and linear velocity PSD in (m/s)²/Hz."""

    frequency_hz: NDArray[np.float64]
    power: NDArray[np.float64]


@dataclass(frozen=True)
class Spectrogram:
    """Linear velocity PSD in (window, frequency) order with NaN rows for gaps."""

    frequency_hz: NDArray[np.float64]
    start_samples: NDArray[np.int64]
    power: NDArray[np.float64]
    valid: NDArray[np.bool_]


def _samples(data: ArrayLike) -> NDArray[np.float64]:
    if np.iscomplexobj(data):
        raise ValueError("samples_m_s must be real")
    samples = np.ma.asarray(data, dtype=np.float64).filled(np.nan)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("samples_m_s must be a one-dimensional array with at least two samples")
    return samples


def _analysis_window(
    size: int, sample_rate_hz: float, window: str, detrend: Detrend
) -> NDArray[np.float64]:
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, Real)
        or not np.isfinite(sample_rate_hz)
        or sample_rate_hz <= 0
    ):
        raise ValueError("sample_rate_hz must be finite and positive")
    if detrend is not False and detrend not in ("constant", "linear"):
        raise ValueError("detrend must be False, 'constant', or 'linear'")
    return signal.get_window(window, size)


def _estimate(
    samples: NDArray[np.float64],
    sample_rate_hz: float,
    window: NDArray[np.float64],
    detrend: Detrend,
) -> Spectrum:
    frequency, power = signal.periodogram(
        samples, fs=sample_rate_hz, window=window, detrend=detrend, scaling="density"
    )
    if not np.isfinite(power).all() or not np.isfinite(frequency).all():
        raise ValueError("PSD calculation produced nonfinite frequencies or power")
    return Spectrum(frequency, power)


def periodogram(
    samples_m_s: ArrayLike,
    *,
    sample_rate_hz: float,
    window: str = "hann",
    detrend: Detrend = "constant",
) -> Spectrum:
    """Estimate one-sided PSD from uniformly sampled velocity in m/s."""
    samples = _samples(samples_m_s)
    weights = _analysis_window(samples.size, sample_rate_hz, window, detrend)
    if not np.isfinite(samples).all():
        raise ValueError("samples_m_s must be finite for a single periodogram")
    return _estimate(samples, sample_rate_hz, weights, detrend)


def windowed_psd(
    samples_m_s: ArrayLike,
    *,
    sample_rate_hz: float,
    window_samples: int,
    step_samples: int,
    window: str = "hann",
    detrend: Detrend = "constant",
) -> Spectrogram:
    """Estimate PSDs of complete windows, marking windows with missing samples."""
    samples = _samples(samples_m_s)
    if (
        isinstance(window_samples, bool)
        or not isinstance(window_samples, Integral)
        or not 2 <= window_samples <= samples.size
    ):
        raise ValueError("window_samples must be an integer between 2 and the sample count")
    if (
        isinstance(step_samples, bool)
        or not isinstance(step_samples, Integral)
        or not 1 <= step_samples <= window_samples
    ):
        raise ValueError("step_samples must be an integer between 1 and window_samples")
    weights = _analysis_window(window_samples, sample_rate_hz, window, detrend)
    frequency_hz = np.fft.rfftfreq(window_samples, d=1 / sample_rate_hz)
    starts = np.arange(0, samples.size - window_samples + 1, step_samples, dtype=np.int64)
    power = np.full((starts.size, frequency_hz.size), np.nan)
    valid = np.zeros(starts.size, dtype=np.bool_)
    for row, start in enumerate(starts):
        segment = samples[start : start + window_samples]
        if np.isfinite(segment).all():
            power[row] = _estimate(segment, sample_rate_hz, weights, detrend).power
            valid[row] = True
    return Spectrogram(frequency_hz, starts, power, valid)


def _frequency_grid(frequencies: ArrayLike) -> NDArray[np.float64]:
    if np.iscomplexobj(frequencies) or np.ma.is_masked(frequencies):
        raise ValueError("frequency_hz must contain real, unmasked values")
    grid = np.asarray(frequencies, dtype=np.float64)
    if (
        grid.ndim != 1
        or grid.size == 0
        or not np.isfinite(grid).all()
        or grid[0] < 0
        or np.any(grid[1:] <= grid[:-1])
    ):
        raise ValueError("frequency_hz must be finite, nonnegative and strictly increasing")
    return grid


def regrid_spectrogram(spectrogram: Spectrogram, frequency_hz: ArrayLike) -> Spectrogram:
    """Linearly interpolate power within the observed band, preserving gap rows."""
    source = _frequency_grid(spectrogram.frequency_hz)
    target = _frequency_grid(frequency_hz)
    if target[0] < source[0] or target[-1] > source[-1]:
        raise ValueError("target frequency_hz must lie within the observed frequency band")
    power = np.full((spectrogram.start_samples.size, target.size), np.nan)
    for row in np.flatnonzero(spectrogram.valid):
        power[row] = np.interp(target, source, spectrogram.power[row])
    return Spectrogram(
        target.copy(), spectrogram.start_samples.copy(), power, spectrogram.valid.copy()
    )
