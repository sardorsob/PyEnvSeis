"""Default periodogram conventions of eseis 0.8.1 and the fluvial notebook.

Numeric input to R's stats::spec.pgram has unit sample frequency. The
linear scale is squared velocity per cycle/sample, with no one-sided fold
factor. The eseis labels span 0 to Nyquist even though the first retained
FFT bin is positive. Use the physical PSD functions for density per Hz.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fftpack import next_fast_len
from scipy.linalg import solve_banded

from .spectra import _frequency_grid, _samples

_TAPER_FRACTION = 0.1
_TAPER_CORRECTION = 0.875


@dataclass(frozen=True)
class EseisSpectrum:
    """Linear periodogram power on the eseis frequency labels.

    ``fft_frequency_hz`` identifies the transform bins before interpolation.
    """

    frequency_hz: NDArray[np.float64]
    power: NDArray[np.float64]
    fft_frequency_hz: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class EseisSpectrogram:
    """Legacy dB power in (window, frequency) order.

    Time labels identify the first sample of each window. The dB reference
    is one unit of legacy linear power. A clipped grid has no FFT bin axis.
    """

    frequency_hz: NDArray[np.float64]
    start_samples: NDArray[np.int64]
    time_seconds: NDArray[np.float64]
    power_db: NDArray[np.float64]
    valid: NDArray[np.bool_]
    fft_frequency_hz: NDArray[np.float64] | None = None


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _spectral_grid(
    fft_size: int, sample_rate_hz: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    fft_frequency = np.arange(1, fft_size // 2 + 1, dtype=np.float64) * (sample_rate_hz / fft_size)
    frequency = np.linspace(0, sample_rate_hz / 2, fft_frequency.size)
    return frequency, fft_frequency


def _taper(size: int) -> NDArray[np.float64]:
    edge_size = int(size * _TAPER_FRACTION)
    weights = np.ones(size)
    if edge_size:
        phase = np.pi * (np.arange(edge_size) + 0.5) / edge_size
        edge = 0.5 * (1 - np.cos(phase))
        weights[:edge_size] = edge
        weights[-edge_size:] = edge[::-1]
    return weights


def _pgram(
    samples: NDArray[np.float64], taper: NDArray[np.float64], fft_size: int
) -> NDArray[np.float64]:
    size = samples.size
    index = np.arange(size) - (size - 1) / 2
    slope = np.dot(samples, index) / (size * (size**2 - 1) / 12)
    detrended = samples - samples.mean() - slope * index
    amplitudes = np.fft.rfft(taper * detrended, n=fft_size)[1:]
    power = (amplitudes.real**2 + amplitudes.imag**2) / (size * _TAPER_CORRECTION)
    if not np.isfinite(power).all():
        raise ValueError("legacy periodogram produced nonfinite power")
    return power


def _running_mean(power: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    # caTools::runmean uses smaller windows at the edges and right-aligns k=2.
    if width == 1:
        return power
    right = 0 if width == 2 else width // 2
    left = width - 1 - right
    index = np.arange(power.size)
    first = np.maximum(0, index - left)
    last = np.minimum(power.size - 1, index + right)

    # A window spans at most two width-sized blocks. Add their partial sums
    # so a strong peak never cancels weak power in a later window.
    blocks = np.pad(power, (0, -power.size % width)).reshape(-1, width)
    prefix = np.cumsum(blocks, axis=1).ravel()
    suffix = np.cumsum(blocks[:, ::-1], axis=1)[:, ::-1].ravel()
    totals = np.where(first % width == 0, prefix[last], suffix[first])
    crosses_block = first // width != last // width
    totals[crosses_block] = suffix[first[crosses_block]] + prefix[last[crosses_block]]
    return totals / (last - first + 1)


def _fmm_spline(
    x: NDArray[np.float64], y: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Interpolate one row using the Forsythe–Malcolm–Moler end conditions."""
    if x.size == 1:
        return np.full(target.size, y[0])
    if x.size == 2:
        return y[0] + (target - x[0]) * ((y[1] - y[0]) / (x[1] - x[0]))

    widths = np.diff(x)
    slopes = np.diff(y) / widths
    rhs = np.zeros(x.size)
    rhs[1:-1] = 6 * np.diff(slopes)
    if x.size >= 4:
        # Six times each local third divided difference gives the desired
        # third derivative on the first and last spline intervals.
        second_differences = np.diff(slopes) / (x[2:] - x[:-2])
        rhs[0] = 6 * widths[0] * (second_differences[1] - second_differences[0]) / (x[3] - x[0])
        rhs[-1] = (
            6 * widths[-1] * (second_differences[-1] - second_differences[-2]) / (x[-1] - x[-4])
        )

    # Solve for knot second derivatives. The first and last equations
    # constrain their differences. Interior equations enforce smooth slopes.
    bands = np.zeros((3, x.size))
    bands[0, 2:] = widths[1:]
    bands[0, 1] = 1
    bands[1, 1:-1] = 2 * (widths[:-1] + widths[1:])
    bands[1, 0] = -1
    bands[1, -1] = 1
    bands[2, :-2] = widths[:-1]
    bands[2, -2] = -1
    curvature = solve_banded((1, 1), bands, rhs)

    interval = np.clip(np.searchsorted(x, target, side="right") - 1, 0, x.size - 2)
    h = widths[interval]
    b = (target - x[interval]) / h
    a = 1 - b
    cubic = (a**3 - a) * curvature[interval] + (b**3 - b) * curvature[interval + 1]
    return a * y[interval] + b * y[interval + 1] + h**2 * cubic / 6


def eseis_spectrum(
    samples_m_s: ArrayLike,
    *,
    sample_rate_hz: float,
    smoothing_bins: int | None = None,
    resolution: int | None = None,
    log_frequency: bool = False,
) -> EseisSpectrum:
    """Reproduce the standalone eseis default periodogram for finite samples.

    Optional running-mean smoothing operates on linear power before FMM
    spline interpolation. With ``resolution`` supplied, the grid is equally
    spaced by default or logarithmic from the first positive label when
    ``log_frequency=True``.
    """
    samples = _samples(samples_m_s)
    sample_rate_hz = _finite_real(sample_rate_hz, "sample_rate_hz")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if not np.isfinite(samples).all():
        raise ValueError("samples_m_s must be finite for a single periodogram")
    fft_size = next_fast_len(samples.size)
    frequency, fft_frequency = _spectral_grid(fft_size, sample_rate_hz)
    power = _pgram(samples, _taper(samples.size), fft_size)
    if smoothing_bins is not None:
        width = min(_positive_integer(smoothing_bins, "smoothing_bins"), power.size)
        power = _running_mean(power, width)
    if resolution is not None:
        count = _positive_integer(resolution, "resolution")
        if log_frequency:
            if frequency.size < 2:
                raise ValueError("a logarithmic grid requires a positive frequency label")
            target = np.exp(np.linspace(np.log(frequency[1]), np.log(frequency[-1]), count))
        else:
            target = np.linspace(frequency[0], frequency[-1], count)
        return EseisSpectrum(target, _fmm_spline(frequency, power, target))
    return EseisSpectrum(frequency, power, fft_frequency)


def eseis_spectrogram(
    samples_m_s: ArrayLike,
    *,
    sample_rate_hz: float,
    window_seconds: float,
    overlap: float = 0.5,
    start_time_seconds: float = 0.0,
) -> EseisSpectrogram:
    """Reproduce the eseis default periodogram spectrogram on a regular clock.

    An exactly terminal start is omitted per eseis. As a Python extension,
    windows containing nonfinite samples are reported as invalid NaN rows.
    """
    samples = _samples(samples_m_s)
    sample_rate_hz = _finite_real(sample_rate_hz, "sample_rate_hz")
    window_seconds = _finite_real(window_seconds, "window_seconds")
    overlap = _finite_real(overlap, "overlap")
    start_time_seconds = _finite_real(start_time_seconds, "start_time_seconds")
    if sample_rate_hz <= 0 or window_seconds <= 0:
        raise ValueError("sample_rate_hz and window_seconds must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    size_float = window_seconds * sample_rate_hz
    if not np.isfinite(size_float):
        raise ValueError("window length in samples must be finite")
    size = round(size_float)
    if not 2 <= size < samples.size:
        raise ValueError("rounded window length must be at least 2 and shorter than the record")
    step = size - round(size * overlap)
    if step == 0:
        raise ValueError("rounded overlap leaves no step between windows")

    starts = np.arange(0, samples.size - size, step, dtype=np.int64)
    fft_size = next_fast_len(size)
    frequency, fft_frequency = _spectral_grid(fft_size, sample_rate_hz)
    taper = _taper(size)
    power_db = np.full((starts.size, frequency.size), np.nan)
    valid = np.zeros(starts.size, dtype=np.bool_)
    for row, start in enumerate(starts):
        segment = samples[start : start + size]
        if np.isfinite(segment).all():
            valid[row] = True
            with np.errstate(divide="ignore"):
                power_db[row] = 10 * np.log10(_pgram(segment, taper, fft_size))
    time = start_time_seconds + starts.astype(np.float64) * (1 / sample_rate_hz)
    return EseisSpectrogram(frequency, starts, time, power_db, valid, fft_frequency)


def clip_eseis_spectrogram(
    spectrogram: EseisSpectrogram,
    *,
    frequency_range_hz: tuple[float, float] = (5.0, 120.0),
    resolution: int = 100,
) -> EseisSpectrogram:
    """Match the notebook's FMM interpolation in dB, including extrapolation.

    Gaps and rows containing nonfinite dB values become invalid NaN rows.
    Frequencies outside the input band use its end cubics, as in R.
    """
    source = _frequency_grid(spectrogram.frequency_hz)
    bounds = _frequency_grid(frequency_range_hz)
    if bounds.size != 2:
        raise ValueError("frequency_range_hz must contain two increasing frequency bounds")
    target = np.linspace(bounds[0], bounds[1], _positive_integer(resolution, "resolution", 2))

    n_windows = spectrogram.start_samples.size
    window_axis = (n_windows,)
    valid = spectrogram.valid
    power = spectrogram.power_db
    if (
        spectrogram.start_samples.shape != window_axis
        or spectrogram.time_seconds.shape != window_axis
        or valid.shape != window_axis
        or power.shape != (n_windows, source.size)
    ):
        raise ValueError("spectrogram arrays must match its time and frequency axes")
    valid = valid & np.isfinite(power).all(axis=1)
    interpolated = np.full((n_windows, target.size), np.nan)
    for row in np.flatnonzero(valid):
        interpolated[row] = _fmm_spline(source, power[row], target)
    return EseisSpectrogram(
        target,
        spectrogram.start_samples.copy(),
        spectrogram.time_seconds.copy(),
        interpolated,
        valid,
    )
