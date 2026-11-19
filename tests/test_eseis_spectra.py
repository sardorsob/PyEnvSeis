import numpy as np
import pytest

from fluvialseis.eseis_spectra import (
    EseisSpectrogram,
    clip_eseis_spectrogram,
    eseis_spectrogram,
    eseis_spectrum,
)


def test_eseis_spectrum_pads_odd_record_and_retains_positive_fft_bins():
    # A centered impulse has no slope. Removing its mean and halving the two
    # end samples gives the analytic vector for a 13-sample, 10% cosine taper.
    samples = np.zeros(13)
    samples[6] = 1.0
    tapered = np.array([-0.5, -1, -1, -1, -1, -1, 12, -1, -1, -1, -1, -1, -0.5]) / 13
    positive_bins = np.arange(1, 8)
    phases = -2j * np.pi * np.outer(positive_bins, np.arange(13)) / 15
    expected_power = abs(np.exp(phases) @ tapered) ** 2 / (13 * 0.875)

    result = eseis_spectrum(samples + 2 + 0.25 * np.arange(13), sample_rate_hz=30)

    np.testing.assert_array_equal(result.fft_frequency_hz, [2, 4, 6, 8, 10, 12, 14])
    np.testing.assert_array_equal(result.frequency_hz, [0, 2.5, 5, 7.5, 10, 12.5, 15])
    np.testing.assert_allclose(result.power, expected_power, rtol=1e-13)


@pytest.mark.parametrize(
    ("width", "slices"),
    [
        (2, [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]),
        (4, [(0, 3), (0, 4), (1, 5), (2, 6), (3, 6), (4, 6)]),
    ],
)
def test_eseis_spectrum_even_smoothing_aligns_and_shrinks_at_edges(width, slices):
    samples = [0, 1, 3, 2, 1, -2, 1, 3, 1, 0, -3]
    raw = eseis_spectrum(samples, sample_rate_hz=24)
    smoothed = eseis_spectrum(samples, sample_rate_hz=24, smoothing_bins=width)

    np.testing.assert_array_equal(smoothed.frequency_hz, raw.frequency_hz)
    np.testing.assert_allclose(smoothed.power, [raw.power[a:b].mean() for a, b in slices])


def test_eseis_spectrum_smoothing_preserves_weak_bins_after_a_strong_peak():
    samples = 1e-6 * np.sin(2 * np.pi * 8 * np.arange(8192) / 256)
    raw = eseis_spectrum(samples, sample_rate_hz=256)

    smoothed = eseis_spectrum(samples, sample_rate_hz=256, smoothing_bins=4)

    expected = np.array([raw.power[max(0, i - 1) : i + 3].mean() for i in range(raw.power.size)])
    assert np.all(smoothed.power > 0)
    np.testing.assert_allclose(smoothed.power, expected, rtol=1e-15, atol=0)


def test_eseis_spectrum_resamples_smoothed_linear_power():
    samples = [0, 1, 3, 2, 1, -2, 1, 3, 1, 0, -3]
    smoothed = eseis_spectrum(samples, sample_rate_hz=20, smoothing_bins=4)
    interpolated = eseis_spectrum(samples, sample_rate_hz=20, smoothing_bins=4, resolution=11)

    assert interpolated.fft_frequency_hz is None
    np.testing.assert_array_equal(interpolated.frequency_hz, np.arange(11))
    np.testing.assert_allclose(interpolated.power[::2], smoothed.power)


def test_eseis_spectrum_log_grid_starts_at_first_positive_displayed_frequency():
    samples = [0, 1, 3, 2, 1, -2, 1, 3, 1, 0, -3]

    result = eseis_spectrum(samples, sample_rate_hz=20, resolution=3, log_frequency=True)

    np.testing.assert_allclose(result.frequency_hz, [2, np.sqrt(20), 10])


def test_eseis_spectrogram_omits_exact_terminal_window_and_labels_earlier_starts():
    samples = np.random.default_rng(0).uniform(-1, 1, size=26)
    expected = eseis_spectrum(samples[:13], sample_rate_hz=32)

    result = eseis_spectrogram(
        samples, sample_rate_hz=32, window_seconds=13 / 32, overlap=0, start_time_seconds=100
    )

    np.testing.assert_array_equal(result.start_samples, [0])
    np.testing.assert_array_equal(result.time_seconds, [100])
    np.testing.assert_array_equal(result.frequency_hz, expected.frequency_hz)
    np.testing.assert_array_equal(result.fft_frequency_hz, expected.fft_frequency_hz)
    np.testing.assert_array_equal(result.valid, [True])
    np.testing.assert_allclose(result.power_db, [10 * np.log10(expected.power)])


def test_eseis_spectrogram_preserves_half_open_gap_mask_through_clip():
    samples = np.random.default_rng(0).uniform(-1, 1, size=25)
    samples[8] = np.nan
    result = eseis_spectrogram(
        samples, sample_rate_hz=8, window_seconds=1, overlap=0.5, start_time_seconds=100
    )
    clipped = clip_eseis_spectrogram(result, frequency_range_hz=(1, 3), resolution=5)

    np.testing.assert_array_equal(result.start_samples, [0, 4, 8, 12, 16])
    np.testing.assert_array_equal(result.time_seconds, [100, 100.5, 101, 101.5, 102])
    np.testing.assert_array_equal(result.valid, [True, False, False, True, True])
    np.testing.assert_array_equal(clipped.valid, result.valid)
    assert np.isnan(result.power_db[~result.valid]).all()
    assert np.isnan(clipped.power_db[~clipped.valid]).all()
    assert np.isfinite(clipped.power_db[clipped.valid]).all()


def test_eseis_clip_uses_fmm_boundary_cubics_in_db_including_right_extrapolation():
    frequencies = np.arange(5, dtype=float)
    spectrogram = EseisSpectrogram(
        frequency_hz=frequencies,
        start_samples=np.array([0]),
        time_seconds=np.array([100.0]),
        power_db=frequencies[None, :] ** 4,
        valid=np.array([True]),
        fft_frequency_hz=None,
    )

    clipped = clip_eseis_spectrogram(spectrogram, frequency_range_hz=(0.5, 4.5), resolution=5)

    # For knots y = x**4 at x = 0, ..., 4, the end cubics have third
    # derivatives 36 and 60. Solving the five continuity equations yields
    # these midpoint and extrapolation values.
    np.testing.assert_array_equal(clipped.frequency_hz, [0.5, 1.5, 2.5, 3.5, 4.5])
    np.testing.assert_allclose(clipped.power_db, [[5 / 6, 29 / 6, 233 / 6, 905 / 6, 404]])
    np.testing.assert_array_equal(clipped.time_seconds, spectrogram.time_seconds)
    np.testing.assert_array_equal(clipped.start_samples, spectrogram.start_samples)
    assert clipped.fft_frequency_hz is None


def test_eseis_zero_window_preserves_db_zero_before_clip_mask():
    result = eseis_spectrogram(np.zeros(17), sample_rate_hz=8, window_seconds=1, overlap=0)

    assert np.isneginf(result.power_db).all()
    np.testing.assert_array_equal(result.valid, [True, True])

    clipped = clip_eseis_spectrogram(result, frequency_range_hz=(1, 3), resolution=5)
    np.testing.assert_array_equal(clipped.valid, [False, False])
    assert np.isnan(clipped.power_db).all()


@pytest.mark.parametrize(
    ("starts", "times"),
    [([[0, 4]], [100.0, 100.5]), ([0, 4], [100.0]), ([0, 4], [[100.0, 100.5]])],
)
def test_eseis_clip_rejects_axes_that_do_not_match_window_rows(starts, times):
    spectrogram = EseisSpectrogram(
        frequency_hz=np.array([0.0, 1.0]),
        start_samples=np.array(starts),
        time_seconds=np.array(times),
        power_db=np.array([[-100.0, -110.0], [-102.0, -120.0]]),
        valid=np.array([True, True]),
    )

    with pytest.raises(ValueError, match="time and frequency axes"):
        clip_eseis_spectrogram(spectrogram)


@pytest.mark.parametrize("options", [{"window_seconds": 2}, {"overlap": 0.98}])
def test_eseis_rejects_empty_window_grid_or_rounded_zero_step(options):
    settings = {"sample_rate_hz": 8, "window_seconds": 1, **options}
    with pytest.raises(ValueError):
        eseis_spectrogram(np.ones(16), **settings)
