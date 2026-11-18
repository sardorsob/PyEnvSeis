import numpy as np
import pytest

from fluvialseis import periodogram, regrid_spectrogram, windowed_psd


def test_tone_power_and_amplitude_scaling():
    samples = 2 * np.sin(2 * np.pi * 32 * np.arange(1024) / 256)
    options = {"sample_rate_hz": 256, "window": "boxcar", "detrend": False}
    spec = periodogram(samples, **options)
    assert spec.frequency_hz[np.argmax(spec.power)] == 32
    assert spec.frequency_hz[1] == 0.25
    assert spec.power.max() == pytest.approx(8, abs=1e-12)
    assert spec.power.sum() * 0.25 == pytest.approx(2, abs=1e-12)
    doubled = periodogram(2 * samples, **options)
    assert 10 * np.log10(doubled.power.max() / spec.power.max()) == pytest.approx(
        6.020599913, abs=1e-9
    )


@pytest.mark.parametrize("size", [7, 8])
def test_fft_bins_and_power_at_spectrum_edges(size):
    sample = np.arange(size)
    samples = 0.5 + np.cos(2 * np.pi * (size // 2) * sample / size)
    spec = periodogram(samples, sample_rate_hz=8, window="boxcar", detrend=False)
    np.testing.assert_array_equal(spec.frequency_hz, np.fft.rfftfreq(size, d=1 / 8))
    assert spec.power.sum() * (8 / size) == pytest.approx(np.mean(samples**2))


def test_complete_windows_keep_original_sample_indices_and_gaps():
    samples = np.ma.array(np.arange(1024.0), mask=False)
    samples.mask[300] = True
    spec = windowed_psd(samples, sample_rate_hz=256, window_samples=256, step_samples=128)
    np.testing.assert_array_equal(spec.start_samples, [0, 128, 256, 384, 512, 640, 768])
    np.testing.assert_array_equal(spec.valid, [True, False, False, True, True, True, True])
    assert np.isnan(spec.power[~spec.valid]).all()
    assert np.isfinite(spec.power[spec.valid]).all()
    assert samples.mask[300] and samples.data[300] == 300


def test_all_missing_windows_keep_frequency_grid():
    spec = windowed_psd([np.nan] * 8, sample_rate_hz=8, window_samples=4, step_samples=4)
    np.testing.assert_array_equal(spec.frequency_hz, [0, 2, 4])
    assert spec.power.shape == (2, 3)
    assert not spec.valid.any()


def test_regrid_interpolates_linear_power_and_preserves_mask():
    spec = windowed_psd(
        [1, 2, 3, 4, 0, np.nan, 0, 0], sample_rate_hz=8, window_samples=4, step_samples=4
    )
    reduced = regrid_spectrogram(spec, [1, 3])
    np.testing.assert_allclose(reduced.power[0], (spec.power[0, :-1] + spec.power[0, 1:]) / 2)
    np.testing.assert_array_equal(reduced.valid, spec.valid)
    assert np.isnan(reduced.power[1]).all()
    with pytest.raises(ValueError, match="within"):
        regrid_spectrogram(spec, [1, 5])


@pytest.mark.parametrize("samples", [[1], [[1, 2]], [1, np.inf], [1j, 2j]])
def test_rejects_invalid_waveform(samples):
    with pytest.raises(ValueError):
        periodogram(samples, sample_rate_hz=8)


@pytest.mark.parametrize("rate", [0, -1, np.inf, np.nan, True])
def test_rejects_invalid_sample_rates(rate):
    with pytest.raises(ValueError, match="sample_rate_hz"):
        periodogram([1, 2], sample_rate_hz=rate)


@pytest.mark.parametrize(("window", "step"), [(0, 1), (9, 1), (4, 0), (4, 5), (4.0, 1)])
def test_rejects_invalid_window_geometry(window, step):
    with pytest.raises(ValueError):
        windowed_psd(np.zeros(8), sample_rate_hz=8, window_samples=window, step_samples=step)
