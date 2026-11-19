# PyEnvSeis

PyEnvSeis is a native Python library for analyzing ground vibrations generated
by rivers. It computes velocity power spectral densities (PSDs), models
turbulence and bedload spectra, and finds the best match in an ordered
catalogue of flow depths and bedload fluxes.

Version 0.1.0a2 is a research preview for CPython 3.12–3.14. The package
distribution and Python import are named `fluvialseis`. Pass waveform samples
as arrays after reading and calibrating them with your seismic data tools.

## Install

From a local checkout, run:

```sh
python -m pip install .
```

You can also install the built `fluvialseis-0.1.0a2-py3-none-any.whl` directly
with pip. The runtime requires NumPy and SciPy and runs entirely in Python
without an R installation or an internet connection once installed.

## Estimate a velocity spectrum

```python
import numpy as np

from fluvialseis import periodogram

sample_rate_hz = 256
time_s = np.arange(1024) / sample_rate_hz
velocity_m_s = 2e-6 * np.sin(2 * np.pi * 32 * time_s)

spectrum = periodogram(
    velocity_m_s, sample_rate_hz=sample_rate_hz, window="boxcar", detrend=False
)
peak_hz = spectrum.frequency_hz[np.argmax(spectrum.power)]
print(f"Peak: {peak_hz:g} Hz")  # Peak: 32 Hz
```

Use calibrated ground velocity in m/s when working with real waveforms. The
periodogram uses SciPy's one-sided density normalization and true FFT frequency
bins. Output power is linear in (m/s)²/Hz. The defaults use a Hann window and
remove the mean from each segment. [SciPy periodogram documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html)

## Compare a series of spectra

`windowed_psd` computes a PSD for each complete window, with explicit window
and step lengths in samples. It returns window starts as sample indices. NaN,
infinite, and masked samples invalidate any window that contains them. You can
use `regrid_spectrogram` to interpolate linear power within the observed band.

Pass `fit_catalogue` the observation, a shared frequency axis in Hz, an ordered
list of candidate IDs, and a matrix of positive linear powers with one row per
candidate. Both sides must use the same PSD units and normalization. The
function selects the candidate with the lowest across-frequency RMSE in dB
and returns that score separately from the per-frequency absolute residuals.
It also reports the score margin to the runner-up. Invalid or nonpositive
observed windows have no match. On an exact tie, the first candidate wins.

```python
import numpy as np

from fluvialseis import fit_catalogue, regrid_spectrogram, windowed_psd

sample_rate_hz = 256
time_s = np.arange(1024) / sample_rate_hz
velocity_m_s = 2e-6 * np.sin(2 * np.pi * 32 * time_s)
observed = windowed_psd(
    velocity_m_s, sample_rate_hz=sample_rate_hz, window_samples=256, step_samples=128
)
observed = regrid_spectrogram(observed, [31, 32, 33])
reference = np.array([1.0, 4.0, 1.0]) * (1e-12 / 3)
fit = fit_catalogue(
    observed,
    frequency_hz=observed.frequency_hz,
    candidate_ids=["reference", "double power"],
    candidate_power=np.vstack([reference, 2 * reference]),
)
assert fit.candidate_ids == ("reference",) * 7
```

A catalogue match describes spectral agreement. Interpreting a match as water
depth or sediment transport requires a validated forward model, site parameters,
and independent field measurements.

## Reproduce the eseis fluvial workflow

`eseis_spectrum` and `eseis_spectrogram` reproduce the default periodogram
convention of `eseis` 0.8.1, including detrending, tapering and padding. In
that convention, the reported bins are relabeled from zero to Nyquist. Use
`fft_frequency_hz` when inspecting the actual retained FFT bins. The
spectrogram omits an exactly terminal window. The clipping helper interpolates
in dB with the cubic spline used by the reference workflow. Its end cubics
also extrapolate, so choose a band within the measured frequency range.

```python
import numpy as np

from fluvialseis import (
    ModelCandidate,
    build_catalogue,
    clip_eseis_spectrogram,
    eseis_spectrogram,
    fit_eseis_catalogue,
)

fs = 256
t = np.arange(1024) / fs
signal = 1e-6 * (
    np.sin(2 * np.pi * 32 * t)
    + 0.7 * np.cos(2 * np.pi * (12 * t + 2 * t**2))
    + 0.35 * np.sin(2 * np.pi * 87.3 * t)
    + 0.2 * np.cos(2 * np.pi * (65 * t - 5 * t**2))
)
raw = eseis_spectrogram(signal, sample_rate_hz=fs, window_seconds=1, overlap=0)
observed = clip_eseis_spectrogram(raw, frequency_range_hz=(5, 120), resolution=100)

catalogue = build_catalogue(
    observed.frequency_hz,
    candidates=[ModelCandidate("shallow", 0.2, 0), ModelCandidate("deeper", 1.2, 2.5/2650)],
    d_s=0.07, s_s=0.5, r_s=2650, w_w=11, a_w=0.1,
    r_0=14, f_0=1, q_0=20, v_0=1200, p_0=0.48, n_0_a=0.8, n_0_b=1.0,
)
fit = fit_eseis_catalogue(
    observed.power_db,
    candidate_ids=catalogue.candidate_ids,
    candidate_power_db=catalogue.combined_db,
)
print(fit.candidate_ids)
```

Each candidate specifies a water depth in metres and a bedload volume flux
per unit width in m²/s. `d_s` is median grain diameter in metres, `s_s` is
the dimensionless spread in natural-log diameter, and `a_w` is slope angle in
radians. The catalogue computes separate linear powers and adds them before
converting to dB. The fitter accepts arrays with the same frequency columns
in the same order and reports the winning ID, dB RMSE, absolute bin residuals,
and runner-up score margin.

The comparison against the R source covers finite inputs, the default
periodogram, spline clipping, the parametric grain-size model, and deterministic
candidate rows. Missing or nonfinite spectral windows are skipped in Python.
The default R periodogram's normalization differs from the conventional
velocity PSD above. Replaying the R comparison does not establish physical
comparability with the modeled powers, and the synthetic example has no
known true depth or flux. [eseis source and manual](https://cran.r-project.org/package=eseis)

## Development

```sh
python -m pip install ".[test]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

Build a wheel and source distribution with `python -m build` after installing
the `build` frontend. GitHub Actions is configured to test pip installs across
three Python versions on Linux, macOS, and Windows.

## Citation and license

PyEnvSeis reproduces a fluvial workflow from the R package **eseis**, the Environmental
Seismology Toolbox developed by Michael Dietze and contributors Christoph
Burow, Sophie Lagarde, and Clément Hibert. The catalogue-matching workflow is
informed by the fluvial model inversion method of Dietze and colleagues.
[eseis package](https://cran.r-project.org/package=eseis) ·
[Dietze 2018](https://doi.org/10.5194/esurf-6-669-2018) ·
[Dietze et al. 2019](https://doi.org/10.1029/2019WR026072) ·
[Gimbert et al. 2014, turbulence](https://doi.org/10.1002/2014JF003201) ·
[Tsai et al. 2012, bedload](https://doi.org/10.1029/2011GL050255)

See [CITATION.cff](CITATION.cff) for citation metadata. PyEnvSeis's original
Python implementation and documentation are available under the [MIT License](LICENSE).
The referenced eseis software is distributed separately under GPL-3.
