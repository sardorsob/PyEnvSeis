# PyEnvSeis user guide

PyEnvSeis estimates seismic spectra and reproduces a fluvial model inversion
workflow from R's **eseis** 0.8.1. The project is named PyEnvSeis. Use
`fluvialseis` as the pip distribution and Python import name.

## Install

Use CPython 3.12, 3.13 or 3.14. From the PyEnvSeis checkout, create a virtual
environment and install the package. On Windows, activate it with
`.venv\Scripts\activate` in Command Prompt.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -c "from importlib.metadata import version; print(version('fluvialseis'))"
```

The parity preview reports `0.1.0a2`. If you already have the built wheel,
run `python -m pip install /path/to/fluvialseis-0.1.0a2-py3-none-any.whl`
in your environment. The package uses NumPy and SciPy. R is only needed to
regenerate R reference results.

## Choose the spectrum convention

For a conventional one-sided velocity PSD, pass calibrated ground velocity in
m/s to `periodogram` or `windowed_psd`. Power is in (m/s)²/Hz and the frequency
axis uses the actual FFT bins.

```python
import numpy as np
from fluvialseis import periodogram

fs = 256
t = np.arange(1024) / fs
velocity = 2e-6 * np.sin(2 * np.pi * 32 * t)
psd = periodogram(velocity, sample_rate_hz=fs, window="boxcar", detrend=False)
assert psd.frequency_hz[np.argmax(psd.power)] == 32
```

To replay an R workflow, use `eseis_spectrum` or `eseis_spectrogram`. The
historical R convention uses a linear detrend, a split-cosine taper and FFT
padding. It keeps the positive FFT bins, normalizes power using sample
frequency 1, then labels the output from zero to Nyquist. The raw result
exposes both `frequency_hz` for those labels and `fft_frequency_hz` for the
transform bins. Keep observations and model inputs in an explicitly defined
common convention when interpreting a fit.

## Run a fluvial inversion

The example uses 1,024 synthetic samples and five fixed candidates. Paste each
block in order into a Python script. For field data, read and calibrate the
waveforms upstream and supply regularly spaced samples and their actual rate.

### Compute and clip the observed spectrogram

```python
import numpy as np
from fluvialseis import clip_eseis_spectrogram, eseis_spectrogram

fs = 256
t = np.arange(1024) / fs
samples = 1e-6 * (
    np.sin(2 * np.pi * 32 * t)
    + 0.7 * np.cos(2 * np.pi * (12 * t + 2 * t**2))
    + 0.35 * np.sin(2 * np.pi * 87.3 * t)
    + 0.2 * np.cos(2 * np.pi * (65 * t - 5 * t**2))
)
raw = eseis_spectrogram(samples, sample_rate_hz=fs, window_seconds=1, overlap=0)
observed = clip_eseis_spectrogram(raw, frequency_range_hz=(5, 120), resolution=100)
assert observed.power_db.shape == (3, 100)
assert observed.start_samples.tolist() == [0, 256, 512]
```

Rows are time windows and columns are frequency labels. The R indexing rule
omits a window that would end on the last sample. The clip interpolates in dB
using the Forsythe–Malcolm–Moler cubic spline. Its end cubics extrapolate, so
choose a band within the observed frequency range. Python marks windows that
contain nonfinite samples as invalid.

### Build the reference catalogue

```python
from fluvialseis import ModelCandidate, build_catalogue

catalogue = build_catalogue(
    observed.frequency_hz,
    candidates=[
        ModelCandidate("A", 0.2, 0),
        ModelCandidate("B", 0.2, 2.5 / 2650),
        ModelCandidate("C", 1.2, 0),
        ModelCandidate("D", 1.2, 2.5 / 2650),
        ModelCandidate("E", 1.2, 5 / 2650),
    ],
    d_s=0.07,
    s_s=0.5,
    r_s=2650,
    w_w=11,
    a_w=0.1,
    r_0=14,
    f_0=1,
    q_0=20,
    v_0=1200,
    p_0=0.48,
    n_0_a=0.8,
    n_0_b=1.0,
    e_0=0.17,
)
assert catalogue.combined_db.shape == (5, 100)
```

Each row specifies an ID, water depth `h_w` in metres, and volumetric
bedload flux `q_s` per unit width in m²/s. The example uses a median grain
diameter `d_s` of 0.07 m, a dimensionless spread `s_s` in natural-log
diameter, sediment density `r_s` in kg/m³, channel width `w_w` in metres,
and slope angle `a_w` in radians. The default turbulence formulation
requires depth greater than half the roughness length, `1.5 * d_s`.

Set `r_0` to the source-to-station distance in metres. The reference
frequency `f_0` is in Hz, the reference phase velocity `v_0` in m/s, and
the quality factor `q_0`, dispersion exponent `p_0` and quality-factor
exponent `e_0` are dimensionless. The catalogue uses both Green's function coefficients
and a fixed quality factor exponent of zero for turbulence. It uses `n_0_a`
and the supplied `e_0` for bedload, following the R catalogue path. The
models produce separate linear velocity powers, add them, then convert the sum
to dB. [Gimbert et al., turbulence](https://doi.org/10.1002/2014JF003201) ·
[Tsai et al., bedload](https://doi.org/10.1029/2011GL050255) ·
[Dietze et al., fluvial model inversion](https://doi.org/10.1029/2019WR026072)

### Match the windows

```python
from fluvialseis import fit_eseis_catalogue

np.testing.assert_array_equal(observed.frequency_hz, catalogue.frequency_hz)
fit = fit_eseis_catalogue(
    observed.power_db,
    candidate_ids=catalogue.candidate_ids,
    candidate_power_db=catalogue.combined_db,
)
assert fit.candidate_ids == ("A", "A", "A")
print(np.round(fit.rmse_db, 2))  # [17.8  19.84 21.52]
```

The matcher minimizes unweighted RMSE in dB. `rmse_db` is one score per
window, `residual_db` is the absolute difference at each frequency for the
winning row, and `margin_db` is the score gap to the runner-up. A zero gap
means an exact tie and the first row wins. Nonfinite observed windows have a
missing candidate ID. The fitter works on arrays, so equal column counts
alone cannot establish alignment. Check the actual frequency axes before
fitting, as in the example.

A winning row describes relative spectral fit within the specified
catalogue. The example has no ground-truth depth or flux. Physical inference
requires calibrated velocity, a justified normalization and site parameters,
and validation against independent observations.

## What matches R

| R workflow operation | Python equivalent | Comparison scope |
| --- | --- | --- |
| `signal_spectrum` | `eseis_spectrum` | Default periodogram, running mean and linear/log frequency-grid resampling. Near-zero roundoff from the R running mean can differ. |
| `signal_spectrogram` | `eseis_spectrogram` | Default serial periodogram, overlap, window alignment and historical frequency labels on finite traces. Python also masks missing-sample windows. |
| Notebook `aux_clipspectrogram` | `clip_eseis_spectrogram` | Band clipping with FMM interpolation in dB. The audited helper is notebook code used alongside eseis. |
| `model_turbulence`, `model_bedload` | Same function names | Separate model powers on a common grid with the supported parametric grain-size distribution. |
| `fmi_parameters`, `fmi_spectra` | `ModelCandidate`, `build_catalogue` | Explicit ordered rows, component mapping and linear power addition before dB. |
| `fmi_inversion` | `fit_eseis_catalogue` | Unweighted dB scoring and first-row tie rule. R's `rmse` matrix corresponds to Python's per-bin `residual_db`. |

The verified scope excludes R's random candidate generation, Welch and
autoregressive spectra, waveform response removal, empirical grain-size
distributions and other environmental-seismology workflows. [eseis 0.8.1
manual and source](https://cran.r-project.org/package=eseis)

## Tests and reference results

To run the package tests from the checkout:

```sh
python -m pip install ".[test]"
python -m pytest
```

The 67 tests cover analytic native PSDs, legacy spectra and interpolation,
gap and clock handling, component models, candidate ordering and inversion.
The wheel and source distribution passed the suite on macOS arm64 with
CPython 3.12, 3.13 and 3.14.

The numerical benchmark used the pinned eseis 0.8.1 functions under R 4.5.2
and the notebook's clipping helper. Two R runs produced 23 byte-identical
reference tables. The comparison checked frequency axes and window starts
first, then spectra, separate component powers, scores and selected IDs.

| Synthetic comparison | Measured maximum difference |
| --- | ---: |
| Raw spectrogram | 5.78 × 10⁻¹¹ dB |
| Spectrogram after clipping to 5–120 Hz | 2.65 × 10⁻¹¹ dB |
| Combined model spectra | 2.10 × 10⁻⁷ dB |
| Catalogue scores with identical R inputs | 7.11 × 10⁻¹⁵ dB |
| Winning scores with independently computed inputs | 6.38 × 10⁻⁸ dB |

All three coupled windows chose row A, and the R score margins to the
runner-up exceeded 22 dB. The shared-array tests also exercised an exact tie,
nearly equal scores and a missing window. These results apply to the pinned
workflow and the tested inputs.
