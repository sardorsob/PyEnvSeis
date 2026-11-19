"""Fluvial velocity power models and deterministic FMI catalogue construction.

Methods: Gimbert et al. (2014), doi:10.1002/2014JF003201, and Tsai et al.
(2012), doi:10.1029/2011GL050255. Numerical conventions follow the parametric
grain-size workflow of eseis 0.8.1. Frequencies are always explicit in Python.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .spectra import _frequency_grid

_GRAVITY = 9.81
_WATER_DENSITY = 1000.0
_RAISED_COSINE_STD = np.sqrt(1 / 3 - 2 / np.pi**2)
_LOG_GRAIN_NODES, _LOG_GRAIN_WEIGHTS = np.polynomial.legendre.leggauss(64)
_BEDLOAD_DIAMETERS = np.geomspace(1e-4, 10, 10_000)


@dataclass(frozen=True)
class ModelCandidate:
    """One candidate with an ID, water depth h_w (m), and unit flux q_s (m²/s)."""

    candidate_id: str
    h_w: float
    q_s: float


@dataclass(frozen=True)
class ModelCatalogue:
    """Model output matrices with shape (candidate, frequency)."""

    frequency_hz: NDArray[np.float64]
    candidates: tuple[ModelCandidate, ...]
    turbulence_power: NDArray[np.float64]
    bedload_power: NDArray[np.float64]
    combined_power: NDArray[np.float64]
    combined_db: NDArray[np.float64]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)


def _positive_frequency_grid(frequency_hz: ArrayLike) -> NDArray[np.float64]:
    frequency = _frequency_grid(frequency_hz)
    if frequency[0] <= 0:
        raise ValueError("frequency_hz must be strictly positive for the forward models")
    return frequency


def _finite_real(name: str, value: float) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")


def _positive_parameters(**values: float) -> None:
    for name, value in values.items():
        _finite_real(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be strictly positive")


def _bed_shear_velocity(h_w: float, a_w: float) -> np.float64:
    _finite_real("a_w", a_w)
    if not 0 < a_w < np.pi / 2:
        raise ValueError("a_w must lie between 0 and π/2 radians")
    # Take the square roots before multiplying to keep extreme positive depths
    # or angles from overflowing or underflowing the intermediate product.
    return np.sqrt(_GRAVITY * np.sin(a_w)) * np.sqrt(h_w)


def _propagation(
    frequency: NDArray[np.float64],
    *,
    r_0: float,
    f_0: float,
    q_0: float,
    v_0: float,
    p_0: float,
    e_0: float,
) -> NDArray[np.float64]:
    _positive_parameters(r_0=r_0, f_0=f_0, q_0=q_0, v_0=v_0)
    _finite_real("p_0", p_0)
    _finite_real("e_0", e_0)
    if p_0 <= -1:
        raise ValueError("p_0 must exceed -1 for positive Rayleigh wave group velocity")

    frequency_ratio = frequency / f_0
    phase_speed = v_0 * frequency_ratio ** (-p_0)
    group_speed = phase_speed / (1 + p_0)
    quality_factor = q_0 * frequency_ratio**e_0
    beta = (2 * np.pi * r_0) * frequency / (group_speed * quality_factor)
    # The line-source spreading/attenuation factor of Tsai et al. (2012).
    attenuation = np.exp(-beta)
    spreading = attenuation * (
        2 * np.log1p(1 / beta) * attenuation - np.expm1(-beta) * np.sqrt(2 * np.pi / beta)
    )
    return spreading / (phase_speed**3 * group_speed**2)


def _check_power(
    component: str, power: NDArray[np.float64], *, allow_zero: bool = False
) -> NDArray[np.float64]:
    if not np.isfinite(power).all() or np.any(power < 0) or (not allow_zero and np.any(power == 0)):
        requirement = "nonnegative" if allow_zero else "strictly positive"
        raise ValueError(f"{component} power must be finite and {requirement} on this grid")
    return power


def model_turbulence(
    frequency_hz: ArrayLike,
    *,
    d_s: float,
    s_s: float,
    r_s: float = 2650.0,
    h_w: float,
    w_w: float,
    a_w: float,
    r_0: float,
    f_0: float,
    q_0: float,
    v_0: float,
    p_0: float,
    n_0: ArrayLike,
    e_0: float = 0.0,
) -> NDArray[np.float64]:
    """Evaluate the Gimbert turbulence velocity PSD in (m/s)²/Hz.

    Supply grain diameter `d_s` (m), log-diameter standard deviation `s_s`,
    sediment density `r_s` (kg/m³), depth `h_w` and width `w_w` (m), slope
    angle `a_w` (radians), distance `r_0` (m), and Rayleigh propagation terms
    `q_0` and `v_0` (m/s) at `f_0` (Hz) with exponents `e_0` and `p_0`.
    `n_0` contains two displacement Green's function coefficients. The
    bed-profile approximation requires water depth above half the roughness
    length (3 × `d_s`).
    """
    frequency = _positive_frequency_grid(frequency_hz)
    _positive_parameters(d_s=d_s, s_s=s_s, r_s=r_s, h_w=h_w, w_w=w_w)
    shear_velocity = _bed_shear_velocity(h_w, a_w)
    coefficients = np.asanyarray(n_0)
    if coefficients.shape != (2,) or np.ma.is_masked(coefficients):
        raise ValueError("n_0 must contain two finite Green's function coefficients")
    for coefficient in coefficients:
        _finite_real("n_0", coefficient)
    if not np.any(coefficients):
        raise ValueError("at least one turbulence n_0 coefficient must be nonzero")

    roughness = 3 * d_s
    relative_roughness = roughness / h_w
    if relative_roughness >= 2:
        raise ValueError("h_w must exceed half the roughness length (3 * d_s)")

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        propagation = _propagation(frequency, r_0=r_0, f_0=f_0, q_0=q_0, v_0=v_0, p_0=p_0, e_0=e_0)
        greens_power = np.sum(coefficients.astype(np.float64) ** 2)
        # Eqs. (43)–(44): integrate the spectral shape over the log-raised cosine.
        # Using a normalized log diameter cancels the density's 1/D Jacobian.
        convection = (4 - relative_roughness) * shear_velocity
        half_width = s_s / _RAISED_COSINE_STD
        diameter = d_s * np.exp(half_width * _LOG_GRAIN_NODES)
        response = 1 / (1 + np.outer(2 * frequency / convection, diameter) ** (4 / 3))
        grain_measure = 0.5 * (1 + np.cos(np.pi * _LOG_GRAIN_NODES)) * _LOG_GRAIN_WEIGHTS
        grain_integral = response**2 @ (grain_measure * diameter**2)

        # Bed-profile coefficients and fixed fluid constants of the eseis parameterization.
        eddy_height = 8 * (1 - relative_roughness / 2)
        turbulence_intensity = 0.2 * (4 - 5.62 * np.log10(relative_roughness))
        bed_profile = eddy_height ** (2 / 3) * (4 - relative_roughness) ** (8 / 3)
        bed_profile *= turbulence_intensity ** (4 / 3)
        scale = greens_power * bed_profile * w_w / (24 * np.float64(roughness) ** (2 / 3))
        scale *= (_WATER_DENSITY / np.float64(r_s)) ** 2 * shear_velocity ** (14 / 3)
        power = scale * propagation * frequency ** (4 / 3) * grain_integral
    return _check_power("turbulence", power)


def _grain_bins(
    d_s: float, s_s: float, adjust: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # eseis 0.8.1 uses the same fixed diameter template with either setting.
    half_width = s_s / _RAISED_COSINE_STD
    offset = (np.log(_BEDLOAD_DIAMETERS) - np.log(d_s)) / half_width
    inside = np.abs(offset) < 1
    diameter = _BEDLOAD_DIAMETERS[inside]
    if diameter.size == 0 or (adjust and diameter.size == 1):
        raise ValueError("grain-size distribution has insufficient support on the bedload grid")

    density = np.cos(np.pi * offset[inside] / 2) ** 2 / diameter
    if adjust:
        # Right endpoint weights and a zero first width reproduce eseis's
        # truncated-template integral rather than a renormalized distribution.
        return diameter, density * np.diff(diameter, prepend=diameter[0]) / half_width
    return diameter, density / density.sum()


def _impact_strength(
    diameter: NDArray[np.float64],
    *,
    d_s: float,
    h_w: float,
    a_w: float,
    shear_velocity: np.float64,
    r_s: float,
) -> NDArray[np.float64]:
    # Tsai et al. (2012), Eqs. (13)–(18), with the reference's caps on
    # hop height (water depth) and horizontal bedload speed (bulk flow).
    buoyant_gravity = _GRAVITY * ((r_s - _WATER_DENSITY) / _WATER_DENSITY)
    log_slope = 0.407 * np.log(142 * np.tan(a_w))
    critical_shields = np.exp(
        np.polynomial.polynomial.polyval(log_slope, [-3.14, 0.41, 0.142, 0.0894, 0.0259])
    )
    shields_ratio = (shear_velocity**2 / (buoyant_gravity * diameter)) / (
        critical_shields * (diameter / d_s) ** (-0.9)
    )
    hop_height = np.minimum(1.44 * diameter * np.sqrt(shields_ratio), h_w)
    bulk_velocity = 8.1 * shear_velocity * (np.float64(h_w) / (3 * d_s)) ** (1 / 6)
    bedload_velocity = np.minimum(
        1.56 * np.sqrt(buoyant_gravity * diameter) * shields_ratio**0.56, bulk_velocity
    )

    # The reference uses the Dietrich settling law with shape factor 0.8,
    # roundness 3.5 and kinematic viscosity 10⁻⁶ m²/s.
    viscosity = 1e-6
    log_transport = np.log10(buoyant_gravity * diameter**3 / viscosity**2)
    settling_curve = np.polynomial.polynomial.polyval(
        log_transport, [-3.76715, 1.92944, -0.09815, -0.00575, 0.00056]
    )
    shape = 0.8
    shape_tanh = np.tanh(log_transport - 4.6)
    settling_curve += np.log10(1 - (1 - shape) / 0.85) - (1 - shape) ** 2.3 * shape_tanh
    settling_curve += 0.3 * (0.5 - shape) * (1 - shape) ** 2 * (log_transport - 4.6)
    settling_scale = 0.65 - (shape / 2.83) * shape_tanh
    terminal_velocity = np.cbrt(buoyant_gravity * viscosity * settling_scale * 10**settling_curve)
    drag = (4 / 3) * buoyant_gravity * diameter / terminal_velocity**2
    normal = np.cos(a_w)
    fall = 1.5 * drag * (_WATER_DENSITY / r_s) * hop_height / (diameter * normal)
    impact_fraction = np.sqrt(-np.expm1(-fall))
    impact_velocity = terminal_velocity * normal * impact_fraction
    # Algebraic form of 2 arccosh(exp(fall/2)) that avoids exp(fall) overflow.
    settling_velocity = terminal_velocity * normal * fall / (fall + 2 * np.log1p(impact_fraction))
    particle_volume = (np.pi / 6) * diameter**3
    return (
        particle_volume * settling_velocity * impact_velocity**2 / (bedload_velocity * hop_height)
    )


def model_bedload(
    frequency_hz: ArrayLike,
    *,
    d_s: float,
    s_s: float,
    r_s: float = 2650.0,
    q_s: float,
    h_w: float,
    w_w: float,
    a_w: float,
    r_0: float,
    f_0: float,
    q_0: float,
    v_0: float,
    p_0: float,
    n_0: float,
    e_0: float = 0.0,
    adjust: bool = True,
) -> NDArray[np.float64]:
    """Evaluate the Tsai bedload velocity PSD in (m/s)²/Hz.

    Parameters shared with `model_turbulence` have the same units. `q_s` is
    nonnegative volumetric sediment flux per unit channel width (m²/s).
    `r_s` exceeds the water density of 1,000 kg/m³. `n_0` is the scalar
    vertical-response coefficient. The parametric grain-size distribution
    uses eseis's fixed 10,000-point template from 10⁻⁴ to 10 m. `adjust=True`
    applies its grain-bin widths. `False` uses normalized sampled densities.
    """
    frequency = _positive_frequency_grid(frequency_hz)
    _positive_parameters(d_s=d_s, s_s=s_s, r_s=r_s, h_w=h_w, w_w=w_w)
    shear_velocity = _bed_shear_velocity(h_w, a_w)
    _finite_real("q_s", q_s)
    _finite_real("n_0", n_0)
    if q_s < 0:
        raise ValueError("q_s must be nonnegative")
    if r_s <= _WATER_DENSITY:
        raise ValueError("r_s must exceed the fluid density (1000 kg/m³)")
    if not isinstance(adjust, (bool, np.bool_)):
        raise ValueError("adjust must be a boolean")

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        propagation = _propagation(frequency, r_0=r_0, f_0=f_0, q_0=q_0, v_0=v_0, p_0=p_0, e_0=e_0)
        diameter, weights = _grain_bins(d_s, s_s, adjust)
        if q_s == 0 or n_0 == 0:
            return np.zeros_like(frequency)
        impact = _impact_strength(
            diameter, d_s=d_s, h_w=h_w, a_w=a_w, shear_velocity=shear_velocity, r_s=r_s
        )
        strength = (2 / 3) * np.pi**2 * w_w * q_s * np.float64(n_0) ** 2 * np.dot(weights, impact)
        power = strength * frequency**3 * propagation
    return _check_power("bedload", power)


def build_catalogue(
    frequency_hz: ArrayLike,
    *,
    candidates: Sequence[ModelCandidate],
    d_s: float,
    s_s: float,
    r_s: float = 2650.0,
    w_w: float,
    a_w: float,
    r_0: float,
    f_0: float,
    q_0: float,
    v_0: float,
    p_0: float,
    n_0_a: float,
    n_0_b: float,
    e_0: float = 0.0,
) -> ModelCatalogue:
    """Build a deterministic forward catalogue in input candidate order.

    The eseis 0.8.1 FMI mapping sends both Green's coefficients and `e_0=0`
    to turbulence. Bedload receives `n_0_a`, the caller's `e_0`, and adjusted
    grain-bin widths. Combined dB is `10 * log10(turbulence + bedload)`.
    Both models evaluate the supplied grid, including uneven spacing as a
    Python extension to the regularly spaced FMI reference catalogue.
    """
    frequency = _positive_frequency_grid(frequency_hz).copy()
    rows = tuple(candidates)
    if not rows or any(not isinstance(row, ModelCandidate) for row in rows):
        raise ValueError("candidates must contain at least one ModelCandidate")
    ids = tuple(row.candidate_id for row in rows)
    if any(not isinstance(name, str) or not name for name in ids):
        raise ValueError("candidate IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")

    turbulence = np.empty((len(rows), frequency.size), dtype=np.float64)
    bedload = np.empty_like(turbulence)
    reach = dict(
        d_s=d_s, s_s=s_s, r_s=r_s, w_w=w_w, a_w=a_w, r_0=r_0, f_0=f_0, q_0=q_0, v_0=v_0, p_0=p_0
    )
    for index, candidate in enumerate(rows):
        try:
            turbulence[index] = model_turbulence(
                frequency, **reach, h_w=candidate.h_w, n_0=(n_0_a, n_0_b)
            )
            bedload[index] = model_bedload(
                frequency,
                **reach,
                h_w=candidate.h_w,
                q_s=candidate.q_s,
                n_0=n_0_a,
                e_0=e_0,
                adjust=True,
            )
        except ValueError as error:
            raise ValueError(
                f"candidate {candidate.candidate_id!r} has invalid model input: {error}"
            ) from error
    with np.errstate(over="ignore"):
        combined = _check_power("combined", turbulence + bedload)
    return ModelCatalogue(frequency, rows, turbulence, bedload, combined, 10 * np.log10(combined))
