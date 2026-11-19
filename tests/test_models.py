import numpy as np
import pytest

from fluvialseis import models
from fluvialseis.models import ModelCandidate, build_catalogue, model_bedload, model_turbulence

FREQUENCY = np.array([5.0, 12.0, 40.0, 117.0])
REACH = {
    "d_s": 0.07,
    "s_s": 0.5,
    "r_s": 2650.0,
    "w_w": 11.0,
    "a_w": 0.1,
    "r_0": 14.0,
    "f_0": 1.0,
    "q_0": 20.0,
    "v_0": 1200.0,
    "p_0": 0.48,
}


def test_direct_model_options_against_eseis_081():
    # Rows 1, 44, and 100 of seq(5, 120, length.out=100), evaluated in
    # R 4.5.2 by the CRAN 0.8.1 model_turbulence and model_bedload functions.
    # These settings exercise direct calls with a nonzero turbulence Q
    # exponent and the unadjusted bedload grain-bin weighting.
    frequency = [5.0, 54.94949494949495, 120.0]
    turbulence = model_turbulence(frequency, **REACH, h_w=0.5, n_0=(0.8, 1.0), e_0=0.17)
    bedload = model_bedload(
        frequency, **REACH, h_w=0.5, n_0=1.0, e_0=0.17, q_s=2.5 / 2650, adjust=False
    )

    np.testing.assert_allclose(
        turbulence,
        [7.463387111601984e-14, 2.412411958260617e-12, 8.268927904877184e-13],
        rtol=1e-7,
        atol=0,
    )
    np.testing.assert_allclose(
        bedload,
        [4.749297091945815e-16, 2.340551840385854e-11, 1.742817566215496e-10],
        rtol=1e-12,
        atol=0,
    )


def test_turbulence_quadrature_converges_when_doubling_nodes(monkeypatch):
    frequency = np.linspace(5, 120, 100)
    power = model_turbulence(frequency, **REACH, h_w=0.2, n_0=(0.8, 1.0))

    nodes, weights = np.polynomial.legendre.leggauss(128)
    monkeypatch.setattr(models, "_LOG_GRAIN_NODES", nodes)
    monkeypatch.setattr(models, "_LOG_GRAIN_WEIGHTS", weights)
    refined = model_turbulence(frequency, **REACH, h_w=0.2, n_0=(0.8, 1.0))

    np.testing.assert_allclose(power, refined, rtol=1e-12, atol=0)


def test_turbulence_uses_both_greens_coefficients():
    power = model_turbulence(FREQUENCY, **REACH, h_w=0.5, n_0=(0.8, 1.0))
    normal_only = model_turbulence(FREQUENCY, **REACH, h_w=0.5, n_0=(0.8, 0.0))

    assert power.shape == FREQUENCY.shape
    assert np.all(np.isfinite(power) & (power > 0))
    np.testing.assert_allclose(power, normal_only * (0.8**2 + 1.0) / 0.8**2, atol=0)


@pytest.mark.parametrize("adjust", [True, False])
def test_bedload_scales_with_flux_and_preserves_exact_zero(adjust):
    args = dict(REACH, h_w=0.5, n_0=0.8, e_0=0.17, adjust=adjust)
    power = model_bedload(FREQUENCY, **args, q_s=0.001)

    assert np.all(np.isfinite(power) & (power > 0))
    np.testing.assert_allclose(model_bedload(FREQUENCY, **args, q_s=0.002), 2 * power, atol=0)
    np.testing.assert_array_equal(model_bedload(FREQUENCY, **args, q_s=0.0), np.zeros(4))


def test_models_evaluate_the_exact_frequency_grid():
    # Two frequencies are two evaluation points even on an uneven grid.
    select = [0, 2]
    for model, options in (
        (model_turbulence, {"n_0": (0.8, 1.0)}),
        (model_bedload, {"n_0": 0.8, "q_s": 0.001}),
    ):
        kwargs = dict(REACH, h_w=0.5, **options)
        full = model(FREQUENCY, **kwargs)
        np.testing.assert_allclose(model(FREQUENCY[select], **kwargs), full[select], atol=0)


def test_catalogue_keeps_row_order_and_uses_the_eseis_parameter_mapping():
    candidates = (
        ModelCandidate("higher", h_w=1.2, q_s=0.002),
        ModelCandidate("lower", h_w=0.2, q_s=0.0),
    )
    catalogue = build_catalogue(
        FREQUENCY, **REACH, candidates=candidates, n_0_a=0.8, n_0_b=1.0, e_0=0.17
    )

    assert catalogue.candidate_ids == ("higher", "lower")
    assert catalogue.candidates == candidates
    np.testing.assert_array_equal(catalogue.frequency_hz, FREQUENCY)
    for row, candidate in enumerate(candidates):
        turbulence = model_turbulence(
            FREQUENCY, **REACH, h_w=candidate.h_w, n_0=(0.8, 1.0), e_0=0.0
        )
        bedload = model_bedload(
            FREQUENCY, **REACH, h_w=candidate.h_w, q_s=candidate.q_s, n_0=0.8, e_0=0.17
        )
        np.testing.assert_array_equal(catalogue.turbulence_power[row], turbulence)
        np.testing.assert_array_equal(catalogue.bedload_power[row], bedload)
        np.testing.assert_array_equal(catalogue.combined_power[row], turbulence + bedload)
        np.testing.assert_array_equal(
            catalogue.combined_db[row], 10 * np.log10(turbulence + bedload)
        )


@pytest.mark.parametrize("frequency", [[0.0, 1.0], [-1.0], [2.0, 1.0], [2.0, 2.0], [np.nan]])
def test_forward_models_reject_invalid_frequency_axes(frequency):
    with pytest.raises(ValueError, match="frequency_hz"):
        model_turbulence(frequency, **REACH, h_w=0.5, n_0=(0.8, 1.0))
    with pytest.raises(ValueError, match="frequency_hz"):
        model_bedload(frequency, **REACH, h_w=0.5, q_s=0.001, n_0=0.8)


@pytest.mark.parametrize(
    ("field", "value"),
    [("d_s", 0.0), ("s_s", np.nan), ("a_w", 0.0), ("a_w", np.pi / 2), ("p_0", -1.0)],
)
def test_models_reject_invalid_common_parameters(field, value):
    kwargs = dict(REACH, h_w=0.5)
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        model_turbulence(FREQUENCY, **kwargs, n_0=(0.8, 1.0))
    with pytest.raises(ValueError, match=field):
        model_bedload(FREQUENCY, **kwargs, q_s=0.001, n_0=0.8)


def test_invalid_depth_flux_and_grain_support_raise():
    with pytest.raises(ValueError, match="h_w"):
        model_turbulence(FREQUENCY, **REACH, h_w=0.05, n_0=(0.8, 1.0))
    with pytest.raises(ValueError, match="q_s"):
        model_bedload(FREQUENCY, **REACH, h_w=0.5, q_s=-0.001, n_0=0.8)
    with pytest.raises(ValueError, match="r_s"):
        model_bedload(FREQUENCY, **dict(REACH, r_s=999.0), h_w=0.5, q_s=0.001, n_0=0.8)
    with pytest.raises(ValueError, match="grain"):
        model_bedload(FREQUENCY, **dict(REACH, d_s=1000.0), h_w=0.5, q_s=0.001, n_0=0.8)


@pytest.mark.parametrize("coefficient", [0.8, (0.8,), (0.8, np.nan), (0.0, 0.0)])
def test_turbulence_requires_two_valid_greens_coefficients(coefficient):
    with pytest.raises(ValueError, match="n_0"):
        model_turbulence(FREQUENCY, **REACH, h_w=0.5, n_0=coefficient)


@pytest.mark.parametrize("ids", [(), ("",), ("repeat", "repeat")])
def test_catalogue_requires_nonempty_distinct_ids(ids):
    candidates = [ModelCandidate(name, h_w=0.5, q_s=0.001) for name in ids]
    with pytest.raises(ValueError, match="candidate"):
        build_catalogue(FREQUENCY, **REACH, candidates=candidates, n_0_a=0.8, n_0_b=1.0)
