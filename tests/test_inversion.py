import numpy as np
import pytest

from fluvialseis import Spectrogram, fit_catalogue


def test_first_tied_candidate_wins_with_separate_rmse_and_bin_residuals():
    frequency = np.array([15.0, 20.0, 25.0])
    candidate_db = np.array([[-135, -155, -160], [-140, -145, -152], [-140, -145, -152]])
    observed_db = candidate_db[1] + [0, 3, -4]
    power = np.vstack([10 ** (observed_db / 10), [np.nan] * 3, [0, 1e-14, 1e-15]])
    observed = Spectrogram(frequency, np.array([0, 10, 20]), power, np.array([True, False, True]))
    fit = fit_catalogue(
        observed,
        frequency_hz=frequency,
        candidate_ids=["a", "b", "c"],
        candidate_power=10 ** (candidate_db / 10),
    )
    assert fit.candidate_ids == ("b", None, None)
    assert fit.rmse_db[0] == pytest.approx(5 / np.sqrt(3))
    assert fit.margin_db[0] == 0
    np.testing.assert_allclose(fit.residual_db[0], [0, 3, 4], atol=1e-12)
    assert np.isnan(fit.rmse_db[1:]).all()


@pytest.mark.parametrize(
    ("frequency", "ids", "rows"),
    [
        ([1, 3], ["a"], [[1, 2]]),
        ([1, 2], ["a", "a"], [[1, 2], [2, 3]]),
        ([1, 2], [], []),
        ([1, 2], ["a"], [[0, 2]]),
    ],
)
def test_rejects_ambiguous_or_incompatible_catalogues(frequency, ids, rows):
    observed = Spectrogram(
        np.array([1.0, 2.0]), np.array([0]), np.array([[1.0, 2.0]]), np.array([True])
    )
    with pytest.raises(ValueError):
        fit_catalogue(observed, frequency_hz=frequency, candidate_ids=ids, candidate_power=rows)


def test_eseis_matcher_accepts_db_and_preserves_row_order_and_missing_windows():
    from fluvialseis.inversion import fit_eseis_catalogue

    references = [[-100, -90, -80], [-100, -90, -80], [-92, -80, -70]]
    observations = [[-100, -93, -84], [-92, -80, -70], [np.nan, -90, -80]]
    fit = fit_eseis_catalogue(
        observations, candidate_ids=["first", "duplicate", "other"], candidate_power_db=references
    )
    assert fit.candidate_ids == ("first", "other", None)
    np.testing.assert_allclose(fit.rmse_db, [5 / np.sqrt(3), 0, np.nan])
    np.testing.assert_allclose(fit.residual_db[0], [0, 3, 4])
    assert fit.margin_db[0] == 0
    assert fit.margin_db[1] > 0
    assert np.isnan(fit.residual_db[2]).all()


def test_eseis_matcher_rejects_mismatched_frequency_dimensions():
    from fluvialseis.inversion import fit_eseis_catalogue

    with pytest.raises(ValueError, match="frequency"):
        fit_eseis_catalogue([[-95, -90]], candidate_ids=["a"], candidate_power_db=[[-95]])
