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
