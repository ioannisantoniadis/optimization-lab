import numpy as np
import pytest

from optimlab.inverse.deblurring import blur_image, deblurring_problem
from optimlab.optimizers import bfgs


def _synthetic_scene(seed=0):
    n = 24
    true_image = np.zeros((n, n))
    true_image[8:16, 8:16] = 1.0
    rng = np.random.default_rng(seed)
    blurred = blur_image(true_image, sigma=1.5)
    observed = blurred + 0.02 * rng.standard_normal(blurred.shape)
    return true_image, observed


def test_blur_image_preserves_total_brightness():
    """A Gaussian blur redistributes intensity but shouldn't create or destroy it --
    each row of the underlying blur matrix sums to 1 by construction.
    """
    n = 16
    image = np.zeros((n, n))
    image[5:10, 5:10] = 2.0
    blurred = blur_image(image, sigma=1.0)
    assert blurred.sum() == pytest.approx(image.sum(), abs=1e-8)


def test_blur_image_smooths_a_sharp_edge():
    n = 16
    image = np.zeros((n, n))
    image[:, 8:] = 1.0
    blurred = blur_image(image, sigma=2.0)
    # the sharp transition (max adjacent difference) should be softened by blurring
    sharp_edge = np.max(np.abs(np.diff(image[8])))
    blurred_edge = np.max(np.abs(np.diff(blurred[8])))
    assert blurred_edge < sharp_edge


def test_well_regularized_deblurring_beats_the_blurry_observation():
    """The actual inverse-problem claim: recovering a sharper image from the blurry,
    noisy observation, not just returning the observation itself.
    """
    true_image, observed = _synthetic_scene()
    observed_mse = np.mean((observed - true_image) ** 2)

    problem = deblurring_problem(observed, sigma=1.5, alpha=0.01)
    result = bfgs(problem, max_iter=3000)
    recovered = result.x.reshape(true_image.shape)
    recovered_mse = np.mean((recovered - true_image) ** 2)

    assert recovered_mse < 0.7 * observed_mse


def test_too_little_regularization_amplifies_noise_past_the_observation():
    """The other half of the bias-variance story: an inverse problem this ill-posed
    isn't automatically improved by 'just optimize harder' -- with too little
    regularization, noise in the near-annihilated high frequencies gets amplified
    without bound, landing *worse* than simply keeping the blurry observation.
    """
    true_image, observed = _synthetic_scene()
    observed_mse = np.mean((observed - true_image) ** 2)

    problem = deblurring_problem(observed, sigma=1.5, alpha=1e-4)
    result = bfgs(problem, max_iter=1000)
    recovered = result.x.reshape(true_image.shape)
    recovered_mse = np.mean((recovered - true_image) ** 2)

    assert recovered_mse > observed_mse
