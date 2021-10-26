import numpy as np
from pytest import approx
from numpy.random import default_rng
from magnet.core import core_loss_iGSE_arbitrary, core_loss_iGSE_sine, core_loss_iGSE_sawtooth, core_loss_iGSE_trapezoid


def test_coreloss_random1():
    flux = np.array([-4.54545455, 5.45454545, 15.45454545, 5.45454545, 15.45454545, 25.45454545, -14.54545455,
                     -34.54545455, 5.45454545, -14.54545455, -4.54545455])
    duty = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    loss = core_loss_iGSE_arbitrary(10_000, flux, duty, k_i=4.88e-10, alpha=1.09, beta=2.44)

    assert loss == approx(0.8216212137732434)


def test_coreloss_random2():
    rng = default_rng(515242)
    flux = 100 * rng.random(11)
    duty = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    loss = core_loss_iGSE_arbitrary(10_000, flux, duty, k_i=4.88e-10, alpha=1.09, beta=2.44)

    assert loss == approx(1.6603434050633004)


def test_coreless_sine1():
    freq = 10_000
    flux_p2p = 182.0
    dc_bias = np.random.random() * 10  # A DC bias should have no effect on core loss per time period

    loss = core_loss_iGSE_sine(freq, flux_p2p, k_i=4.88e-10, alpha=1.09, beta=2.44, dc_bias=dc_bias)
    assert loss == approx(7.889979705299859)


def test_coreless_triangle1():
    freq = 10_000
    flux_p2p = 182.0
    duty_ratio = 0.3
    dc_bias = 6.45

    loss = core_loss_iGSE_sawtooth(freq, flux_p2p, duty_ratio, k_i=4.88e-10, alpha=1.09, beta=2.44, dc_bias=dc_bias)
    assert loss == approx(7.848518722152074)


def test_coreless_trapezoid1():
    freq = 10_000
    flux_p2p = 182.0
    duty_ratios = [0.2, 0.4, 0.8]
    dc_bias = 6.45

    loss = core_loss_iGSE_trapezoid(freq, flux_p2p, duty_ratios, k_i=4.88e-10, alpha=1.09, beta=2.44, dc_bias=dc_bias)
    assert loss == approx(8.195712329473336)