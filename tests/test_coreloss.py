import numpy as np
from pytest import approx
from numpy.random import default_rng
from magnet.core import core_loss_iGSE


def test_coreloss1():
    flux = np.array([-4.54545455, 5.45454545, 15.45454545, 5.45454545, 15.45454545, 25.45454545, -14.54545455,
                     -34.54545455, 5.45454545, -14.54545455, -4.54545455])
    duty = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    loss = core_loss_iGSE(10_000, flux, duty, k_i=4.88e-10, alpha=1.09, beta=2.44)

    assert loss == approx(0.002811408865609484)


def test_coreloss2():
    rng = default_rng(515242)
    flux = 100 * rng.random(11)
    duty = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    loss = core_loss_iGSE(10_000, flux, duty, k_i=4.88e-10, alpha=1.09, beta=2.44)

    assert loss == approx(0.030457525717187316)
