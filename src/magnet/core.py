import numpy as np
from magnet.constants import materials


def core_loss_iGSE(freq, flux, duty, k_i, alpha, beta, n_interval=10001):
    """
    Calculate magnetic core loss using iGSE

    :param freq: Frequency of excitation waveform (Hz)
    :param flux: Relative Flux Density (mT) in a single waveform cycle, as an ndarray
    :param duty: Duty (%) in a single waveform cycle, as an ndarray
    :param k_i: Stenmetz coefficient k_i
    :param alpha: Steinmetz coefficient alpha
    :param beta: Steinmetz coefficient beta
    :param n_interval: No. of intervals to use to solve iGSE using trapezoidal rule
    :return: Core loss (kW/m^3)
    """
    period = 1 / freq
    flux_delta = np.amax(flux) - np.amin(flux)
    time, dt = np.linspace(start=0, stop=period, num=n_interval, retstep=True)
    B = np.interp(time, np.multiply(duty, period), flux)
    dBdt = np.gradient(B, dt)
    core_loss = freq * np.trapz(k_i * (np.abs(dBdt) ** alpha) * (flux_delta ** (beta - alpha)), time)

    return core_loss


def core_loss(material, freq, flux, duty, algorithm='iGSE'):
    """
    Calculate magnetic core loss of a material

    :param material: One of N27/N87/N49
    :param freq: Frequency of excitation waveform (Hz)
    :param flux: Relative Flux Density (mT) in a single waveform cycle, as an ndarray
    :param duty: Duty (%) in a single waveform cycle, as an ndarray
    :param algorithm: Only 'iGSE' supported for now
    :return: Core loss (kW/m^3)
    """
    assert material in materials, f'Material {material} not found'
    k_i, alpha, beta = materials[material]
    core_loss_fn = {
        'iGSE': core_loss_iGSE
    }[algorithm]

    return core_loss_fn(freq, flux, duty, k_i, alpha, beta)