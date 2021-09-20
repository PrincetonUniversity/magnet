import numpy as np
from pmagnet.constants import materials


def core_loss_iGSE(freq, flux_delta, flux, duty, k_i, alpha, beta, n_interval=10001):
    """
    Calculate magnetic core loss using iGSE

    :param freq: frequency of excitation waveform
    :param flux_delta:
    :param flux:
    :param duty:
    :param k_i:
    :param alpha:
    :param beta:
    :param n_interval:
    :return:
    """
    period = 1 / freq
    time, dt = np.linspace(start=0, stop=period, num=n_interval, retstep=True)
    B = np.interp(time, np.multiply(duty, period), flux)
    dBdt = np.gradient(B, dt)
    core_loss = freq * np.trapz(k_i * (np.abs(dBdt) ** alpha) * (flux_delta ** (beta - alpha)), time)

    return core_loss


def core_loss(material, freq, flux_delta, flux, duty, algorithm='iGSE'):
    """
    Calculate magnetic core loss of a material

    :param material: One of N27/N87/N49
    :param freq: Frequency of excitation waveform
    :param flux_delta:
    :param flux: Waveform Pattern Relative Flux Density (mT)
    :param duty: Waveform Pattern Duty in a Cycle (%)
    :param algorithm: Only 'iGSE' supported for now
    :return:
    """
    assert material in materials, f'Material {material} not found'
    k_i, alpha, beta = materials[material]
    core_loss_fn = {'iGSE': core_loss_iGSE}[algorithm]
    return core_loss_fn(freq, flux_delta, flux, duty, k_i, alpha, beta)