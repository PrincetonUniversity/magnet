import numpy as np
import torch

from magnet.constants import materials
from magnet.net import model


def default_units(prop):  # Probably we are not going to need the default units
    prop = prop.lower().strip()
    return {
        'frequency': 'Hz',
        'flux_density': 'T',
        # 'duty_ratio': '',
        'power_loss': '[W/m^3]',
        'outlier_factor': '%'
    }[prop]


def plot_label(prop):
    prop = prop.lower().strip()
    return {
        'frequency_khz': 'Frequency [kHz]',
        'flux_density_mt': 'AC Flux Density Amplitude [mT]',
        'power_loss_kw/m3': 'Power Loss [kW/m^3]',
        'frequency': 'Frequency [Hz]',
        'flux_density': 'AC Flux Density Amplitude [T]',
        'power_loss': 'Power Loss [W/m^3]',
        'duty_ratio': 'Duty Ratio',
        'outlier_factor': 'Outlier Factor [%]'
    }[prop]


def plot_title(prop):
    prop = prop.lower().strip()
    return {
        'frequency_khz': 'Frequency',
        'flux_density_mt': 'Flux Density',
        'power_loss_kw/m3': 'Power Loss',
        'frequency': 'Frequency',
        'flux_density': 'Flux Density',
        'power_loss': 'Power Loss',
        'outlier_factor': 'Outlier Factor'
    }[prop]


def core_loss_iGSE_arbitrary(freq, flux_list, frac_time, k_i=None, alpha=None, beta=None, material=None, n_interval=10_000):

    """
    Calculate magnetic core loss using iGSE

    :param freq: Frequency of excitation waveform (Hz)
    :param flux_list: Relative Flux Density (T) in a single waveform cycle, as an ndarray
    :param frac_time: Fractional time wrt time period, in [0, 1], in a single waveform cycle, as an ndarray
    :param k_i: Steinmetz coefficient k_i
    :param alpha: Steinmetz coefficient alpha
    :param beta: Steinmetz coefficient beta
    :param material: Name of material. If specified, k_i/alpha/beta are ignored.
    :param n_interval: No. of intervals to use to solve iGSE using trapezoidal rule
    :return: Core loss (W/m^3)
    """
    if material is not None:
        assert material in materials, f'Material {material} not found'
        k_i, alpha, beta = materials[material]

    period = 1 / freq
    flux_delta = np.amax(flux_list) - np.amin(flux_list)
    time, dt = np.linspace(start=0, stop=period, num=n_interval, retstep=True)
    B = np.interp(time, np.multiply(frac_time, period), flux_list)
    dBdt = np.gradient(B, dt)
    core_loss = freq * np.trapz(k_i * (np.abs(dBdt) ** alpha) * (flux_delta ** (beta - alpha)), time)

    return core_loss


def core_loss_iGSE_sine(freq, flux, k_i=None, alpha=None, beta=None, material=None, dc_bias=0, n_interval=10_000):
    if material is not None:
        assert material in materials, f'Material {material} not found'
        k_i, alpha, beta = materials[material]

    frac_time = np.linspace(0, 1, n_interval)
    flux_list = dc_bias + flux * np.sin(2 * np.pi * frac_time)

    return core_loss_iGSE_arbitrary(freq, flux_list, frac_time, k_i=k_i, alpha=alpha, beta=beta, material=material, n_interval=n_interval)


def core_loss_ML_sine(freq, flux, material):
    nn = model(material=material,waveform='Sinusoidal')
    core_loss = 10.0 ** nn(
        torch.from_numpy(
            np.array([
                np.log10(float(freq)),
                np.log10(float(flux))
            ])
        )
    ).item()
    return core_loss


def core_loss_iGSE_triangle(freq, flux, duty_ratio, k_i=None, alpha=None, beta=None, material=None, dc_bias=0):
    if material is not None:
        assert material in materials, f'Material {material} not found'
        k_i, alpha, beta = materials[material]

    assert 0 <= duty_ratio <= 1.0, 'Duty ratio should be between 0 and 1'
    frac_time = np.array([0, duty_ratio, 1])
    flux_list = dc_bias + np.array([-flux, flux, -flux])

    return core_loss_iGSE_arbitrary(freq, flux_list, frac_time, k_i=k_i, alpha=alpha, beta=beta, material=material)


def core_loss_ML_triangle(freq, flux, duty_ratio, material):
    nn = model(material=material,waveform='Trapezoidal')
    core_loss = 10.0 ** nn(
        torch.from_numpy(
            np.array([
                np.log10(float(freq)),
                np.log10(float(flux)),
                duty_ratio,
                0,
                1-duty_ratio,
                0
            ])
        )
    ).item()
    return core_loss


def core_loss_iGSE_trapezoid(freq, flux, duty_ratios, k_i=None, alpha=None, beta=None, material=None, dc_bias=0):
    if material is not None:
        assert material in materials, f'Material {material} not found'
        k_i, alpha, beta = materials[material]

    assert len(duty_ratios) == 3, 'Please specify 3 values as the Duty Ratios'
    assert np.all((0 <= np.array(duty_ratios)) & (np.array(duty_ratios) <= 1)), 'Duty ratios should be between 0 and 1'

    frac_time = np.array([0, duty_ratios[0], duty_ratios[0]+duty_ratios[2], 1-duty_ratios[2], 1])
    if duty_ratios[0]>duty_ratios[1] :
        BPplot=flux # Since Bpk is proportional to the voltage, and the voltage is proportional to (1-dp+dN) times the dp
        BNplot=-BPplot*((-1-duty_ratios[0]+duty_ratios[1])*duty_ratios[1])/((1-duty_ratios[0]+duty_ratios[1])*duty_ratios[0]) # proportional to (-1-dp+dN)*dn
    else :
        BNplot=flux # proportional to (-1-dP+dN)*dN
        BPplot=-BNplot*((1-duty_ratios[0]+duty_ratios[1])*duty_ratios[0])/((-1-duty_ratios[0]+duty_ratios[1])*duty_ratios[1]) # proportional to (1-dP+dN)*dP
    flux_list = dc_bias + np.array([-BPplot,BPplot,BNplot,-BNplot,-BPplot])
    
    return core_loss_iGSE_arbitrary(freq, flux_list, frac_time, k_i=k_i, alpha=alpha, beta=beta, material=material)


def core_loss_ML_trapezoid(freq, flux, duty_ratios, material):
    nn = model(material=material,waveform='Trapezoidal')
    core_loss = 10.0 ** nn(
        torch.from_numpy(
            np.array([
                np.log10(float(freq)),
                np.log10(float(flux)),
                duty_ratios[0],
                duty_ratios[2],
                duty_ratios[1],
                duty_ratios[2]
            ])
        )
    ).item()
    return core_loss


def core_loss_ML_arbitrary(material, freq, flux_list, frac_time):
    return 0
    #raise NotImplementedError


def loss(waveform, algorithm, **kwargs):
    if algorithm == 'Machine Learning':
        algorithm = 'ML'
    assert waveform in ('sine', 'triangle', 'trapezoid', 'arbitrary'), f'Unknown waveform {waveform}'
    assert algorithm in ('iGSE', 'ML' ,'Ref'), f'Unknown algorithm {algorithm}'

    fn = globals()[f'core_loss_{algorithm}_{waveform}']
    return fn(**kwargs)
