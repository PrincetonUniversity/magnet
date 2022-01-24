import numpy as np
import torch
from magnet.constants import materials
from magnet.net import model, model_lstm


def default_units(prop):  # Probably we are not going to need the default units
    prop = prop.lower().strip()
    return {
        'frequency': 'Hz',
        'flux_density': 'T',
        'power_loss': '[W/m^3]',
        'outlier_factor': '%'
    }[prop]


def plot_label(prop):
    prop = prop.lower().strip()
    return {
        'frequency_khz': 'Frequency [kHz]',
        'flux_density_mt': 'AC Flux Density [mT]',
        'power_loss_kw/m3': 'Power Loss [kW/m^3]',
        'frequency': 'Frequency [Hz]',
        'flux_density': 'AC Flux Density [T]',
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


def core_loss_iGSE_arbitrary(freq, flux, duty, k_i=None, alpha=None, beta=None, material=None,
                             n_interval=10_000):

    """
    Calculate magnetic core loss using iGSE

    :param freq: Frequency of excitation waveform (Hz)
    :param flux: Relative Flux Density (T) in a single waveform cycle, as an ndarray
    :param duty: Fractional time wrt time period, in [0, 1], in a single waveform cycle, as an ndarray
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
    flux_delta = np.amax(flux) - np.amin(flux)
    time, dt = np.linspace(start=0, stop=period, num=n_interval, retstep=True)
    B = np.interp(time, np.multiply(duty, period), flux)
    dBdt = np.gradient(B, dt)
    core_loss = freq * np.trapz(k_i * (np.abs(dBdt) ** alpha) * (flux_delta ** (beta - alpha)), time)

    return core_loss


def core_loss_iGSE_sinusoidal(freq, flux, duty=None, k_i=None, alpha=None, beta=None, material=None, dc_bias=0, n_interval=10_000):
    # Here duty is not needed, but it is convenient to call the function recursively
    if material is not None:
        assert material in materials, f'Material {material} not found'
        k_i, alpha, beta = materials[material]

    frac_time = np.linspace(0, 1, n_interval)
    flux_list = dc_bias + flux * np.sin(2 * np.pi * frac_time)

    return core_loss_iGSE_arbitrary(freq, flux_list, frac_time, k_i=k_i, alpha=alpha, beta=beta, material=material,
                                    n_interval=n_interval)


def core_loss_iGSE_triangular(freq, flux, duty, k_i=None, alpha=None, beta=None, material=None, dc_bias=0):
    # Here duty is the fraction of time where the current rises
    if material is not None:
        assert material in materials, f'Material {material} not found'
        k_i, alpha, beta = materials[material]

    assert 0 <= duty <= 1.0, 'Duty ratio should be between 0 and 1'
    frac_time = np.array([0, duty, 1])
    flux_list = dc_bias + np.array([-flux, flux, -flux])

    return core_loss_iGSE_arbitrary(freq, flux_list, frac_time, k_i=k_i, alpha=alpha, beta=beta, material=material)


def core_loss_iGSE_trapezoidal(freq, flux, duty, k_i=None, alpha=None, beta=None, material=None, dc_bias=0):
    # Here duty is a vector
    if material is not None:
        assert material in materials, f'Material {material} not found'
        k_i, alpha, beta = materials[material]

    assert len(duty) == 3, 'Please specify 3 values as the Duty Ratios'
    assert np.all((0 <= np.array(duty)) & (np.array(duty) <= 1)), 'Duty ratios should be between 0 and 1'

    frac_time = np.array([0, duty[0], duty[0] + duty[2], 1 - duty[2], 1])

    if duty[0] > duty[1]:
        # Since Bpk is proportional to the voltage, and the voltage is proportional to (1-dp+dN) times the dp
        BPplot = flux
        BNplot = -BPplot * ((-1 - duty[0] + duty[1]) * duty[1]) / (
                    (1 - duty[0] + duty[1]) * duty[0])  # proportional to (-1-dp+dN)*dn
    else:
        BNplot = flux  # proportional to (-1-dP+dN)*dN
        BPplot = -BNplot * ((1 - duty[0] + duty[1]) * duty[0]) / (
                    (-1 - duty[0] + duty[1]) * duty[1])  # proportional to (1-dP+dN)*dP

    flux_list = dc_bias + np.array([-BPplot, BPplot, BNplot, -BNplot, -BPplot])

    return core_loss_iGSE_arbitrary(freq, flux_list, frac_time, k_i=k_i, alpha=alpha, beta=beta, material=material)


def core_loss_ML_sinusoidal(freq, flux, material, duty=None):
    nn = model(material=material, waveform='Sinusoidal')
    core_loss = 10.0 ** nn(
        torch.from_numpy(
            np.array([
                np.log10(float(freq)),
                np.log10(float(flux))
            ])
        )
    ).item()
    return core_loss


def core_loss_ML_triangular(freq, flux, duty, material):
    nn = model(material=material, waveform='Trapezoidal')
    core_loss = 10.0 ** nn(
        torch.from_numpy(
            np.array([
                np.log10(float(freq)),
                np.log10(float(flux)),
                duty,
                0,
                1-duty,
                0
            ])
        )
    ).item()
    return core_loss


def core_loss_ML_trapezoidal(freq, flux, duty, material):
    nn = model(material=material, waveform='Trapezoidal')
    core_loss = 10.0 ** nn(
        torch.from_numpy(
            np.array([
                np.log10(float(freq)),
                np.log10(float(flux)),
                duty[0],
                duty[2],
                duty[1],
                duty[2]
            ])
        )
    ).item()
    return core_loss


def core_loss_ML_arbitrary(material, freq, flux, duty):
    nn = model_lstm(material=material)
    Num = 100
    period = 1/freq
    time = np.linspace(start=0, stop=period, num=Num)
    flux_interpolated = np.interp(time, np.multiply(duty, period), flux)
    
    # Manually get rid of the dc-bias in the flux, for now
    # print(np.average(flux_interpolated))
    flux_interpolated = flux_interpolated - np.average(flux_interpolated)
    
    flux_interpolated = torch.from_numpy(flux_interpolated).view(-1, Num, 1)
    freq = torch.from_numpy(np.asarray(np.log10(freq))).view(-1, 1)
    core_loss = 10.0 ** nn(flux_interpolated,freq).item()
    return core_loss


def loss(waveform, algorithm, **kwargs):
    if algorithm == 'Machine Learning':
        algorithm = 'ML'
    assert waveform.lower() in ('sinusoidal', 'triangular', 'trapezoidal', 'arbitrary'), f'Unknown waveform {waveform}'
    assert algorithm in ('iGSE', 'ML'), f'Unknown algorithm {algorithm}'

    fn = globals()[f'core_loss_{algorithm}_{waveform.lower()}']
    return fn(**kwargs)