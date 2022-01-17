import numpy as np
import torch
import math

from magnet.constants import materials
from magnet.net import model, model_lstm
from magnet.io import load_dataframe_nearby, load_dataframe_datasheet_nearby

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


def waveform_shorts(prop):
    prop = prop.lower().strip()
    return {
        'sinusoidal': 'sine',
        'triangular': 'triangle',
        'trapezoidal': 'trapezoid',
        'arbitrary': 'arbitrary',
        'simulated': 'simulated'
    }[prop]


def core_loss_iGSE_arbitrary(freq, flux_list, frac_time, k_i=None, alpha=None, beta=None, material=None,
                             n_interval=10_000):

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

    return core_loss_iGSE_arbitrary(freq, flux_list, frac_time, k_i=k_i, alpha=alpha, beta=beta, material=material,
                                    n_interval=n_interval)


def core_loss_ML_sine(freq, flux, material):
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


def core_loss_DI_sine(freq, flux, material):
    df = load_dataframe_datasheet_nearby(material, freq, flux, 25.0)
    if df['Power_Loss'].size == 4:  # Only calculate the losses if the point is interpolable (surrounded by 4 points)
        # log interpolation (linear) of the four points.
        log_flux = math.log10(flux)
        log_freq = math.log10(freq)
        power_loss = df['Power_Loss']
        index_freq_min = df[df['Frequency'] < freq].index.tolist()
        index_flux_min = df[df['Flux_Density'] < flux].index.tolist()
        index_freq_max = df[df['Frequency'] > freq].index.tolist()
        index_flux_max = df[df['Flux_Density'] > flux].index.tolist()
        log_loss_f_max_b_max = math.log10(float(power_loss[np.intersect1d(index_freq_max, index_flux_max)]))
        log_loss_f_max_b_min = math.log10(float(power_loss[np.intersect1d(index_freq_max, index_flux_min)]))
        log_loss_f_min_b_max = math.log10(float(power_loss[np.intersect1d(index_freq_min, index_flux_max)]))
        log_loss_f_min_b_min = math.log10(float(power_loss[np.intersect1d(index_freq_min, index_flux_min)]))
        log_flux_max = math.log10(df['Flux_Density'].max())
        log_flux_min = math.log10(df['Flux_Density'].min())
        log_freq_max = math.log10(df['Frequency'].max())
        log_freq_min = math.log10(df['Frequency'].min())
        log_loss_f_max = log_loss_f_max_b_min + (log_loss_f_max_b_max - log_loss_f_max_b_min) / (log_flux_max - log_flux_min) * (log_flux - log_flux_min)
        log_loss_f_min = log_loss_f_min_b_min + (log_loss_f_min_b_max - log_loss_f_min_b_min) / (log_flux_max - log_flux_min) * (log_flux - log_flux_min)
        log_loss = log_loss_f_min + (log_loss_f_max - log_loss_f_min) / (log_freq_max - log_freq_min) * (log_freq - log_freq_min)
        core_loss = 10 ** log_loss
    else:
        core_loss = 0.0  # Not interpolable
    return core_loss

def core_loss_SI_sine(freq, flux, material):
    df = load_dataframe_nearby(material, freq, flux)
    if df['Power_Loss'].size == 4:  # Only calculate the losses if the point is interpolable (surrounded by 4 points)
        # log interpolation (linear) of the four points.
        log_flux = math.log10(flux)
        log_freq = math.log10(freq)
        power_loss = df['Power_Loss']
        index_freq_min = df[df['Frequency'] < freq].index.tolist()
        index_flux_min = df[df['Flux_Density'] < flux].index.tolist()
        index_freq_max = df[df['Frequency'] > freq].index.tolist()
        index_flux_max = df[df['Flux_Density'] > flux].index.tolist()
        log_loss_f_max_b_max = math.log10(float(power_loss[np.intersect1d(index_freq_max, index_flux_max)]))
        log_loss_f_max_b_min = math.log10(float(power_loss[np.intersect1d(index_freq_max, index_flux_min)]))
        log_loss_f_min_b_max = math.log10(float(power_loss[np.intersect1d(index_freq_min, index_flux_max)]))
        log_loss_f_min_b_min = math.log10(float(power_loss[np.intersect1d(index_freq_min, index_flux_min)]))
        log_flux_max = math.log10(df['Flux_Density'].max())
        log_flux_min = math.log10(df['Flux_Density'].min())
        log_freq_max = math.log10(df['Frequency'].max())
        log_freq_min = math.log10(df['Frequency'].min())
        log_loss_f_max = log_loss_f_max_b_min + (log_loss_f_max_b_max - log_loss_f_max_b_min) / (log_flux_max - log_flux_min) * (log_flux - log_flux_min)
        log_loss_f_min = log_loss_f_min_b_min + (log_loss_f_min_b_max - log_loss_f_min_b_min) / (log_flux_max - log_flux_min) * (log_flux - log_flux_min)
        log_loss = log_loss_f_min + (log_loss_f_max - log_loss_f_min) / (log_freq_max - log_freq_min) * (log_freq - log_freq_min)
        core_loss = 10 ** log_loss
    else:
        core_loss = 0.0  # Not interpolable
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
    nn = model(material=material, waveform='Trapezoidal')
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

    if duty_ratios[0] > duty_ratios[1]:
        # Since Bpk is proportional to the voltage, and the voltage is proportional to (1-dp+dN) times the dp
        BPplot = flux
        BNplot = -BPplot*((-1-duty_ratios[0]+duty_ratios[1])*duty_ratios[1])/((1-duty_ratios[0]+duty_ratios[1])*duty_ratios[0]) # proportional to (-1-dp+dN)*dn
    else:
        BNplot = flux  # proportional to (-1-dP+dN)*dN
        BPplot = -BNplot*((1-duty_ratios[0]+duty_ratios[1])*duty_ratios[0])/((-1-duty_ratios[0]+duty_ratios[1])*duty_ratios[1]) # proportional to (1-dP+dN)*dP
    
    flux_list = dc_bias + np.array([-BPplot, BPplot, BNplot, -BNplot, -BPplot])

    
    return core_loss_iGSE_arbitrary(freq, flux_list, frac_time, k_i=k_i, alpha=alpha, beta=beta, material=material)


def core_loss_ML_trapezoid(freq, flux, duty_ratios, material):
    nn = model(material=material, waveform='Trapezoidal')
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
    nn = model_lstm(material=material)
    Ts = 10e-9
    Num = 5000
    time = np.linspace(start=Ts, stop=Ts*Num, num=Num)
    period = 1/freq
    flux = np.interp(np.remainder(time, period), np.multiply(frac_time, period), flux_list)
    flux = torch.from_numpy(flux).view(-1, 5000, 1)
    core_loss = 10.0 ** nn(flux).item()
    return core_loss


def loss(waveform, algorithm, **kwargs):
    if algorithm == 'Machine Learning':
        algorithm = 'ML'
    assert waveform in ('sine', 'triangle', 'trapezoid', 'arbitrary'), f'Unknown waveform {waveform}'
    assert algorithm in ('iGSE', 'ML', 'DI', 'SI'), f'Unknown algorithm {algorithm}'

    fn = globals()[f'core_loss_{algorithm}_{waveform}']
    return fn(**kwargs)


def cycle_points_sine(point):
    cycle_list = np.linspace(0, 1, point)
    flux_list = np.sin(np.multiply(cycle_list, np.pi * 2))
    volt_list = np.cos(np.multiply(cycle_list, np.pi * 2))
    return [cycle_list, flux_list, volt_list]


def cycle_points_trap(duty_p, duty_n, duty_0):
    if duty_p > duty_n:
        volt_p = (1 - (duty_p - duty_n)) / -(-1 - (duty_p - duty_n))
        volt_0 = - (duty_p - duty_n) / -(-1 - (duty_p - duty_n))
        volt_n = -1  # The negative voltage is maximum
        b_p = 1  # Bpk is proportional to the voltage, which is is proportional to (1-dp+dN) times the dp
        b_n = -(-1 - duty_p + duty_n) * duty_n / ((1 - duty_p + duty_n) * duty_p)  # Prop to (-1-dp+dN)*dn
    else:
        volt_p = 1  # The positive voltage is maximum
        volt_0 = - (duty_p - duty_n) / (1 - (duty_p - duty_n))
        volt_n = (-1 - (duty_p - duty_n)) / (1 - (duty_p - duty_n))
        b_n = 1  # Proportional to (-1-dP+dN)*dN
        b_p = -(1 - duty_p + duty_n) * duty_p / ((-1 - duty_p + duty_n) * duty_n)  # Prop to (1-dP+dN)*dP
    cycle_list = [0, 0, duty_p, duty_p, duty_p + duty_0, duty_p + duty_0, 1 - duty_0, 1 - duty_0, 1]
    flux_list = [-b_p, -b_p, b_p, b_p, b_n, b_n, -b_n, -b_n, -b_p]
    volt_list = [volt_0, volt_p, volt_p, volt_0, volt_0, volt_n, volt_n, volt_0, volt_0]
    return [cycle_list, flux_list, volt_list]