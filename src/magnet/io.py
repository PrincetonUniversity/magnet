import pandas as pd
import streamlit as st

from importlib.resources import path


# Adapted from
#   https://stackoverflow.com/questions/29129095/save-additional-attributes-in-pandas-dataframe/29130146#29130146
def h5_store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('dataframe', df)
    store.get_storer('dataframe').attrs.metadata = kwargs
    store.close()


def h5_load(filename):
    with pd.HDFStore(filename) as store:
        data = store['dataframe']
        metadata = store.get_storer('dataframe').attrs.metadata
        return data, metadata


@st.cache
def load_dataframe(material, excitation, freq_min=None, freq_max=None, flux_min=None, flux_max=None,
                   duty_1=None, duty_3=None, out_max=None):
    duty_margin = 0.01
    excitation = excitation.lower()
    with path('magnet.data', f'{material}_{excitation}.h5') as h5file:
        data, metadata = h5_load(h5file)

        data['Frequency_kHz'] = data['Frequency'] / 1e3
        data['Flux_Density_mT'] = data['Flux_Density'] * 1e3
        data['Power_Loss_kW/m3'] = data['Power_Loss'] / 1e3

        if freq_min is None:
            freq_min = data['Frequency'].min()
        if freq_max is None:
            freq_max = data['Frequency'].max()
        if flux_min is None:
            flux_min = data['Flux_Density'].min()
        if flux_max is None:
            flux_max = data['Flux_Density'].max()
        if out_max is None:
            out_max = data['Outlier_Factor'].max()
        if duty_1 is None:
            d1_max = data['Duty_1'].max()
            d1_min = data['Duty_1'].min()
        else:
            d1_max = duty_1 + duty_margin
            d1_min = duty_1 - duty_margin
        if duty_3 is None:
            d3_max = data['Duty_3'].max()
            d3_min = data['Duty_3'].min()
        else:
            d3_max = duty_3 + duty_margin
            d3_min = duty_3 - duty_margin

        query = f'({freq_min} <= Frequency <= {freq_max}) & ' \
                f'({flux_min} <= Flux_Density <= {flux_max}) & ' \
                f'({d1_min} <= Duty_1 <= {d1_max}) & ' \
                f'({d3_min} <= Duty_3 <= {d3_max}) & ' \
                f'({-out_max} <= Outlier_Factor <= {out_max})'

        data = data.query(query)
    return data


@st.cache
def load_dataframe_datasheet(material, freq_min=None, freq_max=None, flux_min=None, flux_max=None,
                             temp=None):
    temp_margin = 1.0
    with path('magnet.data', f'{material}_datasheet.h5') as h5file:
        data, metadata = h5_load(h5file)

        data['Frequency_kHz'] = data['Frequency'] / 1e3
        data['Flux_Density_mT'] = data['Flux_Density'] * 1e3
        data['Power_Loss_kW/m3'] = data['Power_Loss'] / 1e3

        if freq_min is None:
            freq_min = data['Frequency'].min()
        if freq_max is None:
            freq_max = data['Frequency'].max()
        if flux_min is None:
            freq_min = data['Flux_Density'].min()
        if flux_max is None:
            flux_max = data['Flux_Density'].max()
        if temp is None:
            temp_max = data['Temperature'].max()
            temp_min = data['Temperature'].min()
        else:
            temp_max = temp + temp_margin
            temp_min = temp - temp_margin

        query = f'({freq_min} <= Frequency <= {freq_max}) & ' \
                f'({flux_min} <= Flux_Density <= {flux_max}) & ' \
                f'({temp_min} <= Temperature <= {temp_max})'

        data = data.query(query)
    return data


@st.cache
def load_dataframe_point(material, excitation, freq, flux):
    freq_margin = 500.0
    flux_margin = 5e-4
    with path('magnet.data', f'{material}_{excitation}_interpolated.h5') as h5file:
        data, metadata = h5_load(h5file)
        query = f'({freq-freq_margin} <= Frequency <= {freq+freq_margin}) & ' \
                f'({flux-flux_margin} <= Flux_Density <= {flux+flux_margin})'
        data = data.query(query)
    return data


def load_metadata(material, excitation):
    excitation = excitation.lower()
    with path('magnet.data', f'{material}_{excitation}.h5') as h5file:
        data, metadata = h5_load(h5file)
    return metadata


# Core losses for interpolation: cannot be in core.py due to errors when testing
def core_loss_DI_sinusoidal(freq, flux, material):
    excitation_read = 'datasheet'
    df = load_dataframe_point(material, excitation_read, freq, flux)
    if not df.empty:
        core_loss = float(df['Power_Loss'])

    else:
        core_loss = 0.0  # Not interpolable
    return core_loss


def core_loss_SI_sinusoidal(freq, flux, material):
    excitation_read = 'sinusoidal'
    df = load_dataframe_point(material, excitation_read, freq, flux)
    if not df.empty:
        core_loss = float(df['Power_Loss'])

    else:
        core_loss = 0.0  # Not interpolable
    return core_loss


def loss_interpolated(waveform, algorithm, **kwargs):
    assert waveform.lower() in ('sinusoidal', 'triangular', 'trapezoidal', 'arbitrary'), f'Unknown waveform {waveform}'
    assert algorithm in ('DI', 'SI'), f'Unknown algorithm {algorithm}'

    fn = globals()[f'core_loss_{algorithm}_{waveform.lower()}']
    return fn(**kwargs)