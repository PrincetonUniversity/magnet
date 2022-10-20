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
def load_dataframe(material, freq_min=None, freq_max=None, flux_min=None, flux_max=None, bias=None,
                   duty_p=None, duty_n=None, temp=None):
    bias_margin = 4
    duty_margin = 0.04
    temp_margin = 4

    with path('magnet.data', f'{material}_database.h5') as h5file:
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
        if bias is None:
            bias_max = data['DC_Bias'].max()
            bias_min = data['DC_Bias'].min()
        else:
            bias_max = bias + bias_margin
            bias_min = bias - bias_margin
        if duty_p is None:
            dp_max = data['Duty_P'].max()
            dp_min = data['Duty_P'].min()
        else:
            dp_max = duty_p + duty_margin
            dp_min = duty_p - duty_margin
        if duty_n is None:
            dn_max = data['Duty_N'].max()
            dn_min = data['Duty_N'].min()
        else:
            dn_max = duty_n + duty_margin
            dn_min = duty_n - duty_margin
        if temp is None:
            temp_max = data['DC_Bias'].max()
            temp_min = data['DC_Bias'].min()
        else:
            temp_max = temp + temp_margin
            temp_min = temp - temp_margin

        query = f'({freq_min} <= Frequency <= {freq_max}) & ' \
                f'({flux_min} <= Flux_Density <= {flux_max}) & ' \
                f'({bias_min} <= DC_Bias <= {bias_max}) & ' \
                f'({dp_min} <= Duty_P <= {dp_max}) & ' \
                f'({dn_min} <= Duty_N <= {dn_max}) & ' \
                f'({temp_min} <= Temperature <= {temp_max})'

        data = data.query(query)
    return data


def load_metadata(material):
    with path('magnet.data', f'{material}_database.h5') as h5file:
        data, metadata = h5_load(h5file)
    return metadata


def loss_interpolated(waveform, algorithm, **kwargs):
    assert waveform.lower() in ('sinusoidal', 'triangular', 'trapezoidal', 'arbitrary'), f'Unknown waveform {waveform}'
    assert algorithm in ('DI', 'SI'), f'Unknown algorithm {algorithm}'

    fn = globals()[f'core_loss_{algorithm}_{waveform.lower()}']
    return fn(**kwargs)
