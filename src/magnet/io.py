from importlib.resources import path
import pandas as pd
import streamlit as st

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
def load_dataframe(material, excitation, freq_min=None, freq_max=None, flux_min=None, flux_max=None, duty_ratios=None,
              duty_ratio_margin=0):

    excitation = excitation.lower()
    with path('magnet.data', f'{material}_{excitation}.h5') as h5file:
        data, metadata = h5_load(h5file)

        if freq_min is None:
            freq_min = data['Frequency'].min()
        if freq_max is None:
            freq_max = data['Frequency'].max()
        if flux_min is None:
            freq_min = data['Flux_Density'].min()
        if flux_max is None:
            flux_max = data['Flux_Density'].max()

        query = f'({freq_min} <= Frequency <= {freq_max}) & ({flux_min} <= Flux_Density <= {flux_max})'
        if duty_ratios is not None:
            query += '& (' + \
                     '|'.join([f'({d - duty_ratio_margin} <= Duty_Ratio <= {d + duty_ratio_margin})' for d in duty_ratios]) + \
                     ')'

        data = data.query(query)
    return data
