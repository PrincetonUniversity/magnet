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


def load_dataframe_short(material, excitation, freq_min=None, freq_max=None, flux_min=None, flux_max=None,
                         duty_1=None, duty_3=None, out_max=None):
    duty_margin = 0.01
    excitation = excitation.lower()
    with path('magnet.data', f'{material}_{excitation}.h5') as h5file:
        data, metadata = h5_load(h5file)

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
def load_dataframe_datasheet_short(material, freq_min=None, freq_max=None, flux_min=None, flux_max=None,
                                   temp=None):
    temp_margin = 1.0
    with path('magnet.data', f'{material}_datasheet.h5') as h5file:
        data, metadata = h5_load(h5file)

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


def load_dataframe_datasheet_nearby(material, freq, flux, temp):
    temp_margin = 1.0
    with path('magnet.data', f'{material}_datasheet.h5') as h5file:
        data, metadata = h5_load(h5file)
        temp_max = temp + temp_margin
        temp_min = temp - temp_margin
        query = f'({temp_min} <= Temperature <= {temp_max})'
        data = data.query(query)

        query = f'(Frequency <= {freq})'
        data_aux = data.query(query)
        if data_aux.empty:
            freq_min = freq
        else:
            freq_min = data_aux['Frequency'].max()

        query = f'({freq} <=Frequency)'
        data_aux = data.query(query)
        if data_aux.empty:
            freq_max = freq
        else:
            freq_max = data_aux['Frequency'].min()

        query = f'(Flux_Density <= {flux})'
        data_aux = data.query(query)
        if data_aux.empty:
            flux_min = freq
        else:
            flux_min = data_aux['Flux_Density'].max()

        query = f'({flux} <=Flux_Density)'
        data_aux = data.query(query)
        if data_aux.empty:
            flux_max = flux
        else:
            flux_max = data_aux['Flux_Density'].min()

        query = f'({freq_min} <= Frequency <= {freq_max}) & ' \
                f'({flux_min} <= Flux_Density <= {flux_max})'

        data = data.query(query)
    return data

def load_dataframe_nearby(material, freq, flux):
    temp_margin = 1.0
    with path('magnet.data', f'{material}_interpolated.h5') as h5file:
        data, metadata = h5_load(h5file)

        query = f'(Frequency <= {freq})'
        data_aux = data.query(query)
        if data_aux.empty:
            freq_min = freq
        else:
            freq_min = data_aux['Frequency'].max()

        query = f'({freq} <=Frequency)'
        data_aux = data.query(query)
        if data_aux.empty:
            freq_max = freq
        else:
            freq_max = data_aux['Frequency'].min()

        query = f'(Flux_Density <= {flux})'
        data_aux = data.query(query)
        if data_aux.empty:
            flux_min = freq
        else:
            flux_min = data_aux['Flux_Density'].max()

        query = f'({flux} <=Flux_Density)'
        data_aux = data.query(query)
        if data_aux.empty:
            flux_max = flux
        else:
            flux_max = data_aux['Flux_Density'].min()

        query = f'({freq_min} <= Frequency <= {freq_max}) & ' \
                f'({flux_min} <= Flux_Density <= {flux_max})'

        data = data.query(query)
    return data


def load_metadata(material, excitation):
    excitation = excitation.lower()
    with path('magnet.data', f'{material}_{excitation}.h5') as h5file:
        data, metadata = h5_load(h5file)
    return metadata
