import os.path
import streamlit as st

from magnet import config as c
from magnet.constants import material_names, material_manufacturers, excitations_raw
from magnet.io import load_metadata


def ui_download_raw_data(m, streamlit_root):
    st.sidebar.header(f'Information: Case {m}')
    excitation = st.sidebar.selectbox(
        f'Excitation:',
        excitations_raw,
        key=f'excitation {m}')
    material = st.sidebar.selectbox(
        f'Material:',
        material_names,
        key=f'material {m}')
    if excitation == "Triangular-Trapezoidal":
        read_excitation = 'Trapezoidal'
    if excitation == "Sinusoidal":
        read_excitation = 'Sinusoidal'

    st.title(f'Download Data: Case {m}')
    st.subheader(f'{material_manufacturers[material]} - {material}, '
                 f'{excitation} excitation')

    metadata = load_metadata(material, read_excitation)
    with st.expander("Measurement details"):
        st.write(metadata['info_date'])
        st.write(metadata['info_excitation'])
        st.write(metadata['info_core'])
        st.write(metadata['info_setup'])
        st.write(metadata['info_scope'])
        st.write(metadata['info_volt_meas'])
        st.write(metadata['info_curr_meas'])

    st.subheader('Raw data - 20 us voltage and current')
    if os.path.isfile(os.path.join(
        streamlit_root,
        c.streamlit.raw_data_file.format(material=material, excitation=read_excitation))):

        data_file_raw = os.path.join(
            streamlit_root,
            c.streamlit.raw_data_file.format(material=material, excitation=read_excitation))
        with open(data_file_raw, 'rb') as file:
            st.download_button(f'Download zip file', file, os.path.basename(data_file_raw), key=[m, 'Raw'])
    else:
        st.write('Download data missing, please contact us')

    st.subheader('Post-processed data - B-H in a single cycle')
    if os.path.isfile(os.path.join(
        streamlit_root,
        c.streamlit.single_cycle_file.format(material=material, excitation=read_excitation))):
        data_file_cycle = os.path.join(
            streamlit_root,
            c.streamlit.single_cycle_file.format(material=material, excitation=read_excitation))
        with open(data_file_cycle, 'rb') as file:
            st.download_button(f'Download zip file', file, os.path.basename(data_file_cycle), key=[m, 'Cycle'])
    else:
        st.write('Download data missing, please contact us')
    st.sidebar.markdown("""---""")
    st.markdown("""---""")
