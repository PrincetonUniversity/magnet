import os.path
import streamlit as st
from magnet import config
from magnet.constants import material_names
from magnet.io import load_metadata


def header(material, excitation):
    s = f'Download Data - {material} - {excitation} '
    return st.header(s)


def ui_download_raw_data(m, streamlit_root):
    st.sidebar.header(f'Information for Material {m}')
    material = st.sidebar.selectbox(f'Material {m}:', material_names)
    excitation = st.sidebar.selectbox(f'Excitation {m}:', ("Sinusoidal", "Triangular-Trapezoidal"))
    # Changed as we don't have "Arbitrary-Periodic" or "Non-Periodic" yet
    # It does not make sense to have "Datasheet", also, Triangular and Trapezoidal are saved into the same zip file

    if excitation == "Triangular-Trapezoidal":
        read_excitation = 'Trapezoidal'
    if excitation == "Sinusoidal":
        read_excitation = 'Sinusoidal'

    header(material, excitation)

    metadata = load_metadata(material, read_excitation)
    with st.expander("Measurement details"):
        st.write(metadata['info_date'])
        st.write(metadata['info_excitation'])
        st.write(metadata['info_core'])
        st.write(metadata['info_setup'])
        st.write(metadata['info_scope'])
        st.write(metadata['info_volt_meas'])
        st.write(metadata['info_curr_meas'])

    st.write('Raw data - 20us sample')
    data_file_raw = os.path.join(streamlit_root,
                            config.streamlit.raw_data_file.format(material=material, excitation=read_excitation))
    with open(data_file_raw, 'rb') as file:
        st.download_button(f'Download zip file', file, os.path.basename(data_file_raw), key=[m, 'Raw'])

    st.write('Post-processed data - a single cycle')

    data_file_cycle = os.path.join(streamlit_root,
                             config.streamlit.single_cycle_file.format(material=material, excitation=read_excitation))
    with open(data_file_cycle, 'rb') as file:
        st.download_button(f'Download zip file', file, os.path.basename(data_file_cycle), key=[m, 'Cycle'])

    st.sidebar.markdown("""---""")
    st.markdown("""---""")
