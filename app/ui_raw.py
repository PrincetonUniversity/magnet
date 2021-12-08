import os.path
import streamlit as st
from magnet import config
from magnet.constants import material_names
from magnet.io import load_metadata

def header(material, excitation):
    s = f'Download Raw Data - {material} - {excitation} '
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

    data_file = os.path.join(streamlit_root, config.data.raw_data_file.format(material=material, excitation=read_excitation))
    with open(data_file, 'rb') as file:
        st.download_button(f'Download Data file', file, os.path.basename(data_file), key=m)

    metadata = load_metadata(material, read_excitation)
    st.write(metadata['info_date'])
    st.write(metadata['info_excitation'])
    st.write(metadata['info_core'])
    st.write(metadata['info_setup'])
    st.write(metadata['info_scope'])
    st.write(metadata['info_volt_meas'])
    st.write(metadata['info_curr_meas'])

    st.sidebar.markdown("""---""")
    st.markdown("""---""")
