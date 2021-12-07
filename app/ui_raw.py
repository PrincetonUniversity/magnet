import os.path
import streamlit as st

from magnet import config
from magnet.constants import material_names, input_dir


def header(material, excitation):
    s = f'Download Raw Data - {material} Material {excitation} '
    return st.header(s)


def ui_download_raw_data(m, streamlit_root):
    st.sidebar.header(f'Information for Material {m}')
    material = st.sidebar.selectbox(f'Material {m}:', material_names)
    excitation = st.sidebar.selectbox(f'Excitation {m}:', ("Sinusoidal", "Triangular-Trapezoidal"))
    # Changed as we don't have "Arbitrary-Periodic" or "Non-Periodic" yet
    # It does not make sense to have "Datasheet", also, Triangular and Trapezoidal are saved into the same zip file
    header(material, excitation)

    data_file = os.path.join(streamlit_root, config.data.raw_data_file.format(material=material, excitation=excitation))
    with open(data_file, 'rb') as file:
        st.download_button(f'Download Data file', file, os.path.basename(data_file), key=m)

    file_txt = input_dir + material + "_" + excitation + "_Test_Info.txt"
    txt_info_file = open(file_txt)
    for line in txt_info_file:
        st.write(line)

    st.sidebar.markdown("""---""")
    st.markdown("""---""")
