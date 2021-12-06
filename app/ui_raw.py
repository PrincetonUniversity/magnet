import re
import numpy as np
import streamlit as st

from magnet import config
from magnet.constants import material_names  # , excitations # We don't need "Datasheet"

def header(material, excitation):
    s = f'Download Raw Data - {material} Material {excitation} '
    return st.header(s)

def ui_download_raw_data(m):
    st.sidebar.header(f'Information for Material {m}')
    material = st.sidebar.selectbox(f'Material {m}:', material_names)
    excitation = st.sidebar.selectbox(f'Excitation {m}:', ("Sinusoidal", "Trapezoidal"))
    # Changed as we don't have "Arbitrary-Periodic" or "Non-Periodic" yet
    # It does not make sense to have "Datasheet", also, Triangular and Trapezoidal are saved into the same zip file
    header(material, excitation)

    input_dir = "src/magnet/data/"

    file_zip = input_dir + material + "_" + excitation + "_Raw_Data_Short.zip"
    st.download_button(f'Download ZIP file', open(file_zip, "rb"), material + "_" + excitation + ".zip", "zip", key=m)

    file_txt = input_dir + material + "_" + excitation + "_Test_Info.txt"
    txt_info_file = open(file_txt)
    for line in txt_info_file:
        st.write(line)

    st.sidebar.markdown("""---""")
    st.markdown("""---""")
