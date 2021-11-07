import re
import numpy as np
import streamlit as st

from magnet import config
from magnet.constants import material_names, excitations

def header(material, excitation):
    s = f'Download Raw Data - {material} Material {excitation} '
    return st.header(s)

def ui_download_raw_data(m):
    st.sidebar.header(f'Information for Material {m}')
    material = st.sidebar.selectbox(f'Material {m}:', material_names)
    excitation = st.sidebar.selectbox(f'Excitation {m}:', excitations + ("Arbitrary-Periodic", "Non-Periodic"))
    
    header(material, excitation)
    file=""
    st.download_button(f'Download Material {m}',file,material+"-"+excitation+".csv","text/csv",key=m)
    st.write("CSV Format: ")

    st.sidebar.markdown("""---""")
    st.markdown("""---""")