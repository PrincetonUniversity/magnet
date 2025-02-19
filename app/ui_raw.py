import os.path
from PIL import Image
import streamlit as st
from magnet import config as c
from magnet.constants import material_list, material_manufacturers
from magnet.io import load_metadata

STREAMLIT_ROOT = os.path.dirname(__file__)


def ui_download_data(m, streamlit_root):

    st.title('MagNet Download for Open-Source Sharing')

    col1, col2, col3 = st.columns(3)
    with col1:
        material = st.selectbox(
            f'Material:',
            material_list + list("-"),
            index=len(material_list),
            key=f'material {m}')

    # User must select data first, to avoid loading both large files
    with col2:
        selection = st.selectbox(
            f'Type of data:',
            ('Voltage and Current Data', 'Single Cycle B and H'),
            key=f'data {m}',
            index=1)
        
    st.warning("For MagNet Challenge Participants, please be sure to download the \"Single Cycle B and H\" files.")

    if material != '-':

        with st.expander(f'{material_manufacturers[material]} - {material}: measurement details'):
            metadata = load_metadata(material)
            st.write('Core information:')
            st.write(metadata['info_core'])
            st.write('Setup information:')
            st.write(metadata['info_setup'])
            st.write('Data-processing information:')
            st.write(metadata['info_processing'])
            

        st.write('Units: V [V], I [A], B [T], H [A/m], Ts [s], Temp [C]')

        with st.spinner('MagNet AI is Collecting the Data, Please Wait...'):
            if selection == 'Voltage and Current Data':
                st.subheader('Original measured voltage and current data')
                col1, col2 = st.columns([2, 1])
                with col1:
                    if os.path.isfile(os.path.join(
                            streamlit_root, c.streamlit.data_file.format(material=material, excitation='measurements'))):
                        data_file_raw = os.path.join(
                            streamlit_root, c.streamlit.data_file.format(material=material, excitation='measurements'))
                        with open(data_file_raw, 'rb') as file:
                            st.download_button(f'Download zip file',
                                               file,
                                               os.path.basename(data_file_raw),
                                               key=[m, 'Raw'])
                    else:
                        st.subheader('Download data missing, please contact us')
                    st.write("""
                    This .zip file contains a .txt file with information regarding the setup and core tested and three .csv files. The .csv files contain the current and voltage waveforms saved from the oscilloscope. Each row is a different measurement time sequence for B or H waveform (i.e. an operation point with a different frequency, flux density, etc.) and each column is a time step sample (i.e., 10000 steps). There are 10,000 step samples per waveform. The sampling time, provided as a separate .csv file, is the same for each time sequence waveform (8 ns). A .csv file contains the information of the temperature where each time sequence is measured.
                    """)
                with col2:
                    st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'download_raw.jpg')), width=500)

            if selection == 'Single Cycle B and H':
                st.subheader('Single switching cycle post-processed B-H data')
                col1, col2 = st.columns([2, 1])
                with col1:
                    if os.path.isfile(os.path.join(
                            streamlit_root, c.streamlit.data_file.format(material=material, excitation='cycle'))):
                        data_file_cycle = os.path.join(
                            streamlit_root, c.streamlit.data_file.format(material=material, excitation='cycle'))
                        with open(data_file_cycle, 'rb') as file:
                            st.download_button(f'Download zip file',
                                               file,
                                               os.path.basename(data_file_cycle),
                                               key=[m, 'Cycle'])
                    else:
                        st.subheader('Download data missing, please contact us')
                    st.write("""
                    This .zip file contains a .txt file contains information regarding the setup, core under test, and post-processing information. Two .cvs file include the B and H waveform, where each row is a 1024 step time sequence, and each column is a sample point. 1024 samples are saved per time sequence. Additionally, a .csv file contains the information of the sampling time, where each row corresponds to the data-point of the B and H files. In this case, the sampling time depends on the frequency of the waveform, as a single cycle is provided. A .csv file contains the information of the temperature where each time sequence is measured.
                        """)
                with col2:
                    st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'download_single.jpg')), width=500)
        st.success('Done!')
        st.markdown("""---""")
            
    st.header('Terms of Access:')
    st.subheader("""
    By downloading the MagNet data you by default agree:
    """)
    st.caption("""
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """)
    st.markdown("""---""")
