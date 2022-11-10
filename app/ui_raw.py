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
            key=f'data {m}')

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
                    This .zip file contains a .txt file with information regarding the setup and core tested and three .csv files.
    
                    The .csv files contain the current and voltage waveforms saved from the oscilloscope. 
                    Each row is a different measurement (i.e. a data point with a different frequency, flux density, etc.) 
                    and each column is a sample. There are 10.000 samples per waveform. 
    
                    The sampling time, provided as a separate .csv file, is the same for each waveform (8 ns).
                    """)
                with col2:
                    st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'download_raw.png')), width=500)

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
                    This .zip file contains a .txt file contains information regarding the setup, core under test, and post-processing information.
    
                    Two .cvs file include the B and H waveform, where each row is a data point, and each column is a sample. 
                    1024 samples are saved per data point. Additionally, a .csv file contains the information of the sampling time, 
                    where each row corresponds to the data-point of the B and H files. 
                    In this case, the sampling time depends on the frequency of the waveform, as a single cycle is provided.
                        """)
                with col2:
                    st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'download_single.png')), width=500)
        st.success('Done!')
        st.markdown("""---""")
            
    st.header('Terms of Access:')
    st.subheader("""
    By downloading the MagNet data you by default agree:
    """)
    st.caption('RESEARCHER_FULLNAME (the "Researcher") has requested permission to use the MagNet database (the "Database") at Princeton University. In exchange for such permission, Researcher hereby agrees to the following terms and conditions: Researcher shall use the Database only for non-commercial research and educational purposes. Princeton University make no representations or warranties regarding the Database, including but not limited to warranties of non-infringement or fitness for a particular purpose. Researcher accepts full responsibility for his or her use of the Database and shall defend and indemnify the MagNet team, Princeton University, including their employees, Trustees, officers and agents, against any and all claims arising from Researchers use of the Database, including but not limited to Researchers use of any copies of copyrighted datasheet that he or she may create from the Database. Researcher may provide research associates and colleagues with access to the Database provided that they first agree to be bound by these terms and conditions. Princeton University reserve the right to terminate Researchers access to the Database at any time. If Researcher is employed by a for-profit, commercial entity, Researchers employer shall also be bound by these terms and conditions, and Researcher hereby represents that he or she is fully authorized to enter into this agreement on behalf of such employer. The law of the State of New Jersey shall apply to all disputes under this agreement.')
    st.markdown("""---""")
