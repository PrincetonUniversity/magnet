import os.path
from PIL import Image
import streamlit as st

from magnet import __version__
from ui_db import ui_core_loss_db
from ui_predict import ui_core_loss_predict
from ui_raw import ui_download_raw_data
from ui_faq import ui_faq
from ui_intro import ui_intro
from magnet.simplecs.simfunctions import SimulationPLECS

STREAMLIT_ROOT = os.path.dirname(__file__)


def ui_multiple_materials(fn, n=1, *args, **kwargs):
    """
    Display multiple instances of input UI widgets, one for each 'material'
      denoted by 'A', 'B', ...
    :param fn: Function or callable that renders UI elements for Streamlit
      This function should take the material identifier ('A', 'B', ..) as the
      first input.
    :param n: Number of times to call `fn`
    :return: None
    """
    for i in range(int(n)):
        fn(chr(ord('A') + i), *args, **kwargs)


def contributor(name, email):
    st.sidebar.markdown(f'<h5>{name} ({email})</h5>', unsafe_allow_html=True)

 
if __name__ == '__main__':

    st.set_page_config(page_title='MagNet', layout='wide')

    st.sidebar.header('Welcome to Princeton MagNet')
    function_select = st.sidebar.radio(
        'Select a MagNet Function:',
        ('Introduction to MagNet', 'Core Loss Database', 'Core Loss Analysis', 'Core Loss Simulation', 'Download Waveform Data', 'Frequently Asked Questions')
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.title('Princeton-Dartmouth-Plexim MagNet Project')
        st.subheader('Data Driven Methods for Magnetic Core Loss Modeling')
        st.subheader('GitHub: https://github.com/PrincetonUniversity/Magnet')
    with col2:
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'magnetlogo.jpg')), width=300)

    st.markdown('---')

    if 'n_material' not in st.session_state:
        st.session_state.n_material = 1

    if function_select in ['Core Loss Database', 'Core Loss Analysis', 'Download Waveform Data']:
        clicked = st.sidebar.button("Add another case")
        if clicked:
            st.session_state.n_material += 1

    if function_select == 'Introduction to MagNet':
        ui_multiple_materials(ui_intro)
        st.session_state.n_material = 1  # Resets the number of plots

    if function_select == 'Core Loss Database':
        ui_multiple_materials(ui_core_loss_db, st.session_state.n_material)

    if function_select == 'Core Loss Analysis':
        ui_multiple_materials(ui_core_loss_predict, st.session_state.n_material)
        
    if function_select == 'Core Loss Simulation':
        ui_multiple_materials(SimulationPLECS)
            
    if function_select == 'Download Waveform Data':
        ui_multiple_materials(ui_download_raw_data, st.session_state.n_material, streamlit_root=STREAMLIT_ROOT)
        
    if function_select == 'Frequently Asked Questions':
        ui_multiple_materials(ui_faq)
        st.session_state.n_material = 1  # Resets the number of plots

    st.header('MagNet Research Team')
    st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'magnetteam.jpg')), width=1000)
    st.header('MagNet Sponsors')
    st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'sponsor.jpg')), width=1000)

    st.markdown('---')
    st.markdown(f"<h6>MAGNet v{__version__}</h6>", unsafe_allow_html=True)

    st.sidebar.header('Thanks for using MagNet!')
    contributor('Haoran Li', 'haoranli@princeton.edu')
    contributor('Diego Serrano', 'ds9056@princeton.edu')
    contributor('Evan Dogariu', 'edogariu@princeton.edu')
    contributor('Arielle Rivera', 'aerivera@princeton.edu')
    contributor('Yuxin Chen', 'yuxinc@wharton.upenn.edu')
    contributor('Thomas Guillod', 'Thomas.Paul.Henri.Guillod@dartmouth.edu')
    contributor('Vineet Bansal', 'vineetb@princeton.edu')
    contributor('Niraj Jha', 'jha@princeton.edu')
    contributor('Min Luo', 'luo@plexim.com')
    contributor('Charles R. Sullivan', 'charles.r.sullivan@dartmouth.edu')
    contributor('Minjie Chen', 'minjie@princeton.edu')
