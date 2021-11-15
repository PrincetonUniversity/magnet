import os.path
from PIL import Image
import streamlit as st

from magnet import __version__
from ui_db import ui_core_loss_db
from ui_predict import ui_core_loss_predict
from ui_raw import ui_download_raw_data

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
    st.sidebar.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'magnetlogo.jpg')), width=300)

    st.sidebar.header('Welcome to Princeton MagNet')
    function_select = st.sidebar.radio(
        'Select MagNet Function:',
        ('Core Loss Database', 'Core Loss Analysis', 'Download Raw Data')
    )

    st.title('Princeton-Dartmouth-Plexim MagNet Project')
    st.subheader('Data Driven Methods for Magnetic Core Loss Modeling')
    st.subheader('GitHub: https://github.com/PrincetonUniversity/Magnet')
    st.markdown('---')

    if 'n_material' not in st.session_state:
        st.session_state.n_material = 1

    clicked = st.sidebar.button("Add another material")
    if clicked:
        st.session_state.n_material += 1

    if function_select == 'Core Loss Database':
        ui_multiple_materials(ui_core_loss_db, st.session_state.n_material)

    if function_select == 'Core Loss Analysis':
        ui_multiple_materials(ui_core_loss_predict, st.session_state.n_material)
            
    if function_select == 'Download Raw Data':
        ui_multiple_materials(ui_download_raw_data, st.session_state.n_material)

    st.title('MagNet Research Team')
    st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'magnetteam.jpg')), width=1000)
    st.title('MagNet Sponsors')
    st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'sponsor.jpg')), width=1000)

    st.markdown('---')
    st.markdown(f"<h6>MAGNet v{__version__}</h6>", unsafe_allow_html=True)

    st.sidebar.title('Thanks for using MagNet!')
    contributor('Haoran Li', 'haoranli@princeton.edu')
    contributor('Diego Serrano Lopez', 'ds9056@princeton.edu')
    contributor('Evan Dogariu', 'edogariu@princeton.edu')
    contributor('Thomas Guillod', 'Thomas.Paul.Henri.Guillod@dartmouth.edu')
    contributor('Vineet Bansal', 'vineetb@princeton.edu')
    contributor('Niraj Jha', 'jha@princeton.edu')
    contributor('Min Luo', 'luo@plexim.com')
    contributor('Charles R. Sullivan', 'charles.r.sullivan@dartmouth.edu')
    contributor('Minjie Chen', 'minjie@princeton.edu')
