import os.path
import streamlit as st
from magnet import __version__
from ui_db import ui_core_loss_db
from ui_predict import ui_core_loss_predict

STREAMLIT_ROOT = os.path.dirname(__file__)

if __name__ == '__main__':

    st.set_page_config(page_title='MagNet', layout='wide')
    st.image(os.path.join(STREAMLIT_ROOT, 'img', 'pulogo.jpg'), width=600)
    st.sidebar.image(os.path.join(STREAMLIT_ROOT, 'img', 'magnetlogo.jpg'), width=300)

    st.sidebar.header('Princeton MagNet Project')
    function_select = st.sidebar.radio(
        'Select one of the two functions',
        ('Core Loss Database', 'Core Loss Prediction')
    )

    st.title('Princeton-Dartmouth-Plexim MagNet Project')
    st.header('Data Driven Methods for Magnetic Core Loss Modeling')
    st.header('GitHub: https://github.com/PrincetonUniversity/Magnet')
    st.markdown('''---''')

    if function_select == 'Core Loss Database':
        ui_core_loss_db('A')
        ui_core_loss_db('B')

    if function_select == 'Core Loss Prediction':
        ui_core_loss_predict()

    st.markdown('''---''')
    st.title('Mag-Net Research Team')
    st.image(os.path.join(STREAMLIT_ROOT, 'img', 'magnetteam.jpg'), width=1000)
    st.title('Mag-Net Sponsors')
    st.image(os.path.join(STREAMLIT_ROOT, 'img', 'sponsor.jpg'), width=1000)
    st.title('Mag-Net Website Developers')
    st.subheader('Haoran Li (haoranli@princeton.edu)')
    st.subheader('Diego Serrano Lopez (ds9056@princeton.edu)')
    st.subheader('Evan Dogariu (edogariu@princeton.edu)')
    st.subheader('Min Luo (luo@plexim.com )')
    st.subheader('Thomas Guillod (Thomas.Paul.Henri.Guillod@dartmouth.edu)')
    st.subheader('Vineet Bansal (vineetb@princeton.edu)')
    st.subheader('Minjie Chen (minjie@princeton.edu)')