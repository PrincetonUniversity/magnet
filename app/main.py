import os.path
import streamlit as st
from ui_db import ui_core_loss_db
from ui_predict import ui_core_loss_predict
from ui_raw import ui_download_raw_data

STREAMLIT_ROOT = os.path.dirname(__file__)

if __name__ == '__main__':

    st.set_page_config(page_title='MagNet', layout='wide')
    st.sidebar.image(os.path.join(STREAMLIT_ROOT, 'img', 'magnetlogo.jpg'), width=300)

    st.sidebar.header('Welcome to Princeton MagNet')
    function_select = st.sidebar.radio(
        'Select MagNet Function:',
        ('Core Loss Database', 'Core Loss Analysis','Download Raw Data')
    )

    
    st.title('Princeton-Dartmouth-Plexim MagNet Project')
    st.subheader('Data Driven Methods for Magnetic Core Loss Modeling')
    st.subheader('GitHub: https://github.com/PrincetonUniversity/Magnet')
    st.markdown('''---''')

    if function_select == 'Core Loss Database':
        itemnum = st.sidebar.selectbox("Number of Materials for Analysis:",('1','2','3'))
        if itemnum == '1':
            ui_core_loss_db('A')
        if itemnum == '2':
            ui_core_loss_db('A')
            ui_core_loss_db('B')
        if itemnum == '3':
            ui_core_loss_db('A')
            ui_core_loss_db('B')
            ui_core_loss_db('C')

    if function_select == 'Core Loss Analysis':
        itemnum = st.sidebar.selectbox("Number of Materials for Analysis:",('1','2','3'))
        if itemnum == '1':
            ui_core_loss_predict('A')
        if itemnum == '2':
            ui_core_loss_predict('A')
            ui_core_loss_predict('B')
        if itemnum == '3':
            ui_core_loss_predict('A')
            ui_core_loss_predict('B')
            ui_core_loss_predict('C')
            
    if function_select == 'Download Raw Data':
        itemnum = st.sidebar.selectbox("Number of Materials for Analysis:",('1','2','3'))
        if itemnum == '1':
            ui_download_raw_data('A')
        if itemnum == '2':
            ui_download_raw_data('A')
            ui_download_raw_data('B')
        if itemnum == '3':
            ui_download_raw_data('A')
            ui_download_raw_data('B')
            ui_download_raw_data('C')       
            

    st.title('MagNet Research Team')
    st.image(os.path.join(STREAMLIT_ROOT, 'img', 'magnetteam.jpg'), width=1000)
    st.title('MagNet Sponsors')
    st.image(os.path.join(STREAMLIT_ROOT, 'img', 'sponsor.jpg'), width=1000)
    st.sidebar.title('Thanks for using MagNet!')
    st.sidebar.subheader('Haoran Li (haoranli@princeton.edu)')
    st.sidebar.subheader('Diego Serrano Lopez (ds9056@princeton.edu)')
    st.sidebar.subheader('Evan Dogariu (edogariu@princeton.edu)')
    st.sidebar.subheader('Thomas Guillod (Thomas.Paul.Henri.Guillod@dartmouth.edu)')
    st.sidebar.subheader('Vineet Bansal (vineetb@princeton.edu)')
    st.sidebar.subheader('Niraj Jha (jha@princeton.edu)')
    st.sidebar.subheader('Min Luo (luo@plexim.com )')
    st.sidebar.subheader('Charles R. Sullivan (charles.r.sullivan@dartmouth.edu)')
    st.sidebar.subheader('Minjie Chen (minjie@princeton.edu)')
