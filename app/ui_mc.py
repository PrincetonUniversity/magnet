import os.path
from PIL import Image
import streamlit as st

STREAMLIT_ROOT = os.path.dirname(__file__)


def ui_mc(m):

    st.title('2023 IEEE PELS MagNet Challenge')
    st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'mclogo.jpg')), width=300)
    
    st.header('MagNet Competition')
    st.write("""
        The MagNet Challenge evaluates algorithms for modelling the material characteristics of power magnetics. One high level motivation is to allow researchers to compare progress in power magnetic models across a wide operation range for many different materials -- taking advantages of the expensive magnetic data collection effort. Another motivation is to advance the research on data driven and computer aided design methods in power electronics.
    """)
    st.header('MagNet Workshop')
    st.write("""
    Every year of the challenge there is a corresponding workshop at one of the premier power electronics conferences. The purpose of the workshop is to present the methods and results of the challenge. Challenge participants with the most successful and innovative entries are invited to present. Please visit the corresponding challenge page for workshop schedule and information.
    """)
    st.header('MagNet Database')
    st.write("""
    You can find extensive data on power magnetic materials from this website under the "MagNet Database" and "MagNet Download" page. These data can be used for developing analytical model or training numerical models for modeling power magnetic material characteristics.
    
    Contact us if you have high quality data to share with the community.
    """)
    st.header('MagNet Evaluation Server')
    st.write("""
    The MagNet evaluation server can be used to evaluate magnetic modeling results on the test set of the past competition.
    """)

    st.markdown("""---""")
