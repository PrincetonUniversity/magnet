import os.path
from PIL import Image
import streamlit as st
#import streamlit_analytics

from magnet import __version__
from ui_db import ui_core_loss_db
from ui_predict import ui_core_loss_predict
from ui_raw import ui_download_data
from ui_faq import ui_faq
from ui_intro import ui_intro
from ui_mc import ui_mc
from magnet.simplecs.simfunctions import SimulationPLECS
from magnet.constants import material_list
from magnet.io import load_dataframe

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
    

    st.set_page_config(page_title='MagNet', page_icon="⚡", layout='wide')
    st.sidebar.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'magnetlogo.jpg')), width=300)
    st.sidebar.markdown('[GitHub](https://github.com/PrincetonUniversity/Magnet) | [Doc](https://princetonuniversity.github.io/magnet/) | [Report an Issue](https://github.com/PrincetonUniversity/magnet/issues) ')
    st.sidebar.markdown('[Princeton Power Electronics Lab](https://www.princeton.edu/~minjie/)')
    st.sidebar.caption('by Princeton-Dartmouth-Plexim')
    st.sidebar.header('[Contact Us](https://forms.gle/6SHLF45V8vdkiPENA) | [Error History](https://docs.google.com/spreadsheets/d/1YPfv8w0kzO4UhrgpV7L3LixFJpvesBvC1CmkdJ9fGe0/edit?usp=sharing)')
    st.sidebar.header('MagNet Platform')
    
#    with streamlit_analytics.track():
#        st.sidebar.text_input("Name/Email Address?")
#        st.sidebar.text_input("Comments?")
#        st.sidebar.button("Submit")
    
    function_select = st.sidebar.radio(
        'Select One:',
        ('MagNet AI', 'MagNet Database', 'MagNet Smartsheet',
         'MagNet Simulation', 'MagNet Download', 'MagNet Challenge', 'MagNet Help')
    )
    
    if 'n_material' not in st.session_state:
        st.session_state.n_material = 1

    if function_select in ['MagNet Database', 'MagNet Smartsheet']:
        clicked = st.sidebar.button("Add Another Case")
        if clicked:
            st.session_state.n_material += 1

    if function_select == 'MagNet AI':
        ui_multiple_materials(ui_intro)
        st.session_state.n_material = 1  # Resets the number of plots

    if function_select == 'MagNet Database':
        ui_multiple_materials(ui_core_loss_db, st.session_state.n_material)

    if function_select == 'MagNet Smartsheet':
        ui_multiple_materials(ui_core_loss_predict, st.session_state.n_material)
        
    if function_select == 'MagNet Simulation':
        st.title('MagNet Simulation for Circuit Analysis')
        ui_multiple_materials(SimulationPLECS)
        st.session_state.n_material = 1  # Resets the number of plots
            
    if function_select == 'MagNet Download':
        ui_multiple_materials(ui_download_data, streamlit_root=STREAMLIT_ROOT)
        st.session_state.n_material = 1  # Resets the number of plots
        
    if function_select == 'MagNet Challenge':
        ui_multiple_materials(ui_mc)
        st.session_state.n_material = 1  # Resets the number of plots
            
    if function_select == 'MagNet Help':
        ui_multiple_materials(ui_faq)
        st.session_state.n_material = 1  # Resets the number of plots

    st.header('MagNet Research Team')
    st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'magnetteam.jpg')), width=1000)
    st.header('MagNet Sponsors')
    st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'sponsor.jpg')), width=1000)

    st.markdown('---')
    st.markdown(f"<h6>MAGNet v{__version__}</h6>", unsafe_allow_html=True)
        
    st.sidebar.header('MagNet Data Status')
    n_tot = 0
    for material in material_list:
        n_tot = n_tot + len(load_dataframe(material))
    st.sidebar.write(f'- Number of data points: {n_tot}')
    st.sidebar.write(f'- Number of materials: {len(material_list)}')
        
    st.sidebar.header('MagNet Team')
    contributor('Haoran Li', 'haoranli@princeton.edu')
    contributor('Diego Serrano', 'ds9056@princeton.edu')
    contributor('Shukai Wang', 'sw0123@princeton.edu')
    contributor('Thomas Guillod', 'thomas.paul.henri.guillod@dartmouth.edu')
    contributor('Min Luo', 'luo@plexim.com')
    contributor('Vineet Bansal', 'vineetb@princeton.edu')
    contributor('Yuxin Chen', 'yuxinc@wharton.upenn.edu')
    contributor('Niraj Jha', 'jha@princeton.edu')
    contributor('Charles R. Sullivan', 'charles.r.sullivan@dartmouth.edu')
    contributor('Minjie Chen', 'minjie@princeton.edu')
        
        
