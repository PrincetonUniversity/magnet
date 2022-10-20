import os.path
import pandas as pd
import streamlit as st
from magnet.constants import material_names, materials, materials_extra, material_manufacturers, \
    material_applications, material_core_tested
from magnet.plots import waveform_visualization, core_loss_multiple, waveform_visualization_2axes, \
    cycle_points_sinusoidal, cycle_points_trapezoidal
from magnet.io import load_dataframe
from magnet import config as c
import numpy as np
import csv

STREAMLIT_ROOT = os.path.dirname(__file__)


@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')


def transformer(material, freq, temp, bias, bdata):
    hdata = np.add(bdata, 0.1)
    return hdata


def ui_intro(m):
    
    st.title('MagNet Go: AI for Power Magnetics')
    st.subheader('MagNet Input - Operating Condition')
    col1, col2 = st.columns(2)
    with col1:
        material = st.selectbox(
            f'Material:',
            material_names,
            key=f'material {m}')
        temp = st.number_input(
            "Temperature (C)",
            value=25.0,
            format='%f',
            key=f'temp {m}')
        
    with col2:
        freq = st.number_input(
            "Frequency (kHz)",
            format='%f',
            value=100.0,
            key=f'freq {m}') * 1e3
        bias = st.number_input(
            "DC Bias (mT)",
            format='%f',
            key=f'bias {m}') * 1e-3
    
    st.subheader('MagNet Input - Excitation Waveform')
    inputfile = st.file_uploader(
        "CSV File for B Excitation in Single Cycle [mT]; Default: 100 mT Sinusoidal",
        type='csv',
        key=f'upload {m}',
        help=None
    )
    if inputfile is None:
        bdata = 0.1 * np.linspace(-np.pi, np.pi, 101),
        hdata = transformer(material, freq, temp, bias, bdata),
        output = {'B': bdata, 'H': hdata},
        loss = np.mean(np.multiply(bdata, hdata)),
        csv = convert_df(pd.DataFrame(output))
    if inputfile is not None:
        bdata = inputfile.read(),
        hdata = transformer(material, freq, temp, bias, bdata),
        output = {'B [mT]': bdata, 'H [A/m]': hdata},
        loss = np.mean(bdata * hdata),
        csv = convert_df(pd.DataFrame(output))

    st.subheader(f'MagNet Predicted Volumetric Loss: {np.round(loss,2)} kW/m^3')
    st.download_button(
        "Download the B-H Loop as a CSV File",
        data=csv,
        file_name='BH-Loop.csv',
        mime='text/csv', 
        )
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('B-H Waveform')
        waveform_visualization(
            st,
            x=np.linspace(1, 100, num=100),
            y=bdata,
            x_title='Fraction of a cycle',
            y_title='B - Field Strength [A/m]',
            color='mediumslateblue', width=4)
        
    with col2:
        st.markdown('B-H Loop')
        waveform_visualization(
            st,
            x=np.linspace(1, 100, num=100),
            y=bdata,
            x_title='B - Flux Density [mT]',
            y_title='H - Field Strength [A/m]',
            color='mediumslateblue', width=4)

    st.markdown("""---""")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header('MagNet Status')
        st.write("")
        n_sine = 0
        n_tri = 0
        n_tot = 0

        for material in material_names:
            n_sine = n_sine + len(load_dataframe(
                material, freq_min=None, freq_max=None, flux_min=None, flux_max=None,
                bias=None, duty_p=-1, duty_n=-1, temp=None))
            for duty_aux in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                n_tri = n_tri + len(load_dataframe(
                    material, freq_min=None, freq_max=None, flux_min=None, flux_max=None, bias=None,
                    duty_p=duty_aux, duty_n=1-duty_aux, temp=None))
            n_tot = n_tot + len(load_dataframe(material))
        st.subheader(f'Number of materials added: {len(material_names)}')
        st.write("")
        st.subheader(f'Total number of data points: {n_tot}')
        st.write(f'{n_sine} Sinusoidal points, {n_tri} Triangular points and {n_tot-n_sine-n_tri} Trapezoidal points.')
        st.write(f'During the tests, the core temperature may increase 5 C ~ 10 C in worst case conditions.')

    with col2:
        st.header('How to Cite')
        st.write("""
            If you used MagNet, please cite us with the following:
            
            [1] D. Serrano et al., "Neural Network as Datasheet: Modeling B-H Loops of Power Magnetics with Sequence-to-Sequence LSTM Encoder-Decoder Architecture," IEEE 23rd Workshop on Control and Modeling for Power Electronics (COMPEL), 2022, pp. 1-8.
            
            [2] H. Li, D. Serrano, T. Guillod, E. Dogariu, A. Nadler, S. Wang, M. Luo, V. Bansal, Y. Chen, C. R. Sullivan, and M. Chen, 
            "MagNet: an Open-Source Database for Data-Driven Magnetic Core Loss Modeling," 
            IEEE Applied Power Electronics Conference (APEC), Houston, 2022.
            
            [3] E. Dogariu, H. Li, D. Serrano, S. Wang, M. Luo and M. Chen, 
            "Transfer Learning Methods for Magnetic Core Loss Modeling,” 
            IEEE Workshop on Control and Modeling of Power Electronics (COMPEL), Cartagena de Indias, Colombia, 2021.
            
            [4] H. Li, S. R. Lee, M. Luo, C. R. Sullivan, Y. Chen and M. Chen, 
            "MagNet: A Machine Learning Framework for Magnetic Core Loss Modeling,” 
            IEEE Workshop on Control and Modeling of Power Electronics (COMPEL), Aalborg, Denmark, 2020.
        """)

    df = pd.DataFrame({'Manufacturer': material_manufacturers})
    df['Material'] = materials.keys()
    df['Applications'] = pd.DataFrame({'Applications': material_applications})
    df_extra = pd.DataFrame(materials_extra)
    df['mu_i_r'] = df_extra.iloc[0]
    df['f_min [Hz]'] = df_extra.iloc[1]
    df['f_max [Hz]'] = df_extra.iloc[2]
    df_params = pd.DataFrame(materials)
    df['k_i*'] = df_params.iloc[0]
    df['alpha*'] = df_params.iloc[1]
    df['beta*'] = df_params.iloc[2]
    df['Tested Core'] = pd.DataFrame({'Tested Core': material_core_tested})
    # Hide the index column
    hide_table_row_index = """
                <style>
                tbody th {display:none}
                .blank {display:none}
                </style>
                """  # CSS to inject contained in a string
    st.markdown(hide_table_row_index, unsafe_allow_html=True)  # Inject CSS with Markdown
    st.table(df)

    st.write(f'*iGSE parameters obtained from the sinusoidal measurements at 25 C without bias and data '
             f'between 50 kHz and 500 kHz and 10 mT and 300 mT; '
             f'with Pv, f, and B in W/m^3, Hz and T respectively')

    st.markdown("""---""")
