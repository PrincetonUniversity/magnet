import os.path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from magnet.constants import material_names, materials, materials_extra, material_manufacturers, \
    material_applications, material_core_tested
from magnet.plots import waveform_visualization, core_loss_multiple, waveform_visualization_2axes, \
    cycle_points_sinusoidal, cycle_points_trapezoidal, scatter_plot
from magnet.io import load_dataframe
from magnet import config as c
import numpy as np
import csv
from magnet.core import BH_Transformer

STREAMLIT_ROOT = os.path.dirname(__file__)


@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

def ui_intro(m):
    
    st.title('MagNet AI')
    st.markdown("""---""")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Input Information')
        material = st.selectbox(
            f'Material:',
            material_names,
            key=f'material {m}',
            help='select from a list of available materials')
        temp = st.number_input(
            "Temperature [C]",
            min_value=25.0,
            max_value=90.0,
            value=25.0,
            step=1.0,
            format='%f',
            key=f'temp {m}',
            help='device surface temperature')
        freq = st.number_input(
            "Frequency [kHz]",
            min_value=10.0,
            max_value=1000.0,
            value=100.0,
            step=1.0,
            format='%f',
            key=f'freq {m}',
            help='fundamental frequency of the excitation') * 1e3
        bias = st.number_input(
            "Hdc Bias [A/m]",
            min_value=0.0,
            max_value=500.0,
            value=0.0,
            step=1.0,
            format='%f',
            key=f'bias {m}',
            help='determined by the bias dc current')
        mueff = materials_extra[material][0]
        st.write(f'Initial Relative Permeability (mu) set to {mueff} to determine the center of the B-H loop')

    with col2:
        st.subheader('Bac Input (Unit: mT)')  # Create an example Bac input file
        bdata = 100 * np.sin(np.linspace(-np.pi, np.pi, 128))
        output = {'B [mT]': bdata}
        csv = convert_df(pd.DataFrame(output))
        st.download_button(
            "Download an Example 128-Step Bac Input CSV File",
            data=csv,
            file_name='B-Input.csv',
            mime='text/csv', 
            )
    
        inputB = st.file_uploader(
            "CSV File for Bac in Single Cycle; Default: 100 mT Sinusoidal  with 128-Steps",
            type='csv',
            key=f'bfile {m}',
            help=None
                )

        if inputB is None:  # default input for display
            bdata = 100 * np.sin(np.linspace(-np.pi, np.pi, 128))
            hdata = BH_Transformer(material, freq, temp, bias, bdata)
            output = {'B [mT]': bdata, 'H [A/m]': hdata}
            loss = np.mean(np.multiply(bdata, hdata))
            csv = convert_df(pd.DataFrame(output))

        if inputB is not None:  # user input
            df = pd.read_csv(inputB)
            st.write(df)
            hdata = BH_Transformer(material, freq, temp, bias, bdata)
            output = {'B [mT]': bdata, 'H [A/m]': hdata}
            loss = np.mean(np.multiply(bdata, hdata))
            csv = convert_df(pd.DataFrame(output))
    st.markdown("""---""")
    st.header('MagNet AI Predicted Results')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('B-H Waveform')
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=np.linspace(1, 128, num=128),
                y=bdata+bias/mueff * np.ones(128),
                line=dict(color='mediumslateblue', width=4),
                name="B [mT]"),
            secondary_y=False,
            )
        fig.add_trace(
            go.Scatter(
                x=np.linspace(1, 128, num=128),
                y=bias/mueff * np.ones(128), 
                line=dict(color='brown', dash='longdash', width=4),
                name="Bdc [mT]"),
            secondary_y=False,
            )                
        fig.add_trace(
            go.Scatter(
                x=np.linspace(1, 128, num=128),
                y=hdata+bias * np.ones(128), 
                line=dict(color='firebrick', width=4),
                name="H [A/m]"),
            secondary_y=True,
            )
        fig.add_trace(
            go.Scatter(
                x=np.linspace(1, 128, num=128),
                y=bias * np.ones(128), 
                line=dict(color='black', dash='longdash', width=4),
                name="Hdc [A/m]"),
            secondary_y=True,
            )
        fig.update_xaxes(title_text="Fraction of a Cycle [%]")
        fig.update_yaxes(title_text="B - Flux Density [mT]", secondary_y=False)
        fig.update_yaxes(title_text="H - Field Strength [A/m]", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader('B-H Loop')
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=bdata + bias/mueff * np.ones(128),
                y=hdata + bias * np.ones(128),
                line=dict(color='mediumslateblue', width=4),
                name="B-H Loop"),
            secondary_y=False,
            )
        fig.add_trace(
            go.Scatter(
                x=bdata,
                y=bdata / mueff,
                line=dict(color='firebrick', dash='longdash', width=4),
                name="B = mu * H"),
            secondary_y=True,
            )

        fig.update_xaxes(title_text="B - Flux Density [mT]")
        fig.update_yaxes(title_text="H - Field Strength [A/m]")
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader(f'Volumetric Loss: {np.round(loss,2)} kW/m^3')

    st.download_button(
        "Download the B-H Loop as a CSV File",
        data=csv,
        file_name='BH-Loop.csv',
        mime='text/csv', 
        )

    st.markdown("""---""")

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
    st.markdown("""---""")
