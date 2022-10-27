import os.path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from magnet.constants import material_list, material_extra, material_core_params
from magnet.io import load_dataframe
import numpy as np
import csv
from magnet.core import BH_Transformer, loss_BH

STREAMLIT_ROOT = os.path.dirname(__file__)

mu0 = 4e-7*np.pi

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
            material_list,
            key=f'material {m}',
            help='select from a list of available materials')

        df = load_dataframe(material)  # To find the range of the variables

        temp = st.number_input(
            "Temperature [C]",
            min_value=-50.0,
            max_value=200.0,
            value=25.0,
            step=5.0,
            format='%f',
            key=f'temp {m}',
            help='device surface temperature')
        if temp < min(df['Temperature']):
            st.warning(f"The model has not been trained for temperature below {round(min(df['Temperature']))} C")
        if temp > max(df['Temperature']):
            st.warning(f"The model has not been trained for temperature above {round(max(df['Temperature']))} C")

        freq = st.number_input(
            "Frequency [kHz]",
            min_value=1.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            format='%f',
            key=f'freq {m}',
            help='fundamental frequency of the excitation') * 1e3
        if freq < min(df['Frequency']):
            st.warning(f"The model has not been trained for frequencies below {round(min(df['Frequency']) * 1e-3)} kHz")
        if freq > max(df['Frequency']):
            st.warning(f"The model has not been trained for frequencies above {round(max(df['Frequency']) * 1e-3)} kHz")

        bias = st.number_input(
            "Hdc Bias [A/m]",
            min_value=-1000.0,
            max_value=1000.0,
            value=0.0,
            step=5.0,
            format='%f',
            key=f'bias {m}',
            help='determined by the bias dc current')
        if bias < 0:
            st.warning(f"The model has not been trained for bias below 0 A/m")
        if bias > round(max(df['DC_Bias'])):
            st.warning(f"The model has not been trained for bias above {round(max(df['DC_Bias']))} A/m")

        mu_relative = material_extra[material][0]
        st.write(f'Initial Relative Permeability (mu) set to {mu_relative} to determine the center of the B-H loop')

    with col2:
        st.subheader('Bac Input (Unit: mT)')  # Create an example Bac input file
        bdata0 = 100 * np.sin(np.linspace(0, 2*np.pi, 128))
        output = {'B [mT]': bdata0}
        csv = convert_df(pd.DataFrame(output))
        st.download_button(
            "Download an Example 128-Step Bac Input CSV File",
            data=csv,
            file_name='B-Input.csv',
            mime='text/csv', 
            )
    
        inputB = st.file_uploader(
            "CSV File for Bac in Single Cycle; Default: 100 mT Sinusoidal with 128-Steps",
            type='csv',
            key=f'bfile {m}',
            help=None
                )

        if inputB is not None:  # user input
            st.write("Remove the uploaded file for Default Inputs")
            df_input = pd.read_csv(inputB)
            st.write(df_input)
            bdata = 100 * np.sin(np.linspace(0.0, 2*np.pi, 128))  # TODO read the file
        else:  # default input for display
            st.write("\n\n\n")
            default = st.radio(
                "Select One of the Default Inputs for a Quick Start 👇",
                ["Sinusoidal", "Triangular", "Trapezoidal"])
            if default == "Sinusoidal":
                bdata = 100 * np.sin(np.linspace(0.0, 2*np.pi, 128))
            if default == "Triangular":
                bdata = np.interp(np.linspace(0, 1, 128), np.array([0, 0.5, 1]), np.array([-100, 100, -100]))
            if default == "Trapezoidal":
                bdata = np.interp(np.linspace(0, 1, 128), np.array([0, 0.1, 0.4, 0.6, 0.9, 1]), np.array([0, 100, 100, -100, -100, 0]))

        hdata = BH_Transformer(material, freq, temp, bias, bdata)
        output = {'B [mT]': bdata + bias * mu_relative * mu0 * 1e3 * np.ones(128),
                  'H [A/m]': hdata + bias * np.ones(128)}
        loss = loss_BH(bdata / 1e3, hdata, freq)
        csv = convert_df(pd.DataFrame(output))

        if max(abs(bdata)) > round(max(df['Flux_Density']) * 1e3):
            st.warning(f"The model has not been trained for peak flux densities above {round(max(df['Flux_Density']) * 1e3)} mT")

        voltage_max = 160  # max peak to peak voltage in the setup
        effective_area = material_core_params[material][1]
        number_of_turns = material_core_params[material][2]
        dbdt_max = voltage_max / (effective_area * number_of_turns)
        flag_dbdt_high = 0  # Detection of large dB/dt TODO test this limit and check if this is the case
        for i in range(0, len(bdata)-1):
            if abs(bdata[i + 1] - bdata[i]) * 1e-3 * freq * 128 > dbdt_max:
                flag_dbdt_high = 1
        if flag_dbdt_high == 1:
            st.warning(f"The model has not been trained dB/dt above {round(dbdt_max * 1e-3)} mT/us")

        flag_minor_loop = 0  # Detection of minor loops TODO test it once data is read
        if np.argmin(bdata) < np.argmax(bdata):  # min then max
            for i in range(np.argmin(bdata), np.argmax(bdata)):
                if bdata[i + 1] < bdata[i]:
                    flag_minor_loop = 1
            for i in range(np.argmax(bdata), len(bdata)-1):
                if bdata[i + 1] > bdata[i]:
                    flag_minor_loop = 1
            for i in range(0, np.argmin(bdata)):
                if bdata[i + 1] > bdata[i]:
                    flag_minor_loop = 1
        else:  # max then min
            for i in range(0, np.argmax(bdata)):
                if bdata[i + 1] < bdata[i]:
                    flag_minor_loop = 1
            for i in range(np.argmin(bdata), len(bdata)-1):
                if bdata[i + 1] < bdata[i]:
                    flag_minor_loop = 1
            for i in range(np.argmax(bdata), np.argmin(bdata)):
                if bdata[i + 1] > bdata[i]:
                    flag_minor_loop = 1
        if flag_minor_loop == 1:
            st.warning(f"The model has not been trained for flux densities with minor loops")

    st.markdown("""---""")
    st.header('MagNet AI Predicted Results')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('B-H Waveform')
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=np.linspace(1, 128, num=128)/128,
                y=bdata + bias*mu_relative*mu0*1e3 * np.ones(128),
                line=dict(color='mediumslateblue', width=4),
                name="B [mT]"),
            secondary_y=False,
            )
        fig.add_trace(
            go.Scatter(
                x=np.linspace(1, 128, num=128)/128,
                y=bias*mu_relative*mu0*1e3 * np.ones(128),
                line=dict(color='mediumslateblue', dash='longdash', width=2),
                name="Bdc [mT]"),
            secondary_y=False,
            )                
        fig.add_trace(
            go.Scatter(
                x=np.linspace(1, 128, num=128)/128,
                y=hdata+bias * np.ones(128), 
                line=dict(color='firebrick', width=4),
                name="H [A/m]"),
            secondary_y=True,
            )
        fig.add_trace(
            go.Scatter(
                x=np.linspace(1, 128, num=128)/128,
                y=bias * np.ones(128), 
                line=dict(color='firebrick', dash='longdash', width=2),
                name="Hdc [A/m]"),
            secondary_y=True,
            )

        fig.update_xaxes(title_text="Fraction of a Cycle")
        fig.update_yaxes(title_text="B - Flux Density [mT]", color='mediumslateblue', secondary_y=False, zeroline=False, zerolinewidth=1.5, zerolinecolor='gray')
        fig.update_yaxes(title_text="H - Field Strength [A/m]", color='firebrick', secondary_y=True, zeroline=False, zerolinewidth=1.5, zerolinecolor='gray')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader('B-H Loop')
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(
            go.Scatter(
                x=np.tile(hdata + bias * np.ones(128), 2),
                y=np.tile(bdata + bias*mu_relative*mu0*1e3 * np.ones(128),2),
                line=dict(color='mediumslateblue', width=4),
                name="Predicted B-H Loop"),
            secondary_y=False,
            )
        fig.add_trace(
            go.Scatter(
                x=bdata / 1e3 / mu_relative / mu0 + bias * np.ones(128),
                y=bdata + bias*mu_relative*mu0*1e3 * np.ones(128),
                line=dict(color='firebrick', dash='longdash', width=2),
                name="Calculated B = mu * H"),
            secondary_y=False,
            )
        fig.add_trace(
            go.Scatter(
                x=np.array(0),
                y=np.array(0),
                line=dict(color='gray', width=0.25),
                showlegend = False),
            secondary_y=False,
            )

        fig.update_yaxes(title_text="B - Flux Density [mT]", zeroline=True, zerolinewidth=1.5, zerolinecolor='gray')
        fig.update_xaxes(title_text="H - Field Strength [A/m]",  zeroline=True, zerolinewidth=1.5, zerolinecolor='gray')
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
