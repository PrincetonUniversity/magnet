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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('Input Information')
        material = st.selectbox(
            f'Material:',
            material_list,
            key=f'material {m}',
            help='select from a list of available materials')
        
        mueff = materials_extra[material][0]
        st.write(f'Initial Relative Permeability (mu) of {material} is set to {mueff} to determine the center of the B-H loop.')

        dataset = load_dataframe(material)  # To find the range of the variables
        
        temp = st.slider(
                "Temperature [C]",
                -50.0,
                200.0,
                25.0,
                1.0,
                format='%f',
                key=f'temp {m}',
                help='Device surface temperature')
                
        if temp < round(min(dataset['Temperature'])):
            st.warning(f"For temperature below {round(min(dataset['Temperature']))} C, results are potentially extrapolated.")
        if temp > round(max(dataset['Temperature'])):
            st.warning(f"For temperature above {round(max(dataset['Temperature']))} C, results are potentially extrapolated.")
            
        freq = st.slider(
                "Frequency [kHz]",
                10.0,
                600.0,
                100.0,
                10.0,
                format='%f',
                key=f'freq {m}',
                help='Fundamental frequency of the excitation') * 1e3    

        if freq < round(min(dataset['Frequency'])):
            st.warning(f"For frequency below {round(min(dataset['Frequency']) * 1e-3)} kHz, results are potentially extrapolated.")
        if freq > round(max(dataset['Frequency'])):
            st.warning(f"For frequency above {round(max(dataset['Frequency']) * 1e-3)} kHz, results are potentially extrapolated.")

        bias = st.slider(
                "Hdc Bias [A/m]",
                -20.0,
                60.0,
                0.0,
                2.0,
                format='%f',
                key=f'bias {m}',
                help='Determined by the bias dc current')

        if bias < 0:
            st.warning(f"For bias below 0 A/m, results are potentially extrapolated.")
        if bias > round(max(dataset['DC_Bias'])):
            st.warning(f"For bias above {round(max(dataset['DC_Bias']))} A/m, results are potentially extrapolated.")

    with col3:
        st.subheader('User-defined Waveform')  # Create an example Bac input file
        bdata0 = 100 * np.sin(np.linspace(0, 2*np.pi, 128))
        output = {'B [mT]': bdata0}
        csv = convert_df(pd.DataFrame(output))
        st.write("Describe the single cycle waveform of Bac. Here's a template for your reference:")
        st.download_button(
            "Download an Example 128-Step Bac Waveform CSV File. Default: 100 mT Sinusoidal.",
            data=csv,
            file_name='B-Input.csv',
            mime='text/csv', 
            )
    
        inputB = st.file_uploader(

            "Upload the User-defined CSV File Here:",
            type='csv',
            key=f'bfile {m}',
            help="Expected for a 128-points array that describes the waveform in a single cycle of steady state. \n Arrays with other lengths will be automatically interpolated."
            )
        
    with col2:    
        st.subheader('Waveform Input (Unit: mT)')  # Create an example Bac input file
        default = st.radio(
        "Select one of the default inputs for a quick start 🡻, or provide your user-defined waveform 🡺",
        ["Sinusoidal", "Triangular", "Trapezoidal"])

        if inputB is None:  # default input for display
        
            if default == "Sinusoidal":
                flux = st.slider(
                    "Bac Amplitude [mT]",
                    10.0,
                    350.0,
                    100.0,
                    2.0,
                    format='%f',
                    key=f'flux_sine {m}',
                    help='Half of the peak-to-peak flux density')
                phase = st.slider(
                    "Starting Phase [p.u.]",
                    0.0,
                    1.0,
                    0.0,
                    0.05,
                    format='%f',
                    key=f'phase_sine {m}',
                    help='Shift the waveform horizontally. Theoretically, this won\'t change the B-H loop nor the core loss.')
                bdata = flux * np.sin(np.linspace(0.0, 2*np.pi, 128))
                bdata = np.roll(bdata, np.int_(phase*128))
                
            if default == "Triangular":          
                flux = st.slider(
                    "Bac Amplitude [mT]",
                    10.0,
                    350.0,
                    100.0,
                    2.0,
                    format='%f',
                    key=f'flux_tri {m}',
                    help='Half of the peak-to-peak flux density')           
                duty = st.slider(
                    "Duty Ratio [p.u.]",
                    0.0,
                    1.0,
                    0.5,
                    0.02,
                    format='%f',
                    key=f'duty_tri {m}',
                    help='Duty ratio of the rising part.')
                phase = st.slider(
                    "Starting Phase [p.u.]",
                    0.0,
                    1.0,
                    0.0,
                    0.05,
                    format='%f',
                    key=f'phase_tri {m}',
                    help='Shift the waveform horizontally. Theoretically, this won\'t change the B-H loop nor the core loss.')
                bdata = np.interp(np.linspace(0,1,128), np.array([0,duty,1]), np.array([-flux,flux,-flux]))
                bdata = np.roll(bdata, np.int_(phase*128))
                
            if default == "Trapezoidal":
                flux = st.slider(
                    "Bac Amplitude [mT]",
                    10.0,
                    350.0,
                    100.0,
                    2.0,
                    format='%f',
                    key=f'flux_trap {m}',
                    help='Half of the peak-to-peak flux density')           
                duty = st.slider(
                    "Duty Ratio [p.u.]",
                    0.0,
                    0.5,
                    0.2,
                    0.02,
                    format='%f',
                    key=f'duty_trap {m}',
                    help='Duty ratio of the rising part.')
                phase = st.slider(
                    "Starting Phase [p.u.]",
                    0.0,
                    1.0,
                    0.0,
                    0.05,
                    format='%f',
                    key=f'phase_trap {m}',
                    help='Shift the waveform horizontally. Theoretically, this won\'t change the B-H loop nor the core loss.')
                bdata = np.interp(np.linspace(0,1,128), np.array([0,duty/2,0.5-duty/2,0.5+duty/2,1-duty/2,1]), np.array([0,flux,flux,-flux,-flux,0]))
                bdata = np.roll(bdata, np.int_(phase*128))
                
            hdata = BH_Transformer(material, freq, temp, bias, bdata)
            output = {'B [mT]': bdata + bias*mueff*mu0*1e3 * np.ones(128), 
                      'H [A/m]': hdata + bias * np.ones(128)}
            loss = loss_BH(bdata/1e3,hdata,freq)
            csv = convert_df(pd.DataFrame(output))

        if inputB is not None:  # user input
            df = pd.read_csv(inputB)
            st.write("Default inputs have been disabled as the following user-defined waveform is uploaded:")
            st.write(df)
            st.write("To remove the uploaded file and reactivate the default input, click on the cross on the right side🡽")
            bdata = df["B [mT]"].to_numpy()
            bdata = np.interp(np.linspace(0,1,128), np.linspace(0,1,len(bdata)), bdata)
            
            if max(abs(bdata)) > round(max(dataset['Flux_Density']) * 1e3):
                st.warning(f"For peak flux densities above {round(max(df['Flux_Density']) * 1e3)} mT, results are potentially extrapolated.")

            flag_dbdt_high = 0  # Detection of large dB/dt TODO test this limit and check if this is the case
            for i in range(0, len(bdata)-1):
                if abs(bdata[i + 1] - bdata[i]) * freq / 128 > 3e6:
                    flag_dbdt_high = 1
            if flag_dbdt_high == 1:
                st.warning(
                    f"For dB/dt above 3 mT/ns, results are potentially extrapolated.")
    
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
                st.warning(f"The models has not been trained for waveforms with minor loops. Results are potentially unreliable.")
            
            hdata = BH_Transformer(material, freq, temp, bias, bdata)
            output = {'B [mT]': bdata + bias*mueff*mu0*1e3 * np.ones(128), 
                      'H [A/m]': hdata + bias * np.ones(128)}
            loss = loss_BH(bdata/1e3,hdata,freq)
            csv = convert_df(pd.DataFrame(output))

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
        
    with col1:  
        st.subheader(f'Volumetric Loss: {np.round(loss,2)} kW/m^3')

    with col2:
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
