import os.path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from magnet.constants import material_list, material_extra, material_core_params
from magnet.io import load_dataframe, load_hull
import numpy as np
from magnet.core import BH_Transformer, loss_BH, bdata_generation, point_in_hull
from magnet import config as c

STREAMLIT_ROOT = os.path.dirname(__file__)

def convert_df(df):
    return df.to_csv().encode('utf-8')

def ui_intro(m):
    
    st.title('MagNet AI for Research, Education and Design')
    st.subheader('"It is time to upgrade the Steinmetz Equation!" - Try MagNet AI and join us to make it better')
    st.caption('We created MagNet AI to advance power magnetics research, education, and design. The mission of MagNet AI is to replace the traditional curve-fitting models (e.g., Steinmetz Equations and Jiles-Atherton Models) with state-of-the-art data-driven methods such as neural networks and machine learning. MagNet AI is open, transparent, fast, smart, versatile, and is continously learning. It is a new tool to design power magnetics and can do lots of things that traditional methods cannnot do.')
    st.markdown("""---""")
    
    col1, col2 = st.columns(2)
    with col1:
        st.header('MagNet AI Input')
        material = st.selectbox(
            f'Material:',
            material_list,
            index=9,
            key=f'material {m}',
            help='select from a list of available materials')
        
        mu_relative = material_extra[material][0]
        st.caption(f'Initial Relative Permeability (mu) of {material} is set to {mu_relative} to determine the center of the predicted B-H loop.')

        dataset = load_dataframe(material)  # To find the range of the variables

        temp = st.slider(
            "Temperature [C]",
            0.0,
            120.0,
            25.0,
            1.0,
            format='%f',
            key=f'temp {m}',
            help='Device surface temperature')

        if temp < min(dataset['Temperature']):
            st.warning(
                f"For temperature below {round(min(dataset['Temperature']))} C. Results are potentially extrapolated.")
        if temp > max(dataset['Temperature']):
            st.warning(
                f"For temperature above {round(max(dataset['Temperature']))} C. Results are potentially extrapolated.")

        freq = st.slider(
            "Frequency [kHz]",
            10.0,
            1000.0,
            200.0,
            1.0,
            format='%f',
            key=f'freq {m}',
            help='Fundamental frequency of the excitation') * 1e3

        if freq < min(dataset['Frequency']):
            st.warning(
                f"For frequency below {round(min(dataset['Frequency']) * 1e-3)} kHz. "
                f"Results are potentially extrapolated.")
        if freq > max(dataset['Frequency']):
            st.warning(
                f"For frequency above {round(max(dataset['Frequency']) * 1e-3)} kHz. "
                f"Results are potentially extrapolated.")
        
    with col2:
        st.subheader('Option 1: Arbitrary B Input')  # Create an example Bac input file
        bdata0 = 100 * np.sin(np.linspace(0, 2*np.pi, c.streamlit.n_nn))
        output = {'B [mT]': bdata0}
        csv = convert_df(pd.DataFrame(output))
        st.write(f"Describe a single cycle waveform of Bac in mT. Expected for a {c.streamlit.n_nn}-points array that describes the waveform in a single cycle of steady state. The dc bias is automatically detected. Arrays with other lengths will be automatically interpolated. Here's a template for your reference:")
        st.download_button(
            f"Download an Example {c.streamlit.n_nn}-Step-per-Cycle 100 mT Sinusoidal B Waveform CSV File",
            data=csv,
            file_name='B-Input.csv',
            mime='text/csv',
        )

        inputB = st.file_uploader(
            "Upload the User-defined CSV File Here:",
            type='csv',
            key=f'bfile {m}'
        )
        
        st.markdown("""---""")
        
        st.subheader('Option 2: Standard B Input')  # Create an example Bac input file
        if inputB is None:  # default input for display
            default = st.radio(  # TODO disable radio button and make horizontal with new streamlit version
                "Select one of the default inputs for a quick start 🡻",
                ["Sinusoidal", "Triangular", "Trapezoidal"])
            flux = st.slider(
                "Bac Amplitude [mT]",
                10.0,
                350.0,
                50.0,
                1.0,
                format='%f',
                key=f'flux_sine {m}',
                help='Half of the peak-to-peak flux density') / 1e3
            if default == "Sinusoidal":
                duty = None
                dd = 0.5
            if default == "Triangular":
                duty = st.slider(
                    "Duty Cycle",
                    0.0,
                    1.0,
                    0.5,
                    0.01,
                    format='%f',
                    key=f'duty_tri {m}',
                    help='Duty cycle of the rising part.')
                dd = duty
            if default == "Trapezoidal":
                duty_p = st.slider(
                    "Duty Cycle (Rising)",
                    0.01,
                    1-0.01,
                    0.2,
                    0.01,
                    format='%f',
                    key=f'duty_trap_p1 {m}',
                    help='Duty cycle of the rising part.')
                duty_n = st.slider(
                    "Duty Cycle (Falling)",
                    0.01,
                    round((1-duty_p)/0.01)*0.01,
                    duty_p if duty_p<=0.5 else round((1-duty_p)/2/0.01-1)*0.01,
                    0.01,
                    format='%f',
                    key=f'duty_trap_p2 {m}',
                    help='Duty cycle of the falling part.')
                duty = [duty_p, duty_n, (1-duty_p-duty_n)/2]
                dd = duty[0]
            phase = st.slider(
                "Starting Phase",
                0.0,
                360.0,
                0.0,
                1.0,
                format='%f',
                key=f'phase_trap {m}',
                help='Shift the waveform horizontally. '
                     'Theoretically, this won\'t change the B-H loop nor the core loss.') / 360.0

            bdata_start0 = bdata_generation(flux, duty)
            bdata = np.roll(bdata_start0, np.int_(phase * c.streamlit.n_nn))

        if inputB is not None:  # user input
            df = pd.read_csv(inputB)
            st.write("Default inputs have been disabled as the following user-defined waveform is uploaded:")
            st.write(df.T)
            st.write(
                "To remove the uploaded file and reactivate the default input, click on the cross on the right side")
            bdata_read = df["B [mT]"].to_numpy()
            bdata = np.interp(np.linspace(0, 1, c.streamlit.n_nn), np.linspace(0, 1, len(bdata_read)), bdata_read * 1e-3)

            flag_minor_loop = 0  # Detection of minor loops TODO test it once data is read
            if np.argmin(bdata) < np.argmax(bdata):  # min then max
                for i in range(np.argmin(bdata), np.argmax(bdata)):
                    if bdata[i + 1] < bdata[i]:
                        flag_minor_loop = 1
                for i in range(np.argmax(bdata), len(bdata) - 1):
                    if bdata[i + 1] > bdata[i]:
                        flag_minor_loop = 1
                for i in range(0, np.argmin(bdata)):
                    if bdata[i + 1] > bdata[i]:
                        flag_minor_loop = 1
            else:  # max then min
                for i in range(0, np.argmax(bdata)):
                    if bdata[i + 1] < bdata[i]:
                        flag_minor_loop = 1
                for i in range(np.argmin(bdata), len(bdata) - 1):
                    if bdata[i + 1] < bdata[i]:
                        flag_minor_loop = 1
                for i in range(np.argmax(bdata), np.argmin(bdata)):
                    if bdata[i + 1] > bdata[i]:
                        flag_minor_loop = 1
            if flag_minor_loop == 1:
                st.warning(
                    f"The models has not been trained by waveforms with minor loops. "
                    f"Results are potentially extrapolated.")

        with col1:
            if inputB is not None:  # user input
                bias = np.average(bdata) / (mu_relative * c.streamlit.mu_0)
                bdata = bdata - bias * mu_relative * c.streamlit.mu_0  # Removing the average B for the NN
                st.write(f'DC Bias of {round(bias)} A/m based on input B waveform and mu={mu_relative}')
            else:
                bias = st.slider(
                    "Hdc Bias [A/m]",
                    -20.0,
                    40.0,
                    0.0,
                    1.0,
                    format='%f',
                    key=f'bias {m}',
                    help='Determined by the bias dc current')
        
            st.write('The next step is to describe the B waveform with two options.')
            st.markdown("""---""")
            st.subheader('How does MagNet AI work?')
            st.caption('from data acquisition, error analysis, data visualization, to machine learning framework')
            st.write('- D. Serrano et al., "Why MagNet: Quantifying the Complexity of Modeling Power Magnetic Material Characteristics," in IEEE Transactions on Power Electronics, doi: 10.1109/TPEL.2023.3291084. [Paper](https://ieeexplore.ieee.org/document/10169101)')
            st.write('- H. Li et al., "How MagNet: Machine Learning Framework for Modeling Power Magnetic Material Characteristics," in IEEE Transactions on Power Electronics, doi: 10.1109/TPEL.2023.3309232. [Paper](https://ieeexplore.ieee.org/document/10232863)')

            
            if bias < 0:
                st.warning(f"For bias below 0 A/m, results are potentially extrapolated.")
            if bias > max(dataset['DC_Bias']):
                st.warning(
                    f"For bias above {round(max(dataset['DC_Bias']))} A/m, results are potentially extrapolated.")

        with col2:
            if max(abs(bdata)) + bias * mu_relative * c.streamlit.mu_0 > max(dataset['Flux_Density']):
                st.warning(
                    f"For peak flux densities above {round(max(dataset['Flux_Density']) * 1e3)} mT, results are potentially extrapolated."
                    f" (Bac={round((max(bdata)-min(bdata))/2 * 1e3)} mT, Bdc={round(bias * mu_relative * c.streamlit.mu_0 * 1e3)} mT).")

            flag_dbdt_high = 0  # Detection of large dB/dt
            dbdt_max = c.streamlit.vpkpk_max/(material_core_params[material][2] * material_core_params[material][1])
            for i in range(0, len(bdata) - 1):
                if abs(bdata[i + 1] - bdata[i]) * freq * c.streamlit.n_nn > dbdt_max:  # dbdt_max=vpkpk_max/N/Ae
                    flag_dbdt_high = 1
            if flag_dbdt_high == 1:
                st.warning(f"For dB/dt above {round(dbdt_max * 1e-3)} mT/ns, results are potentially extrapolated.")
    
    hdata = BH_Transformer(material, freq, temp, bias, bdata)
    loss = loss_BH(bdata, hdata, freq)
    
    Eq = load_hull(material)
    if inputB is None:
        point = np.array([freq, flux, bias, temp, dd])
        not_extrapolated = point_in_hull(point,Eq)
    else:
        point = np.array([freq, (max(bdata)-min(bdata))/2, bias, temp, 0.5]) #TODO: access the user-defined waveform
        not_extrapolated = point_in_hull(point,Eq)

    st.markdown("""---""")
    st.header('MagNet AI Output')
    st.caption('The data contains measurement artifacts. The B-H loop and volumetric core losses describe component-level behaviors. Material characteristics, parasitics, and measurement error all impact the results.')
    
    if not not_extrapolated:
        st.warning("The specified condition is out of the range of training data.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('Effective B-H Waveform')
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=np.linspace(1, c.streamlit.n_nn, num=c.streamlit.n_nn) / c.streamlit.n_nn,
                y=(bdata + bias * mu_relative * c.streamlit.mu_0 * np.ones(c.streamlit.n_nn)) * 1e3,
                line=dict(color='mediumslateblue', width=4),
                name="B [mT]"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=np.linspace(1, c.streamlit.n_nn, num=c.streamlit.n_nn) / c.streamlit.n_nn,
                y=(bias * mu_relative * c.streamlit.mu_0 * np.ones(c.streamlit.n_nn)) * 1e3,
                line=dict(color='mediumslateblue', dash='longdash', width=2),
                name="Bdc [mT]"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=np.linspace(1, c.streamlit.n_nn, num=c.streamlit.n_nn) / c.streamlit.n_nn,
                y=hdata + bias * np.ones(c.streamlit.n_nn),
                line=dict(color='firebrick', width=4),
                name="H [A/m]"),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=np.linspace(1, c.streamlit.n_nn, num=c.streamlit.n_nn) / c.streamlit.n_nn,
                y=bias * np.ones(c.streamlit.n_nn),
                line=dict(color='firebrick', dash='longdash', width=2),
                name="Hdc [A/m]"),
            secondary_y=True,
        )

        fig.update_xaxes(title_text="Fraction of a Cycle")
        fig.update_yaxes(title_text="B - Flux Density [mT]", color='mediumslateblue', secondary_y=False, zeroline=False,
                         zerolinewidth=1.5, zerolinecolor='gray')
        fig.update_yaxes(title_text="H - Field Strength [A/m]", color='firebrick', secondary_y=True, zeroline=False,
                         zerolinewidth=1.5, zerolinecolor='gray')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader('Effective B-H Loop')
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(
            go.Scatter(
                x=np.tile(hdata + bias * np.ones(c.streamlit.n_nn), 2),
                y=np.tile((bdata + bias * mu_relative * c.streamlit.mu_0 * np.ones(c.streamlit.n_nn)) * 1e3, 2),
                line=dict(color='mediumslateblue', width=4),
                name="Predicted B-H Loop"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=[min(bdata) / mu_relative / c.streamlit.mu_0 + bias, max(bdata) / mu_relative / c.streamlit.mu_0 + bias],
                y=[(min(bdata) + bias * mu_relative * c.streamlit.mu_0) * 1e3, (max(bdata) + bias * mu_relative * c.streamlit.mu_0) * 1e3],
                line=dict(color='firebrick', dash='longdash', width=2),
                name="Bdc = mu * Hdc"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=np.array(0),
                y=np.array(0),
                line=dict(color='gray', width=0.25),
                showlegend=False),
            secondary_y=False,
        )

        fig.update_yaxes(title_text="B - Flux Density [mT]", zeroline=True, zerolinewidth=1.5, zerolinecolor='gray')
        fig.update_xaxes(title_text="H - Field Strength [A/m]", zeroline=True, zerolinewidth=1.5, zerolinecolor='gray')
        st.plotly_chart(fig, use_container_width=True)

    with col1:

        output = {'B [mT]': (bdata + bias * mu_relative * c.streamlit.mu_0 * np.ones(c.streamlit.n_nn)) * 1e3,
                  'H [A/m]': hdata + bias * np.ones(c.streamlit.n_nn)}
        csv = convert_df(pd.DataFrame(output))

        st.download_button(
            "Download the B-H Loop as a CSV File",
            data=csv,
            file_name='BH-Loop.csv',
            mime='text/csv',
            )

    with col3:
        st.subheader(f'Volumetric Loss: {np.round(loss / 1e3, 2)} kW/m^3')
        st.subheader('Ranking among included materials:')
    
        loss_test_list = pd.DataFrame(columns=['Material','Core Loss [kW/m^3]','This one'])
        for material_test in material_list:
            hdata_test = BH_Transformer(material_test, freq, temp, bias, bdata)
            loss_test = loss_BH(bdata, hdata_test, freq)
            this_one = '   ✓' if (material_test==material) else ''
            loss_test_list = loss_test_list.append({
                'Material':material_test,
                'Core Loss [kW/m^3]': np.round(loss_test / 1e3, 2),
                'This one': this_one}, ignore_index=True)
        
        loss_test_list=loss_test_list.sort_values(by='Core Loss [kW/m^3]')
        
        # loss_test_list.index = [''] * len(loss_test_list) # hide index
        loss_test_list.index = range(1, len(loss_test_list) + 1) # re-index from 1 to 10
        
        st.dataframe(data=loss_test_list, width=None, height=None)
    
    
    st.markdown("""---""")

    st.header('How to Cite')
    st.write("""
        If you find MagNet as useful, please cite the following papers as a trilogy:

        - D. Serrano et al., "Why MagNet: Quantifying the Complexity of Modeling Power Magnetic Material Characteristics," in IEEE Transactions on Power Electronics, doi: 10.1109/TPEL.2023.3291084. [Paper](https://ieeexplore.ieee.org/document/10169101)

        - H. Li et al., "How MagNet: Machine Learning Framework for Modeling Power Magnetic Material Characteristics," in IEEE Transactions on Power Electronics, doi: 10.1109/TPEL.2023.3309232. [Paper](https://ieeexplore.ieee.org/document/10232863)

        - H. Li, D. Serrano, S. Wang and M. Chen, "MagNet-AI: Neural Network as Datasheet for Magnetics Modeling and Material Recommendation," in IEEE Transactions on Power Electronics, doi: 10.1109/TPEL.2023.3309233. [Paper](https://ieeexplore.ieee.org/document/10232911)

    """)
    st.markdown("""---""")
