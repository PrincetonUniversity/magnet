import re
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from magnet.constants import material_names, excitations
from magnet.io import load_dataframe
from magnet.plots import power_loss_scatter_plot, waveform_visualization, core_loss_multiple
from magnet.core import loss
from magnet.simplecs.simfunctions import SimulationPLECS


def ui_core_loss_db(material_selector):
    m = material_selector  # short variable name

    st.sidebar.header(f'Information for Material {m}')
    material_type = st.sidebar.selectbox(f'Material f{m}:', material_names)
    excitation_type = st.sidebar.selectbox(f'Excitation {m}:', excitations)

    [Fmin, Fmax] = st.sidebar.slider(
        f'Frequency Range {m} (Hz)',
        10000,
        500000,
        (10000, 500000),
        step=1000
    )
    [Bmin, Bmax] = st.sidebar.slider(
        f'Flux Density Range {m} (mT)',
        10,
        300,
        (10, 300),
        step=1
    )

    if excitation_type in ('Datasheet', 'Sinusoidal'):
        st.header("**" + material_type + ", " + excitation_type + ", f=[" + str(Fmin) + "~" + str(Fmax) + "] Hz"
                  + ", B=[" + str(Bmin) + "~" + str(Bmax) + "] mT" + "**")

        df = load_dataframe(material_type, excitation_type, Fmin,Fmax, Bmin, Bmax)

        if df.empty:
            st.write("Warning: No Data in Range")
        else:
            col1, col2 = st.beta_columns(2)
            with col1:
                st.plotly_chart(power_loss_scatter_plot(df, x='Frequency', color_prop='Flux_Density'), use_container_width=True)
            with col2:
                st.plotly_chart(power_loss_scatter_plot(df, x='Flux_Density', color_prop='Frequency'), use_container_width=True)

    if excitation_type == 'Triangle':
        Duty = st.sidebar.multiselect(
            f'Duty Ratio {m}',
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        Margin = st.sidebar.slider(f'Duty Ratio Margin {m}', 0.0, 1.0, 0.01, step=0.01)

        st.header("**" + material_type + ", " + excitation_type + ", f=[" + str(Fmin) + "~" + str(Fmax) + "] Hz"
                  + ", B=[" + str(Bmin) + "~" + str(Bmax) + "] mT" + ", D=" + str(Duty) + "**")

        df = load_dataframe(material_type, excitation_type, Fmin,Fmax, Bmin, Bmax, Duty, Margin)

        if df.empty:
            st.write("Warning: No Data in Range")
        else:
            col1, col2 = st.beta_columns(2)
            with col1:
                st.plotly_chart(power_loss_scatter_plot(df, x='Frequency', color_prop='Duty_Ratio'), use_container_width=True)
            with col2:
                st.plotly_chart(power_loss_scatter_plot(df, x='Flux_Density', color_prop='Duty_Ratio'), use_container_width=True)

    if excitation_type == 'Trapezoidal':
        Duty = st.sidebar.multiselect(f'Duty Ratio {m}',
                                       [0.1414, 0.2323, 0.3232, 0.3313, 0.4141, 0.4222, 0.5131, 0.5212, 0.6121],
                                       [0.1414, 0.2323, 0.3232, 0.3313, 0.4141, 0.4222, 0.5131, 0.5212, 0.6121])
        Margin = st.sidebar.slider(f'Duty Ratio Margin {m}', 0.0, 1.0, 0.01, step=0.01)

        st.header("**" + material_type + ", " + excitation_type + ", f=[" + str(Fmin) + "~" + str(Fmax) + "] Hz"
                  + ", B=[" + str(Bmin) + "~" + str(Bmax) + "] mT" + ", D=" + str(Duty) + "**")
        st.header("Note: D=0.2332 means **20% Up + 30% Flat + 30% Down + 20% Flat** from left to right")

        df = load_dataframe(material_type, excitation_type, Fmin,Fmax, Bmin, Bmax, Duty, Margin)

        if df.empty:
            st.write("Warning: No Data in Range")
        else:
            col1, col2 = st.beta_columns(2)
            with col1:
                st.plotly_chart(power_loss_scatter_plot(df, x='Frequency', color_prop='Duty_Ratio'), use_container_width=True)
            with col2:
                st.plotly_chart(power_loss_scatter_plot(df, x='Flux_Density', color_prop='Duty_Ratio'), use_container_width=True)

    st.sidebar.markdown("""---""")
    st.markdown("""---""")


def ui_core_loss_predict():
    fluxplot_list = [10, 12, 15, 17, 20, 25, 30, 40, 50, 60, 80, 100, 120, 160, 200, 240, 260, 300]
    freqplot_list = [10000, 15000, 20000, 30000, 40000, 50000, 60000, 80000, 100000, 120000, 160000, 200000, 240000,
                     320000, 400000, 480000]
    dutyplot_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    material_type = st.sidebar.selectbox('Material:', material_names)
    excitation_type = st.sidebar.selectbox('Excitation:', excitations + ("Arbitrary", "Simulated"))
    algorithm_type = st.sidebar.selectbox('Algorithm:', ('iGSE', 'Machine Learning'))

    if excitation_type in ('Datasheet', 'Sinusoidal'):

        col1, col2 = st.beta_columns(2)
        with col1:
            st.header("Please provide waveform information")
            Freq = st.slider('Frequency (Hz)', 10000, 500000, 250000, step=1000)
            Flux = st.slider('Peak to Peak Flux Density (mT)', 10, 300, 150, step=1)
            Bias = st.slider('DC Bias (mT)', -300, 300, 0, step=10)
            duty_list = np.linspace(0, 1, 101)
            flux_read = np.multiply(np.sin(np.multiply(duty_list, np.pi * 2)), Flux / 2)
            flux_list = np.add(flux_read, Bias)

        with col2:
            waveform_visualization(st, x=duty_list, y=flux_list)

        st.header(f'{material_type}, {excitation_type}, f={Freq} Hz, \u0394B={Flux} mT, Bias={Bias} mT')

        core_loss = loss(waveform='sine', algorithm=algorithm_type, material=material_type, freq=Freq, flux_p2p=Flux)
        st.title(f'{algorithm_type} Core Loss: {core_loss} kW/m^3')

        iGSE_coreloss_fluxplot = [loss(waveform='sine', algorithm='iGSE', material=material_type, freq=Freq, flux_p2p=i) for i in fluxplot_list]
        iGSE_coreloss_freqplot = [loss(waveform='sine', algorithm='iGSE', material=material_type, freq=i, flux_p2p=Flux) for i in freqplot_list]
        ML_coreloss_fluxplot = [loss(waveform='sine', algorithm='ML', material=material_type, freq=Freq, flux_p2p=i) for i in fluxplot_list]
        ML_coreloss_freqplot = [loss(waveform='sine', algorithm='ML', material=material_type, freq=i, flux_p2p=Flux) for i in freqplot_list]

        col1, col2 = st.beta_columns(2)

        with col1:
            core_loss_multiple(st, x=freqplot_list, y1=iGSE_coreloss_freqplot, y2=ML_coreloss_freqplot,
                               title=f'Core Loss with Fixed Flux Density {Flux} mT', x_title='Frequency [Hz]')

        with col2:
            core_loss_multiple(st, x=fluxplot_list, y1=iGSE_coreloss_fluxplot, y2=ML_coreloss_fluxplot,
                               title=f'Core Loss with Fixed Frequency {Freq} Hz', x_title='Flux Density [mT]')

    if excitation_type == "Triangle":
        col1, col2 = st.beta_columns(2)

        with col1:
            st.header("Please provide waveform information")
            Freq = st.slider('Frequency (Hz)', 10000, 500000, 250000, step=1000)
            Flux = st.slider('Peak to Peak Flux Density (mT)', 10, 300, 150, step=10)
            Duty = st.slider('Duty Ratio', 0.0, 1.0, 0.5, step=0.01)
            Bias = st.slider('DC Bias (mT)', -300, 300, 0, step=10)
            duty_list = [0, Duty, 1]
            flux_read = [0, Flux, 0]
            flux_mean = Flux / 2
            flux_diff = Bias - flux_mean
            flux_list = np.add(flux_read, flux_diff)

        with col2:
            waveform_visualization(st, x=duty_list, y=flux_list)

        st.header(f'{material_type}, {excitation_type}, f={Freq} Hz \u0394B={Flux} mT, D={Duty}, Bias={Bias} mT')

        core_loss = loss(waveform='sawtooth', algorithm=algorithm_type, material=material_type, freq=Freq, flux_p2p=Flux)
        st.title(f'{algorithm_type} Core Loss: {core_loss} kW/m^3')

        iGSE_coreloss_fluxplot = [loss(waveform='sawtooth', algorithm='iGSE', material=material_type, freq=Freq, flux_p2p=i, duty_ratio=Duty) for i in fluxplot_list]
        iGSE_coreloss_freqplot = [loss(waveform='sawtooth', algorithm='iGSE', material=material_type, freq=i, flux_p2p=Flux, duty_ratio=Duty) for i in freqplot_list]
        iGSE_coreloss_dutyplot = [loss(waveform='sawtooth', algorithm='iGSE', material=material_type, freq=Freq, flux_p2p=Flux, duty_ratio=i) for i in dutyplot_list]

        ML_coreloss_fluxplot = [loss(waveform='sawtooth', algorithm='ML', material=material_type, freq=Freq, flux_p2p=i, duty_ratio=Duty) for i in fluxplot_list]
        ML_coreloss_freqplot = [loss(waveform='sawtooth', algorithm='ML', material=material_type, freq=i, flux_p2p=Flux, duty_ratio=Duty) for i in freqplot_list]
        ML_coreloss_dutyplot = [loss(waveform='sawtooth', algorithm='ML', material=material_type, freq=Freq, flux_p2p=Flux, duty_ratio=i) for i in dutyplot_list]

        col1, col2, col3 = st.beta_columns(3)
        with col1:
            core_loss_multiple(st, x=freqplot_list, y1=iGSE_coreloss_freqplot, y2=ML_coreloss_freqplot,
                               title=f'Core Loss with F Sweep at {Flux} mT and D={Duty}', x_title='Frequency [Hz]')

        with col2:
            core_loss_multiple(st, x=fluxplot_list, y1=iGSE_coreloss_fluxplot, y2=ML_coreloss_fluxplot,
                               title=f'Core Loss with B Sweep at {Freq} Hz and D={Duty}', x_title='Flux Density [mT]')

        with col3:
            core_loss_multiple(st, x=dutyplot_list, y1=iGSE_coreloss_dutyplot, y2=ML_coreloss_dutyplot,
                               title=f'Core Loss with D Sweep at {Freq} Hz and {Flux} mT', x_title='Duty Ratio')

    if excitation_type == "Trapezoidal":
        col1, col2 = st.beta_columns(2)

        with col1:
            st.header("Please provide waveform information")
            Freq = st.slider('Frequency (Hz)', 10000, 500000, step=1000)
            Flux = st.slider('Peak to Peak Flux Density (mT)', 10, 300, step=10)
            Duty1 = st.slider('Duty Ratio 1', 0.0, 1.0, 0.25, step=0.01)
            Duty2 = st.slider('Duty Ratio 2', 0.0, 1.0, 0.5, step=0.01)
            Duty3 = st.slider('Duty Ratio 3', 0.0, 1.0, 0.75, step=0.01)
            Bias = st.slider('DC Bias (mT)', -300, 300, 0, step=10)
            duty_list = [0, Duty1, Duty2, Duty3, 1]
            flux_read = [0, Flux, Flux, 0, 0]
            flux_mean = Flux / 2
            flux_diff = Bias - flux_mean
            flux_list = np.add(flux_read, flux_diff)

        with col2:
            waveform_visualization(st, x=duty_list, y=flux_list)

        st.header(f'{material_type}, {excitation_type}, f={Freq} Hz, \u0394B={Flux} mT, D1={Duty1}, D2={Duty2}, D3={Duty3}, Bias={Bias} mT')

        duty_ratios = [Duty1, Duty2, Duty3]
        core_loss = loss(waveform='trapezoid', algorithm=algorithm_type, material=material_type, freq=Freq, flux_p2p=Flux, duty_ratios=duty_ratios)
        st.title(f'{algorithm_type} Core Loss: {core_loss} kW/m^3')

        iGSE_coreloss_fluxplot = [loss(waveform='trapezoid', algorithm='iGSE', material=material_type, freq=Freq, flux_p2p=i, duty_ratios=duty_ratios) for i in fluxplot_list]
        iGSE_coreloss_freqplot = [loss(waveform='trapezoid', algorithm='iGSE', material=material_type, freq=i, flux_p2p=Flux, duty_ratios=duty_ratios) for i in freqplot_list]
        ML_coreloss_fluxplot = [loss(waveform='trapezoid', algorithm='ML', material=material_type, freq=Freq, flux_p2p=i, duty_ratios=duty_ratios) for i in fluxplot_list]
        ML_coreloss_freqplot = [loss(waveform='trapezoid', algorithm='ML', material=material_type, freq=i, flux_p2p=Flux, duty_ratios=duty_ratios) for i in freqplot_list]

        col1, col2 = st.beta_columns(2)
        with col1:
            core_loss_multiple(st, x=freqplot_list, y1=iGSE_coreloss_freqplot, y2=ML_coreloss_freqplot,
                               title=f'Core Loss with Fixed Flux Density {Flux} mT', x_title='Frequency [Hz]')

        with col2:
            core_loss_multiple(st, x=fluxplot_list, y1=iGSE_coreloss_fluxplot, y2=ML_coreloss_fluxplot,
                               title=f'Core Loss with Fixed Frequency {Freq} Hz', x_title='Flux Density [mT]')

    if excitation_type == "Arbitrary":

        col1, col2 = st.beta_columns(2)
        with col1:
            Freq = st.slider('Cycle Frequency (Hz)', 10000, 500000, step=1000)
            duty_string = st.text_input('Waveform Pattern Duty in a Cycle (%)',
                                        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            flux_string = st.text_input('Waveform Pattern Relative Flux Density (mT)',
                                        [0, 10, 20, 10, 20, 30, -10, -30, 10, -10, 0])
            Bias = st.slider('DC Bias (mT)', -300, 300, 0, step=10)

            duty_split = re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", duty_string)
            flux_split = re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", flux_string)
            duty_list = [float(i) for i in duty_split]
            flux_read = [float(i) for i in flux_split]
            flux_mean = np.average(flux_read)
            flux_diff = Bias - flux_mean
            flux_list = np.add(flux_read, flux_diff)

        with col2:
            st.header("Waveform Visualization")
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=duty_list, y=flux_list,
                                      line=dict(color='firebrick', width=4)))
            fig5.update_layout(xaxis_title='Duty in a Cycle',
                               yaxis_title='Flux Density [mT]')
            st.plotly_chart(fig5, use_container_width=True)

        core_loss = loss(waveform='trapezoid', algorithm=algorithm_type, material=material_type, freq=Freq, flux=flux_list, frac_time=duty_list)
        st.title(f'{algorithm_type} Core Loss: {core_loss} kW/m^3')

    if excitation_type == "Simulated":
        core_loss = SimulationPLECS(material_type, algorithm_type)
        st.title(f'{algorithm_type} Core Loss: {core_loss} kW/m^3')
