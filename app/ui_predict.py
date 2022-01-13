import re
import numpy as np
import streamlit as st

from magnet import config as c
from magnet.constants import material_names, excitations, material_manufacturers
from magnet.plots import waveform_visualization, core_loss_multiple, waveform_visualization_2axes
from magnet.core import loss, waveform_shorts
from magnet.simplecs.simfunctions import SimulationPLECS
from magnet.core import cycle_points_sine, cycle_points_trap


def ui_core_loss_predict(m):
    st.sidebar.header(f'Information: Case {m}')
    excitation = st.sidebar.selectbox(
        f'Excitation:', excitations[1:] + ("Arbitrary", "Simulated"),
        key=f'excitation {m}')  # TBD
    excitation_short = waveform_shorts(excitation)

    material = st.sidebar.selectbox(
        f'Material:',
        material_names,
        key=f'material {m}')
    if excitation != "Simulated":

        freq = st.sidebar.slider(
            "Frequency (kHz)",
            c.streamlit.freq_min / 1e3,
            c.streamlit.freq_max / 1e3,
            c.streamlit.freq_max / 2 / 1e3,
            step=c.streamlit.freq_step / 1e3,
            key=f'freq {m}') * 1e3  # Use kHz for front-end demonstration while Hz for underlying calculation
        if excitation == "Arbitrary":
            flux_string = st.sidebar.text_input(
                f'Waveform Pattern - AC Flux Density (mT)',
                [0, 10, 20, 10, 20, 30, -10, -30, 10, -10, 0],
                key=f'flux {m}')
            duty_string = st.sidebar.text_input(
                f'Waveform Pattern - Duty Cycle (%)',
                [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                key=f'duty {m}')
        else:
            flux = st.sidebar.slider(
                f'AC Flux Density (mT)',
                c.streamlit.flux_min * 1e3,
                c.streamlit.flux_max * 1e3,
                c.streamlit.flux_max / 2 * 1e3,
                step=c.streamlit.flux_step * 1e3,
                key=f'flux {m}') / 1e3  # Use mT for front-end demonstration while T for underlying calculation
        flux_bias = st.sidebar.slider(
            f'DC Flux Density (mT) coming soon!',
            -c.streamlit.flux_max * 1e3,
            c.streamlit.flux_max * 1e3,
            0.0,
            step=1e9,
            key=f'bias {m}',
            help=f'Fixed at 0 mT for now') / 1e3  # Use mT for front-end demonstration while T for underlying

        if excitation == "Triangular":
            duty_p = st.sidebar.slider(
                f'Duty Ratio',
                c.streamlit.duty_min,
                c.streamlit.duty_max,
                (c.streamlit.duty_min + c.streamlit.duty_max) / 2,
                step=c.streamlit.duty_step,
                key=f'duty {m}')
            duty_n = 1 - duty_p
            duty_0 = 0
        if excitation == "Trapezoidal":
            duty_p = st.sidebar.slider(
                f'Duty Ratio (D1)',
                c.streamlit.duty_min,
                c.streamlit.duty_max,
                (c.streamlit.duty_min + c.streamlit.duty_max) / 2,
                step=c.streamlit.flux_step,
                key=f'dutyP {m}',
                help=f'Rising part with the highest slope')
            duty_n = st.sidebar.slider(
                f'Duty Ratio (D3)',
                c.streamlit.duty_min,
                1 - duty_p,
                max(round((1 - duty_p) / 2, 2), c.streamlit.duty_min),
                step=c.streamlit.duty_step,
                key=f'dutyN {m}',
                help=f'Falling part with the highest slope')
            duty_0 = st.sidebar.slider(
                "Duty Ratio (D2=D4=(1-D1-D3)/2)",  # TBD
                0.0,
                1.0,
                round((1 - duty_p - duty_n) / 2, 2),
                step=1e7,
                key=f'duty0 {m}',
                help=f'Low slope regions, fixed by D1 and D3, asymmetric D2 and D4 coming soon!')
            duty_ratios = [duty_p, duty_n, duty_0]

    st.sidebar.markdown("""---""")

    if excitation == "Sinusoidal":
        core_loss_iGSE = loss(
            waveform=excitation_short,
            algorithm='iGSE',
            material=material,
            freq=freq,
            flux=flux) / 1e3
        core_loss_ML = loss(
            waveform=excitation_short,
            algorithm='ML',
            material=material,
            freq=freq,
            flux=flux) / 1e3
    if excitation == "Triangular":
        core_loss_iGSE = loss(
            waveform=excitation_short,
            algorithm='iGSE',
            material=material,
            freq=freq,
            flux=flux,
            duty_ratio=duty_p) / 1e3
        core_loss_ML = loss(
            waveform=excitation_short,
            algorithm='ML',
            material=material,
            freq=freq,
            flux=flux,
            duty_ratio=duty_p) / 1e3
    if excitation == "Trapezoidal":
        core_loss_iGSE = loss(
            waveform=excitation_short,
            algorithm='iGSE',
            material=material,
            freq=freq,
            flux=flux,
            duty_ratios=duty_ratios) / 1e3
        core_loss_ML = loss(
            waveform=excitation_short,
            algorithm='ML',
            material=material,
            freq=freq,
            flux=flux,
            duty_ratios=duty_ratios) / 1e3
    if excitation == "Arbitrary":
        duty_list = [float(i) / 100 for i in re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", duty_string)]
        flux_read = [float(i) for i in re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", flux_string)]
        flux_list = np.multiply(np.add(flux_read, flux_bias), 1e-3)  # TBD
        core_loss_iGSE = loss(
            waveform=excitation_short,
            algorithm='iGSE',
            material=material,
            freq=freq,
            flux_list=flux_list,
            frac_time=duty_list) / 1e3
        # core_loss_ML = loss(waveform=excitation_short, algorithm='ML', material=material, freq=Freq, flux_list=flux_list, frac_time=duty_list) / 1e3
        core_loss_ML = 0

    col1, col2 = st.columns(2)
    with col1:
        st.title(f'Core Loss Analysis: Case {m}')
        if excitation == "Arbitrary":
            st.subheader('Coming Soon! The sequence-based NN model is still under development.')  # TBD
        elif excitation != "Simulated":
            st.subheader(f'{material_manufacturers[material]} - {material}, '
                         f'{excitation} Excitation')
            st.subheader(f'f={round(freq / 1e3)} kHz, '
                         f'B={round(flux * 1e3)} mT, '
                         f'Bias={flux_bias} mT')
            if excitation == "Triangular":
                st.subheader(f'D={round(duty_p, 2)}')
            if excitation == "Trapezoidal":
                st.subheader(f'D1={round(duty_p, 2)}, '
                             f'D2={round(duty_0, 2)}, '
                             f'D3={round(duty_n, 2)}, '
                             f'D4={round(duty_0, 2)}')
        if excitation != "Simulated":
            st.write("")
            st.subheader(f'Core Loss:')
            st.subheader(f'iGSE --- {round(core_loss_iGSE,2)} kW/m^3')
            st.subheader(f'ML ----- {round(core_loss_ML,2)} kW/m^3')

    with col2:
        if excitation == "Arbitrary":
            waveform_visualization(
                st,
                x=duty_list,
                y=np.multiply(flux_list, 1e3))
        elif excitation != "Simulated":
            if excitation == 'Sinusoidal':
                [cycle_list, flux_list, volt_list] = cycle_points_sine(c.streamlit.n_points_plot)
            if excitation in ['Triangular', 'Trapezoidal']:
                [cycle_list, flux_list, volt_list] = cycle_points_trap(duty_p, duty_n, duty_0)
            flux_vector = np.add(np.multiply(flux_list, flux), flux_bias)
            waveform_visualization_2axes(
                st,
                x1=np.multiply(cycle_list, 1e6 / freq),  # In us
                x2=cycle_list,  # Percentage
                y1=np.multiply(flux_vector, 1e3),  # In mT
                y2=volt_list,  # Per unit
                x1_aux=cycle_list,  # Percentage
                y1_aux=flux_list,
                title=f"<b>Waveform visualization</b>")

    if excitation == "Sinusoidal":
        col1, col2 = st.columns(2)
        with col1:
            core_loss_multiple(
                st,
                x=[freq / 1e3 for freq in c.streamlit.core_loss_freq],
                y1=[1e-3 * loss(
                    waveform=excitation_short,
                    algorithm='iGSE',
                    material=material,
                    freq=i, flux=flux) for i in c.streamlit.core_loss_freq],
                y2=[1e-3 * loss(
                    waveform=excitation_short,
                    algorithm='ML', material=material,
                    freq=i, flux=flux) for i in c.streamlit.core_loss_freq],
                x0=list([freq / 1e3]),
                y01=list([core_loss_iGSE]),
                y02=list([core_loss_ML]),
                title=f'<b> Core Loss Sweeping Frequency </b>'
                      f'<br> at a fixed Flux Density ({round(flux * 1e3)} mT)',
                x_title='Frequency [kHz]'
            )
        with col2:
            core_loss_multiple(
                st,
                x=[flux * 1e3 for flux in c.streamlit.core_loss_flux],
                y1=[1e-3 * loss(
                    waveform=excitation_short,
                    algorithm='iGSE', material=material,
                    freq=freq, flux=i) for i in c.streamlit.core_loss_flux],
                y2=[1e-3 * loss(
                    waveform=excitation_short,
                    algorithm='ML', material=material,
                    freq=freq, flux=i) for i in c.streamlit.core_loss_flux],
                x0=list([flux * 1e3]),
                y01=list([core_loss_iGSE]),
                y02=list([core_loss_ML]),
                title=f'<b> Core Loss Sweeping Flux Density </b>'
                      f'<br> at a fixed Frequency ({round(freq / 1e3)} kHz)',
                x_title='AC Flux Density [mT]'
            )

    if excitation == "Triangular":
        col1, col2, col3 = st.columns(3)
        with col1:
            core_loss_multiple(
                st,
                x=[freq / 1e3 for freq in c.streamlit.core_loss_freq],
                y1=[1e-3 * loss(
                    waveform=excitation_short,
                    algorithm='iGSE', material=material,
                    freq=i,
                    flux=flux, duty_ratio=duty_p) for i in c.streamlit.core_loss_freq],
                y2=[1e-3 * loss(
                    waveform=excitation_short,
                    algorithm='ML', material=material,
                    freq=i,
                    flux=flux, duty_ratio=duty_p) for i in c.streamlit.core_loss_freq],
                x0=list([freq / 1e3]),
                y01=list([core_loss_iGSE]),
                y02=list([core_loss_ML]),
                title=f'<b> Core Loss Sweeping Frequency </b>'
                      f'<br> at a fixed Flux Density ({round(flux * 1e3)} mT) and Duty Ratio ({duty_p})',
                x_title='Frequency [kHz]',
                x_log=True,
                y_log=True
            )

        with col2:
            core_loss_multiple(
                st,
                x=[flux * 1e3 for flux in c.streamlit.core_loss_flux],
                y1=[1e-3 * loss(
                    waveform=excitation_short,
                    algorithm='iGSE', material=material,
                    freq=freq,
                    flux=i, duty_ratio=duty_p) for i in c.streamlit.core_loss_flux],
                y2=[1e-3 * loss(
                    waveform=excitation_short,
                    algorithm='ML', material=material,
                    freq=freq,
                    flux=i, duty_ratio=duty_p) for i in c.streamlit.core_loss_flux],
                x0=list([flux * 1e3]),
                y01=list([core_loss_iGSE]),
                y02=list([core_loss_ML]),
                title=f'<b> Core Loss Sweeping Flux Density </b>'
                      f'<br> at a fixed Frequency ({freq / 1e3} kHz) and Duty Ratio ({duty_p})',
                x_title='AC Flux Density [mT]',
                x_log=True,
                y_log=True
            )

        with col3:
            core_loss_multiple(
                st,
                x=c.streamlit.core_loss_duty,
                y1=[1e-3 * loss(
                    waveform=excitation_short,
                    algorithm='iGSE', material=material,
                    freq=freq,
                    flux=flux,
                    duty_ratio=i) for i in c.streamlit.core_loss_duty],
                y2=[1e-3 * loss(
                    waveform=excitation_short,
                    algorithm='ML', material=material,
                    freq=freq,
                    flux=flux,
                    duty_ratio=i) for i in c.streamlit.core_loss_duty],
                x0=list([duty_p]),
                y01=list([core_loss_iGSE]),
                y02=list([core_loss_ML]),
                title=f'<b> Core Loss Sweeping Duty Ratio </b>'
                      f'<br> at a fixed Frequency ({round(freq / 1e3)} kHz) and Flux Density ({round(flux * 1e3)} mT)',
                x_title='Duty Ratio',
                x_log=False,
                y_log=True
            )

    if excitation == "Trapezoidal":
        col1, col2 = st.columns(2)
        with col1:
            core_loss_multiple(
                st,
                x=[freq / 1e3 for freq in c.streamlit.core_loss_freq],
                y1=[1e-3 * loss(
                    waveform=excitation_short,
                    algorithm='iGSE', material=material,
                    freq=i,
                    flux=flux,
                    duty_ratios=duty_ratios) for i in c.streamlit.core_loss_freq],
                y2=[1e-3 * loss(
                    waveform=excitation_short,
                    algorithm='ML', material=material,
                    freq=i,
                    flux=flux,
                    duty_ratios=duty_ratios) for i in c.streamlit.core_loss_freq],
                x0=list([freq / 1e3]),
                y01=list([core_loss_iGSE]),
                y02=list([core_loss_ML]),
                title=f'<b> Core Loss Sweeping Frequency </b>'
                      f'<br> at a fixed Flux Density ({round(flux * 1e3)} mT) and the selected Duty Ratios',
                x_title='Frequency [kHz]'
            )

        with col2:
            core_loss_multiple(
                st,
                x=[flux * 1e3 for flux in c.streamlit.core_loss_flux],
                y1=[1e-3 * loss(
                    waveform=excitation_short,
                    algorithm='iGSE', material=material,
                    freq=freq,
                    flux=i,
                    duty_ratios=duty_ratios) for i in c.streamlit.core_loss_flux],
                y2=[1e-3 * loss(
                    waveform=excitation_short,
                    algorithm='ML', material=material,
                    freq=freq,
                    flux=i,
                    duty_ratios=duty_ratios) for i in c.streamlit.core_loss_flux],
                x0=list([flux * 1e3]),
                y01=list([core_loss_iGSE]),
                y02=list([core_loss_ML]),
                title=f'<b> Core Loss Sweeping Flux Density </b>'
                      f'<br> at a fixed Frequency ({round(freq / 1e3)} kHz) and the selected Duty Ratios',
                x_title='AC Flux Density [mT]'
            )

    if excitation == "Simulated":
        st.header("Coming soon!")
        st.header("This Section is still under development.")  # TBD
        core_loss_iGSE = SimulationPLECS(material, algorithm='iGSE') / 1e3
        # core_loss_ML = SimulationPLECS(material, algorithm='ML') / 1e3
        core_loss_ML = 0
        st.subheader(f'Core Loss: {round(core_loss_iGSE,2)} kW/m^3 (by iGSE), {round(core_loss_ML,2)} kW/m^3 (by ML)')
        
    st.markdown("""---""")
