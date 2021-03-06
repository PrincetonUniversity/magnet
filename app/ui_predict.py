import re
import numpy as np
import streamlit as st

from magnet import config as c
from magnet.constants import material_names, excitations_predict, material_manufacturers
from magnet.plots import waveform_visualization, core_loss_multiple, waveform_visualization_2axes, \
    cycle_points_sinusoidal, cycle_points_trapezoidal
from magnet.core import loss
from magnet.io import loss_interpolated


def ui_core_loss_predict(m):
    # Sidebar: input for all calculations
    st.sidebar.header(f'Information: Case {m}')
    excitation = st.sidebar.selectbox(
        f'Excitation:', excitations_predict,
        key=f'excitation {m}')  # TBD
    material = st.sidebar.selectbox(
        f'Material:',
        material_names,
        key=f'material {m}')
    freq = st.sidebar.slider(
        "Frequency (kHz)",
        round(c.streamlit.freq_min / 1e3),
        round(c.streamlit.freq_max / 1e3),
        round(c.streamlit.freq_max / 2 / 1e3),
        step=round(c.streamlit.freq_step / 1e3),
        key=f'freq {m}') * 1e3  # Use kHz for front-end demonstration while Hz for underlying calculation
    if excitation == "Arbitrary":
        flux_string = st.sidebar.text_input(
            f'Waveform Pattern - AC Flux Density (mT)',
            [0, 150, 0, -20, -27, -20, 0],
            key=f'flux {m}')
        duty_string = st.sidebar.text_input(
            f'Waveform Pattern - Duty Cycle (%)',
            [0, 50, 60, 65, 70, 75, 80],
            key=f'duty {m}')
    else:
        flux = st.sidebar.slider(
            f'AC Flux Density (mT)',
            round(c.streamlit.flux_min * 1e3),
            round(c.streamlit.flux_max * 1e3),
            round(c.streamlit.flux_max / 2 * 1e3),
            step=round(c.streamlit.flux_step * 1e3),
            key=f'flux {m}',
            help=f'Amplitude of the AC signal, not peak to peak') / 1e3  # Use mT for front-end demonstration while T for underlying calculation
    flux_bias = st.sidebar.slider(
        f'DC Flux Density (mT) coming soon!',
        round(-c.streamlit.flux_max * 1e3),
        round(c.streamlit.flux_max * 1e3),
        0,
        step=round(1e9),
        key=f'bias {m}',
        help=f'Fixed at 0 mT for now') / 1e3
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

    st.sidebar.markdown("""---""")

    # Variables that are function of the sliders, different type depending on the excitation

    flag_inputs_ok = 1  # To avoid errors when inputs are not ok
    if excitation == "Sinusoidal":
        duty = None
    if excitation == "Triangular":
        duty = duty_p
    if excitation == "Trapezoidal":
        duty = [duty_p, duty_n, duty_0]
    if excitation == "Arbitrary":
        duty = [float(i) / 100 for i in re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", duty_string)]
        flux_read = [float(i) for i in re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", flux_string)]
        duty.append(1)
        flux_read.append(flux_read[0])
        flux = np.multiply(np.add(flux_read, flux_bias), 1e-3)  # For the calculations, the average is removed
        if len(duty) != len(flux):
            flag_inputs_ok = 0
        if max(duty) > 1:
            flag_inputs_ok = 0
        if min(duty) < 0:
            flag_inputs_ok = 0
        for i in range(0, len(duty)-1):
            if duty[i] >= duty[i+1]:
                flag_inputs_ok = 0

    # Core loss based on iGSE, ML and interpolation of the data (DI = Datasheet SI = Sinusoidal Interpolation)

    core_loss_iGSE = 0.0 if flag_inputs_ok == 0 else loss(
        waveform=excitation, algorithm='iGSE', material=material, freq=freq, flux=flux, duty=duty) / 1e3
    core_loss_ML = 0.0 if flag_inputs_ok == 0 else loss(
        waveform=excitation, algorithm='ML', material=material, freq=freq, flux=flux, duty=duty) / 1e3
    core_loss_DI = 0.0 if excitation != 'Sinusoidal' else loss_interpolated(  # Comparison only available for Sinusoidal
        waveform=excitation, algorithm='DI', material=material, freq=freq, flux=flux) / 1e3
    core_loss_SI = 0.0 if excitation != 'Sinusoidal' else loss_interpolated(  # Comparison only available for Sinusoidal
        waveform=excitation, algorithm='SI', material=material, freq=freq, flux=flux) / 1e3

    # Results summary and waveform
    col1, col2 = st.columns(2)
    # DUT and operation point and core losses
    with col1:
        st.title(f'Core Loss Analysis: Case {m}')
        st.subheader(f'{material_manufacturers[material]} - {material}, '
                     f'{excitation} excitation')
        if excitation == "Arbitrary":
            st.subheader(f'f={round(freq / 1e3)} kHz')
        else:  # if Sinusoidal Trapezoidal or Triangular
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
        st.write("")
        if flag_inputs_ok == 1:  # To avoid problems with the inputs for Arbitrary waveforms
            flag_minor_loop = 0
            if excitation == "Arbitrary":
                if max(abs(flux)) > 0.3:
                    st.subheader('Warning: Peak flux density above 300 mT,')
                    st.subheader('above targeted test values; results might be inaccurate.')
                flag_dbdt_high = 0
                for i in range(0, len(duty)-1):
                    if abs(flux[i + 1] - flux[i]) * freq / (duty[i + 1] - duty[i]) > 3e6:
                        flag_dbdt_high = 1
                if flag_dbdt_high == 1:
                    st.subheader('Warning: dB/dt above 3 mT/ns,')
                    st.subheader('above targeted test values; results might be inaccurate.')

                if np.argmin(flux) < np.argmax(flux):  # min then max
                    for i in range(np.argmin(flux), np.argmax(flux)):
                        if flux[i + 1] < flux[i]:
                            flag_minor_loop = 1
                    for i in range(np.argmax(flux), len(flux)-1):
                        if flux[i + 1] > flux[i]:
                            flag_minor_loop = 1
                    for i in range(0, np.argmin(flux)):
                        if flux[i + 1] > flux[i]:
                            flag_minor_loop = 1
                else:  # max then min
                    for i in range(0, np.argmax(flux)):
                        if flux[i + 1] < flux[i]:
                            flag_minor_loop = 1
                    for i in range(np.argmin(flux), len(flux)-1):
                        if flux[i + 1] < flux[i]:
                            flag_minor_loop = 1
                    for i in range(np.argmax(flux), np.argmin(flux)):
                        if flux[i + 1] > flux[i]:
                            flag_minor_loop = 1
            st.subheader(f'Core Loss:')
            if flag_minor_loop == 1:
                st.subheader('Minor loops present, iGSE not defined for this case.')
            else:
                st.subheader(f'{round(core_loss_iGSE,2)} kW/m^3 - iGSE')
            st.subheader(f'{round(core_loss_ML,2)} kW/m^3 - Machine Learning (ML)')

        else:
            if len(duty) != len(flux):
                st.subheader('The Flux and Duty vectors should have the same length, please fix it to proceed.')
            if max(duty) > 1:
                st.subheader('Duty cycle should be below 100%, please fix it to proceed.')
            if min(duty) < 0:
                st.subheader('Please provide only positive values for the Duty cycle, fix it to proceed.')
            flag_duty_wrong = 0
            for i in range(0, len(duty)-1):
                if duty[i] >= duty[i + 1]:
                    flag_duty_wrong = 1
            if flag_duty_wrong == 1:
                st.subheader('Only increasing duty cycles allowed, fix it to proceed.')
                st.write('Please remove the 100% duty cycle value, the flux is assigned to the value at Duty=0%.')

        if excitation == "Sinusoidal":
            if core_loss_SI != 0.0:
                st.subheader(f'{round(core_loss_SI, 2)} kW/m^3 - Interpolated from Measurements')
            else:
                st.write('No measured data available for interpolation')
            if core_loss_DI != 0.0:
                st.subheader(f'{round(core_loss_DI, 2)} kW/m^3 - Interpolated from Datasheet')
            else:
                st.write('No datasheet information available for interpolation')
        if excitation == "Arbitrary":
            st.write('Please bear in mind that the neural network has been trained using '
                     'trapezoidal, triangular and sinusoidal data. '
                     'The accuracy for waveforms very different from those used for training cannot be guaranteed. '
                     'For iGSE, the waveform is considered as containing only major loops. '
                     'DC bias is not yet implemented, the average flux density is removed before calculation. ')
    # Representation of the waveform
    with col2:
        if excitation == "Arbitrary":
            waveform_visualization(
                st,
                x=duty,
                y=np.multiply(flux, 1e3))
        else:  # if Sinusoidal, Triangular and Trapezoidal, Arbitrary disabled
            if excitation == 'Sinusoidal':
                [cycle_list, flux_list, volt_list] = cycle_points_sinusoidal(c.streamlit.n_points_plot)
            if excitation in ['Triangular', 'Trapezoidal']:
                [cycle_list, flux_list, volt_list] = cycle_points_trapezoidal(duty_p, duty_n, duty_0)
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

    # Plots for different sweeps
    if excitation == "Sinusoidal":
        col1, col2 = st.columns(2)
        with col1:
            core_loss_multiple(
                st,
                x=[freq / 1e3 for freq in c.streamlit.core_loss_freq],
                y1=[1e-3 *
                    loss(waveform=excitation, algorithm='iGSE', material=material, freq=i, flux=flux)
                    for i in c.streamlit.core_loss_freq],
                y2=[1e-3 *
                    loss(waveform=excitation, algorithm='ML', material=material, freq=i, flux=flux)
                    for i in c.streamlit.core_loss_freq],
                y3=[1e-3 *
                    loss_interpolated(waveform=excitation, algorithm='DI', material=material, freq=i, flux=flux)
                    for i in c.streamlit.core_loss_freq],
                y4=[1e-3 *
                    loss_interpolated(waveform=excitation, algorithm='SI', material=material, freq=i, flux=flux)
                    for i in c.streamlit.core_loss_freq],
                x0=list([freq / 1e3]),
                y01=list([core_loss_iGSE]),
                y02=list([core_loss_ML]),
                y03=list([core_loss_DI]),
                y04=list([core_loss_SI]),
                title=f'<b> Core Loss Sweeping Frequency </b>'
                      f'<br> at a fixed Flux Density ({round(flux * 1e3)} mT)',
                x_title='Frequency [kHz]'
            )
        with col2:
            core_loss_multiple(
                st,
                x=[flux * 1e3 for flux in c.streamlit.core_loss_flux],
                y1=[1e-3 *
                    loss(waveform=excitation, algorithm='iGSE', material=material, freq=freq, flux=i)
                    for i in c.streamlit.core_loss_flux],
                y2=[1e-3 *
                    loss(waveform=excitation, algorithm='ML', material=material, freq=freq, flux=i)
                    for i in c.streamlit.core_loss_flux],
                y3=[1e-3 *
                    loss_interpolated(waveform=excitation, algorithm='DI', material=material, freq=freq, flux=i)
                    for i in c.streamlit.core_loss_flux],
                y4=[1e-3 *
                    loss_interpolated(waveform=excitation, algorithm='SI', material=material, freq=freq, flux=i)
                    for i in c.streamlit.core_loss_flux],
                x0=list([flux * 1e3]),
                y01=list([core_loss_iGSE]),
                y02=list([core_loss_ML]),
                y03=list([core_loss_DI]),
                y04=list([core_loss_SI]),
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
                y1=[1e-3 *
                    loss(waveform=excitation, algorithm='iGSE', material=material, freq=i, flux=flux, duty=duty)
                    for i in c.streamlit.core_loss_freq],
                y2=[1e-3 *
                    loss(
                    waveform=excitation, algorithm='ML', material=material, freq=i, flux=flux, duty=duty)
                    for i in c.streamlit.core_loss_freq],
                x0=list([freq / 1e3]),
                y01=list([core_loss_iGSE]),
                y02=list([core_loss_ML]),
                title=f'<b> Core Loss Sweeping Frequency </b>'
                      f'<br> at a fixed Flux Density ({round(flux * 1e3)} mT) and Duty Ratio ({duty})',
                x_title='Frequency [kHz]',
                x_log=True,
                y_log=True
            )
        with col2:
            core_loss_multiple(
                st,
                x=[flux * 1e3 for flux in c.streamlit.core_loss_flux],
                y1=[1e-3 *
                    loss(waveform=excitation, algorithm='iGSE', material=material, freq=freq, flux=i, duty=duty)
                    for i in c.streamlit.core_loss_flux],
                y2=[1e-3 *
                    loss(waveform=excitation, algorithm='ML', material=material, freq=freq, flux=i, duty=duty)
                    for i in c.streamlit.core_loss_flux],
                x0=list([flux * 1e3]),
                y01=list([core_loss_iGSE]),
                y02=list([core_loss_ML]),
                title=f'<b> Core Loss Sweeping Flux Density </b>'
                      f'<br> at a fixed Frequency ({freq / 1e3} kHz) and Duty Ratio ({duty})',
                x_title='AC Flux Density [mT]',
                x_log=True,
                y_log=True
            )
        with col3:
            core_loss_multiple(
                st,
                x=c.streamlit.core_loss_duty,
                y1=[1e-3 *
                    loss(waveform=excitation, algorithm='iGSE', material=material, freq=freq, flux=flux, duty=i)
                    for i in c.streamlit.core_loss_duty],
                y2=[1e-3 *
                    loss(waveform=excitation, algorithm='ML', material=material, freq=freq, flux=flux, duty=i)
                    for i in c.streamlit.core_loss_duty],
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
                y1=[1e-3 *
                    loss(waveform=excitation, algorithm='iGSE', material=material, freq=i, flux=flux, duty=duty)
                    for i in c.streamlit.core_loss_freq],
                y2=[1e-3 *
                    loss(waveform=excitation, algorithm='ML', material=material, freq=i, flux=flux, duty=duty)
                    for i in c.streamlit.core_loss_freq],
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
                y1=[1e-3 *
                    loss(waveform=excitation, algorithm='iGSE', material=material, freq=freq, flux=i, duty=duty)
                    for i in c.streamlit.core_loss_flux],
                y2=[1e-3 *
                    loss(waveform=excitation, algorithm='ML', material=material, freq=freq, flux=i, duty=duty)
                    for i in c.streamlit.core_loss_flux],
                x0=list([flux * 1e3]),
                y01=list([core_loss_iGSE]),
                y02=list([core_loss_ML]),
                title=f'<b> Core Loss Sweeping Flux Density </b>'
                      f'<br> at a fixed Frequency ({round(freq / 1e3)} kHz) and the selected Duty Ratios',
                x_title='AC Flux Density [mT]'
            )

    st.markdown("""---""")
