import re
import numpy as np
import streamlit as st

from magnet import config as c
from magnet.constants import material_list, material_manufacturers, material_extra
from magnet.io import load_dataframe
from magnet.plots import waveform_visualization, waveform_visualization_2axes, plot_core_loss, \
    cycle_points_sinusoidal, cycle_points_trapezoidal
from magnet.core import loss


def ui_core_loss_predict(m):
    # Sidebar: input for all calculations
    st.title('MagNet Prediction')
    st.markdown("""---""")
    st.header(f'Input: Case {m}')
    col1, col2 = st.columns(2)

    with col2:
        excitation = st.selectbox(
            f'Excitation:', ('Sinusoidal', 'Triangular', 'Trapezoidal', 'Arbitrary'),
            key=f'excitation {m}')

    with col1:
        material = st.selectbox(
            f'Material:',
            material_list,
            key=f'material {m}')

    df = load_dataframe(material)  # To find the range of the variables

    with col1:
        freq = st.slider(
            "Frequency (kHz)",
            10,
            1000,
            100,
            step=1,
            key=f'freq {m}') * 1e3  # Use kHz for front-end demonstration while Hz for underlying calculation
        if freq < round(min(df['Frequency'])):
            st.warning(f"The model has not been trained for frequencies below {round(min(df['Frequency']) * 1e-3)} kHz")
        if freq > round(max(df['Frequency'])):
            st.warning(f"The model has not been trained for frequencies above {round(max(df['Frequency']) * 1e-3)} kHz")

        if excitation == "Arbitrary":
            flux_string_mT = st.text_input(
                f'Waveform Pattern - AC Flux Density (mT)',
                [0, 150, 0, -20, -27, -20, 0],
                key=f'flux {m}')
            duty_string_percentage = st.text_input(
                f'Waveform Pattern - Duty Cycle (%)',
                [0, 50, 60, 65, 70, 75, 80],
                key=f'duty {m}')
        else:
            flux = st.slider(  # Use mT for front-end demonstration while T for underlying calculation
                f'AC Flux Density (mT)',
                1,
                500,
                50,
                step=1,
                key=f'flux {m}',
                help=f'Amplitude of the AC signal, not peak to peak') / 1e3
            if flux < min(df['Flux_Density']):
                st.warning(f"The model has not been trained for peak flux densities  below {round(min(df['Flux_Density']) * 1e3)} mT")
            if flux > max(df['Flux_Density']):
                st.warning(f"The model has not been trained for peak flux densities  above {round(max(df['Flux_Density']) * 1e3)} mT")

    duty_step = 0.01
    with col1:
        if excitation == "Triangular":
            duty_p = st.slider(
                f'Duty Cycle',
                duty_step,
                1-duty_step,
                0.5,
                step=duty_step,
                key=f'duty {m}')
            duty_n = 1 - duty_p
            duty_0 = 0
        if excitation == "Trapezoidal":
            duty_p = st.slider(
                f'Duty Cycle (D1)',
                duty_step,
                1-duty_step,
                0.5,
                step=duty_step,
                key=f'dutyP {m}',
                help=f'Rising part with the highest slope')
            duty_n = st.slider(
                f'Duty Cycle (D3)',
                1-duty_step,
                1 - duty_p,
                max(round((1 - duty_p) / 2, 2), duty_step),
                step=duty_step,
                key=f'dutyN {m}',
                help=f'Falling part with the highest slope')
            duty_0 = round((1-duty_p-duty_n)/2, 2)
            st.write(f'Duty cycle D2=D4=(1-D1-D3)/2)={duty_0}'),

            if duty_p < min(df['Duty_P']) or duty_n < min(df['Duty_N']):
                st.warning(f"The model has not been trained for duty cycles below {round(min(df['Duty_P']), 2)}")
            if duty_p > max(df['Duty_P']) or duty_n > max(df['Duty_N']):
                st.warning(f"The model has not been trained for duty cycles above {round(max(df['Duty_P']), 2)}")

        mu_relative = material_extra[material][0]
        if excitation == "Arbitrary":
            flux_bias = 0.010  # TODO implement the equation for flux bias based on the waveform
            bias = flux_bias / (mu_relative * c.streamlit.mu_0)
            st.write(f'Hdc={round(bias, 2)} A/m, '
                     f'approximated based on the average B waveform ({flux_bias * 1e3} mT) '
                     f'using mur={mu_relative}')
        else:
            bias_b_max = 0.3
            max_bias = (bias_b_max - flux) / (mu_relative * c.streamlit.mu_0)
            bias = st.slider(
                f'DC Bias (A/m)',
                0,
                round(max_bias),
                0,
                step=5,
                key=f'bias {m}',
                help=f'Hdc provided as Bdc is not available. '
                     f'Bdc approximated with B=mu*H for mur={mu_relative} for the plots')
            flux_bias = bias * mu_relative * c.streamlit.mu_0
            if bias < 0:
                st.warning(f"The model has not been trained for bias below 0 A/m")
            if bias > round(max(df['DC_Bias'])):
                st.warning(f"The model has not been trained for bias above {round(max(df['DC_Bias']))} A/m")

        temp = st.slider(
            f'Temperature (C)',
            0,
            150,
            25,
            step=5,
            key=f'temp {m}')
        if temp < round(min(df['Temperature'])):
            st.warning(f"The model has not been trained for temperature below {round(min(df['Temperature']))} C")
        if temp > round(max(df['Temperature'])):
            st.warning(f"The model has not been trained for temperature above {round(max(df['Temperature']))} C")

    # Variables that are function of the sliders, different type depending on the excitation

    flag_inputs_ok = 1  # To avoid errors when inputs are not ok
    if excitation == "Sinusoidal":
        duty = None
    if excitation == "Triangular":
        duty = duty_p
    if excitation == "Trapezoidal":
        duty = [duty_p, duty_n, duty_0]

    if excitation == "Arbitrary":
        duty = [float(i) / 100 for i in re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", duty_string_percentage)]
        flux_read = [float(i) * 1e-3 for i in re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", flux_string_mT)]
        duty.append(1)
        flux_read.append(flux_read[0])
        flux = np.multiply(flux_read, 1)  # For the calculations, the average is removed

        if len(duty) != len(flux):
            flag_inputs_ok = 0
        if max(duty) > 1:
            flag_inputs_ok = 0
        if min(duty) < 0:
            flag_inputs_ok = 0
        for i in range(0, len(duty)-1):
            if duty[i] >= duty[i+1]:
                flag_inputs_ok = 0

    # Core loss based on ML
    core_loss_ML = 0.0 if flag_inputs_ok == 0 else loss(
waveform=excitation, material=material, freq=freq, flux=flux, duty=duty)

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

    # Results summary and waveform
    # DUT and operation point and core losses

    st.header(f'Output: Case {m}: {round(core_loss_ML / 1e3 ,2)} kW/m^3')
    if excitation == "Sinusoidal":
        st.write(f'{material_manufacturers[material]} - {material}, {excitation} excitation, '
                 f'f={round(freq / 1e3)} kHz, Bac={round(flux * 1e3)} mT, Bias={bias} A/m, '
                 f'T={round(temp)} C')
    if excitation == "Triangular":
        st.write(f'{material_manufacturers[material]} - {material}, {excitation} excitation, '
                 f'f={round(freq / 1e3)} kHz, Bac={round(flux * 1e3)} mT, Bias={bias} A/m, '
                 f'D={round(duty_p, 2)}, '
                 f'T={round(temp)} C')
    if excitation == "Trapezoidal":
        st.write(f'{material_manufacturers[material]} - {material}, {excitation} excitation, '
                 f'f={round(freq / 1e3)} kHz, Bac={round(flux * 1e3)} mT, Bias={bias} A/m, '
                 f'D1={round(duty_p, 2)}, D2={round(duty_0, 2)}, D3={round(duty_n, 2)}, D4={round(duty_0, 2)}, '
                 f'T={round(temp)} C')
    if excitation == "Arbitrary":
        st.write(f'{material_manufacturers[material]} - {material}, {excitation} excitation, '
                 f'f={round(freq / 1e3)} kHz, Bac={round(1 * 1e3)} mT, Bias={round(bias, 2)} A/m, '  # TODO mT PLACEHOLDER
                 f'T={round(temp)} C')
    st.write("")
    if flag_inputs_ok == 1:  # To avoid problems with the inputs for Arbitrary waveforms
        flag_minor_loop = 0
        if excitation == "Arbitrary":
            if max(abs(flux)) > 0.3:
                st.write('Warning: Peak flux density above 300 mT,')
                st.write('above targeted test values; results might be inaccurate.')
            flag_dbdt_high = 0
            for i in range(0, len(duty)-1):
                if abs(flux[i + 1] - flux[i]) * freq / (duty[i + 1] - duty[i]) > 3e6:
                    flag_dbdt_high = 1
            if flag_dbdt_high == 1:
                st.write('Warning: dB/dt above 3 mT/ns,')
                st.write('above targeted test values; results might be inaccurate.')

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
        if flag_minor_loop == 1:
            st.write('Minor loops present, NN not trained for this waveform')
    else:
        if len(duty) != len(flux):
            st.write('The Flux and Duty vectors should have the same length, please fix it to proceed.')
        if max(duty) > 1:
            st.write('Duty cycle should be below 100%, please fix it to proceed.')
        if min(duty) < 0:
            st.write('Please provide only positive values for the Duty cycle, fix it to proceed.')
        flag_duty_wrong = 0
        for i in range(0, len(duty)-1):
            if duty[i] >= duty[i + 1]:
                flag_duty_wrong = 1
        if flag_duty_wrong == 1:
            st.write('Only increasing duty cycles allowed, fix it to proceed.')
            st.write('Please remove the 100% duty cycle value, the flux is assigned to the value at Duty=0%.')

    if excitation == "Arbitrary":
        st.write('Please bear in mind that the neural network has been trained using '
                 'trapezoidal, triangular and sinusoidal data. '
                 'The accuracy for waveforms very different from those used for training cannot be guaranteed.')

        # TODO implementation with DC bias

    else:

        # Plots for different sweeps
        col1, col2 = st.columns(2)

        if excitation == "Sinusoidal":
            subtitle_plot = ''
        if excitation == "Triangular":
            subtitle_plot = f' and duty cycle ({duty})'
        if excitation == "Trapezoidal":
            subtitle_plot = f' and the selected duty cycles'

        # vs frequency
        with col1:
            plot_core_loss(
                st,
                x=[freq / 1e3 for freq in c.streamlit.core_loss_freq],
                y=[1e-3 * loss(waveform=excitation, material=material, freq=i, flux=flux, duty=duty)
                    for i in c.streamlit.core_loss_freq],
                y_upper=[1e-3 * loss(waveform=excitation, material=material, freq=i, flux=2*flux, duty=duty)
                         for i in c.streamlit.core_loss_freq],
                y_lower=[1e-3 * loss(waveform=excitation, material=material, freq=i, flux=flux/2, duty=duty)
                         for i in c.streamlit.core_loss_freq],
                x0=list([freq / 1e3]),
                y0=list([1e-3 * core_loss_ML]),
                legend=f'{round(flux * 1e3)} mT',
                legend_upper=f'{round(2 * flux * 1e3)} mT',
                legend_lower=f'{round(flux / 2 * 1e3)} mT',
                title=f'<b> Core Loss Sweeping Frequency </b>'
                      f'<br> at a Few Flux Densities {subtitle_plot}',
                x_title='Frequency [kHz]'
            )
        # vs flux density
        with col2:
            plot_core_loss(
                st,
                x=[flux * 1e3 for flux in c.streamlit.core_loss_flux],
                y=[1e-3 * loss(waveform=excitation, material=material, freq=freq, flux=i, duty=duty)
                    for i in c.streamlit.core_loss_flux],
                y_upper=[1e-3 * loss(waveform=excitation, material=material, freq=2 * freq, flux=i, duty=duty)
                   for i in c.streamlit.core_loss_flux],
                y_lower=[1e-3 * loss(waveform=excitation, material=material, freq=freq / 2, flux=i, duty=duty)
                   for i in c.streamlit.core_loss_flux],
                x0=list([flux * 1e3]),
                y0=list([1e-3 * core_loss_ML]),
                legend=f'{round(freq * 1e-3)} kHz',
                legend_upper=f'{round(2 * freq * 1e-3)} kHz',
                legend_lower=f'{round(freq / 2 * 1e-3)} kHz',
                title=f'<b> Core Loss Sweeping Flux Density </b>'
                      f'<br> at a fixed Frequency {subtitle_plot}',
                x_title='AC Flux Density [mT]'
            )

            if excitation == "Triangular":
                # vs duty at a few flux densities
                with col1:
                    plot_core_loss(
                        st,
                        x=c.streamlit.core_loss_duty,
                        y=[1e-3 * loss(waveform=excitation, material=material, freq=freq, flux=flux, duty=i)
                            for i in c.streamlit.core_loss_duty],
                        y_upper=[1e-3 * loss(waveform=excitation, material=material, freq=freq, flux=2 * flux, duty=i)
                           for i in c.streamlit.core_loss_duty],
                        y_lower=[1e-3 * loss(waveform=excitation, material=material, freq=freq, flux=flux / 2, duty=i)
                           for i in c.streamlit.core_loss_duty],
                        x0=list([duty_p]),
                        y0=list([1e-3 * core_loss_ML]),
                        legend=f'{round(freq * 1e-3)} kHz, {round(flux * 1e3)} mT',
                        legend_upper=f'{round(freq * 1e-3)} kHz, {round(2 * flux * 1e3)} mT',
                        legend_lower=f'{round(freq* 1e-3)} kHz, {round(flux / 2 * 1e3)} mT',
                        title=f'<b> Core Loss Sweeping Duty Ratio </b>'
                              f'<br> at a fixed Frequency and Flux Density',
                        x_title='Duty Ratio',
                        x_log=False,
                        y_log=True
                    )
                with col2:
                    # vs duty at a few frequencies
                    plot_core_loss(
                        st,
                        x=c.streamlit.core_loss_duty,
                        y=[1e-3 *
                           loss(waveform=excitation, material=material, freq=freq, flux=flux, duty=i)
                           for i in c.streamlit.core_loss_duty],
                        y_upper=[1e-3 * loss(waveform=excitation, material=material, freq=2 * freq, flux=flux, duty=i)
                                 for i in c.streamlit.core_loss_duty],
                        y_lower=[1e-3 * loss(waveform=excitation, material=material, freq=freq / 2, flux=flux, duty=i)
                                 for i in c.streamlit.core_loss_duty],
                        x0=list([duty_p]),
                        y0=list([1e-3 * core_loss_ML]),
                        legend=f'{round(freq * 1e-3)} kHz, {round(flux * 1e3)} mT',
                        legend_upper=f'{round(2 *freq * 1e-3)} kHz, {round(flux * 1e3)} mT',
                        legend_lower=f'{round(freq /2 * 1e-3)} kHz, {round(flux * 1e3)} mT',
                        title=f'<b> Core Loss Sweeping Duty Ratio </b>'
                              f'<br> at a fixed Frequency and Flux Density',
                        x_title='Duty Ratio',
                        x_log=False,
                        y_log=True
                    )
            # vs dc bias at a few flux densities
            with col1:
                plot_core_loss(
                    st,
                    x=c.streamlit.core_loss_bias,
                    y=[1e-3 * loss(waveform=excitation, material=material, freq=freq, flux=flux, duty=duty)
                       for i in c.streamlit.core_loss_bias],
                    y_upper=[
                        1e-3 * loss(waveform=excitation, material=material, freq=freq, flux=2 * flux, duty=duty)
                        for i in c.streamlit.core_loss_bias],
                    y_lower=[
                        1e-3 * loss(waveform=excitation, material=material, freq=freq, flux=flux / 2, duty=duty)
                        for i in c.streamlit.core_loss_bias],
                    x0=list([bias]),
                    y0=list([1e-3 * core_loss_ML]),
                    legend=f'{round(freq * 1e-3)} kHz, {round(flux * 1e3)} mT',
                    legend_upper=f'{round(freq * 1e-3)} kHz, {round(2 * flux * 1e3)} mT',
                    legend_lower=f'{round(freq * 1e-3)} kHz, {round(flux / 2 * 1e3)} mT',
                    title=f'<b> Core Loss Sweeping DC Bias </b>'
                          f'<br> at a fixed Frequency and Flux Density',
                    x_title='DC Bias [A/m]',
                    x_log=False,
                    y_log=True
                )

                    # TODO add plots at different temperatures and DC bias

    st.markdown("""---""")
