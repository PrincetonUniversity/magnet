import re
import numpy as np
import streamlit as st
import pandas as pd

from magnet import config as c
from magnet.constants import material_list, material_manufacturers, material_extra
from magnet.io import load_dataframe
from magnet.plots import waveform_visualization, waveform_visualization_2axes, plot_core_loss, \
    cycle_points_sinusoidal, cycle_points_trapezoidal
from magnet.core import core_loss_default, core_loss_arbitrary
from magnet.constants import core_loss_range


def convert_df(df):
    return df.to_csv().encode('utf-8')


def ui_core_loss_predict(m):
    st.title('MagNet Smartsheet for Interactive Design')
    st.subheader('"Are you tired of multi-dimentional interpolation when designing power magnetics?" - Try MagNet smartsheet and help us to make it smarter!')
    st.caption('Traditional magnetics datasheet only provide limited information. Multi-dimensional interporation is always a headache. We created MagNet smartsheet to rapidly generate the needed information for what you need in the design process. No more graph reading, no more multi-dimensional interporations.')
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
            index=9,
            key=f'material {m}')

    mu_relative = material_extra[material][0]
    df = load_dataframe(material)  # To find the range of the variables

    with col1:
        freq = st.slider(
            "Frequency (kHz)",
            10,
            1000,
            200,
            step=1,
            key=f'freq {m}') * 1e3  # Use kHz for front-end demonstration while Hz for underlying calculation
        if freq < min(df['Frequency']):
            st.warning(f"The model has not been trained for frequencies below {round(min(df['Frequency']) * 1e-3)} kHz. MagNet AI is doing the extrapolation.")
        if freq > max(df['Frequency']):
            st.warning(f"The model has not been trained for frequencies above {round(max(df['Frequency']) * 1e-3)} kHz. MagNet AI is doing the extrapolation.")

        if excitation == "Arbitrary":
            flux_string_militesla = st.text_input(
                f'Waveform Pattern - AC Flux Density (mT)',
                [0, 150, 0, -20, -27, -20, 0],
                key=f'flux {m}')
            duty_string_percentage = st.text_input(
                f'Waveform Pattern - Duty Cycle (%)',
                [0, 50, 60, 65, 70, 75, 80],
                key=f'duty {m}')
            duty_read = [float(i) / 100 for i in re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", duty_string_percentage)]
            flux_read = [float(i) * 1e-3 for i in re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", flux_string_militesla)]
            duty_read.append(1)
            flux_read.append(flux_read[0])
            flux_vector = np.array(flux_read)
            duty_vector = np.array(duty_read)  # For the calculations, the average is removed

            flag_inputs_ok = 1  # To avoid errors when inputs are not ok
            if len(duty_vector) != len(flux_vector):
                flag_inputs_ok = 0
                st.error('The Flux and Duty vectors should have the same length, please fix it to proceed.')
            if max(duty_vector) > 1:
                flag_inputs_ok = 0
                st.error('Duty cycle should be below 100%, please fix it to proceed.')
            if min(duty_vector) < 0:
                flag_inputs_ok = 0
                st.error('Please provide only positive values for the Duty cycle, fix it to proceed.')
            flag_duty_wrong = 0
            for i in range(0, len(duty_vector) - 1):
                if duty_vector[i] >= duty_vector[i + 1]:
                    flag_duty_wrong = 1
            if flag_duty_wrong == 1:
                flag_inputs_ok = 0
                st.error('Only increasing duty cycles allowed, fix it to proceed. \n'
                         'Please remove the 100% duty cycle value, the flux is assigned to the value at Duty=0%.')
            if flag_inputs_ok == 0:
                st.subheader('Waveform set to 0, fix the above errors to proceed.')
                duty_vector = [0, 1]
                flux_vector = [0, 0]

            flag_minor_loop = 0
            if np.argmin(flux_vector) < np.argmax(flux_vector):  # min then max
                for i in range(np.argmin(flux_vector), np.argmax(flux_vector)):
                    if flux_vector[i + 1] < flux_vector[i]:
                        flag_minor_loop = 1
                for i in range(np.argmax(flux_vector), len(flux_vector) - 1):
                    if flux_vector[i + 1] > flux_vector[i]:
                        flag_minor_loop = 1
                for i in range(0, np.argmin(flux_vector)):
                    if flux_vector[i + 1] > flux_vector[i]:
                        flag_minor_loop = 1
            else:  # max then min
                for i in range(0, np.argmax(flux_vector)):
                    if flux_vector[i + 1] < flux_vector[i]:
                        flag_minor_loop = 1
                for i in range(np.argmin(flux_vector), len(flux_vector) - 1):
                    if flux_vector[i + 1] < flux_vector[i]:
                        flag_minor_loop = 1
                for i in range(np.argmax(flux_vector), np.argmin(flux_vector)):
                    if flux_vector[i + 1] > flux_vector[i]:
                        flag_minor_loop = 1
            if flag_minor_loop == 1:
                st.warning('Minor loops present, NN not trained for this waveform')

            flux_bias = np.average(np.interp(np.linspace(0, 1, c.streamlit.n_nn), np.array(duty_vector), np.array(flux_vector)))
            bias = flux_bias / (mu_relative * c.streamlit.mu_0)
            st.write(f'Hdc={round(bias, 2)} A/m, '
                     f'approximated based on the average B waveform ({round(flux_bias * 1e3)} mT) '
                     f'using mur={round(mu_relative)}')

            flux_vector = flux_vector - flux_bias  # Remove the average B
            flux = (max(flux_vector)-min(flux_vector))/2

        if excitation != "Arbitrary":  # For sinusoidal, triangular or trapezoidal waveforms
            flux = st.slider(  # Use mT for front-end demonstration while T for underlying calculation
                f'AC Flux Density (mT)',
                1,
                500,
                50,
                step=1,
                key=f'flux {m}',
                help=f'Amplitude of the AC signal, not peak to peak') / 1e3

        if flux < min(df['Flux_Density']):
            st.warning(f"The model has not been trained for peak flux densities below {round(min(df['Flux_Density']) * 1e3)} mT. MagNet AI is doing the extrapolation.")
        if flux > max(df['Flux_Density']):
            st.warning(f"The model has not been trained for peak flux densities above {round(max(df['Flux_Density']) * 1e3)} mT. MagNet AI is doing the extrapolation.")

        if excitation != "Arbitrary":  # For sinusoidal, triangular or trapezoidal waveforms

            duty_step = 0.01

            if excitation == "Sinusoidal":
                duty = None
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
                duty = duty_p
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
                    duty_step,
                    1 - duty_p,
                    max(round((1 - duty_p) / 2, 2), duty_step),
                    step=duty_step,
                    key=f'dutyN {m}',
                    help=f'Falling part with the highest slope')
                duty_0 = round((1-duty_p-duty_n)/2, 2)
                st.write(f'Duty cycle D2=D4=(1-D1-D3)/2)={duty_0}'),
                duty = [duty_p, duty_n, duty_0]
            if excitation in ["Triangular", "Trapezoidal"]:
                if duty_p < min(df['Duty_P']) or duty_n < min(df['Duty_N']):
                    st.warning(f"The model has not been trained for duty cycles below {round(min(df['Duty_P']), 2)}. MagNet AI is doing the extrapolation.")
                if duty_p > max(df['Duty_P']) or duty_n > max(df['Duty_N']):
                    st.warning(f"The model has not been trained for duty cycles above {round(max(df['Duty_P']), 2)}. MagNet AI is doing the extrapolation.")


        # TODO add limitations to max B and dB/dt warning
            # if excitation == "Arbitrary":
            #     if max(abs(flux)) > 0.3:
            #         st.write('Warning: Peak flux density above 300 mT,')
            #         st.write('above targeted test values; results might be inaccurate.')
            #     flag_dbdt_high = 0
            #     for i in range(0, len(duty) - 1):
            #         if abs(flux[i + 1] - flux[i]) * freq / (duty[i + 1] - duty[i]) > 3e6:
            #             flag_dbdt_high = 1
            #     if flag_dbdt_high == 1:
            #         st.write('Warning: dB/dt above 3 mT/ns,')
            #         st.write('above targeted test values; results might be inaccurate.')

        if excitation != "Arbitrary":  # For sinusoidal, triangular or trapezoidal waveforms
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
            st.warning(f"The model has not been trained for bias below 0 A/m. MagNet AI is doing the extrapolation.")
        if bias > max(df['DC_Bias']):
            st.warning(f"The model has not been trained for bias above {round(max(df['DC_Bias']))} A/m. MagNet AI is doing the extrapolation.")

        temp = st.slider(
            f'Temperature (C)',
            0,
            120,
            25,
            step=5,
            key=f'temp {m}')
        if temp < min(df['Temperature']):
            st.warning(f"The model has not been trained for temperature below {round(min(df['Temperature']))} C. MagNet AI is doing the extrapolation.")
        if temp > max(df['Temperature']):
            st.warning(f"The model has not been trained for temperature above {round(max(df['Temperature']))} C. MagNet AI is doing the extrapolation.")

    # Variables that are function of the sliders, different type depending on the excitation

    if excitation == 'Arbitrary':
        loss, not_extrapolated = core_loss_arbitrary(material, freq, flux_vector, temp, bias, duty_vector)
    else:
        loss, not_extrapolated = core_loss_default(material, freq, flux, temp, bias, duty, batched = False)

    # Representation of the waveform
    with col2:
        if excitation == "Arbitrary":
            waveform_visualization(
                st,
                x=duty_vector,
                y=np.multiply(flux_vector + flux_bias, 1e3))
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

    st.header(f'Output: Case {m}: {round(loss / 1e3 ,2)} kW/m^3')
    if not not_extrapolated:
        st.warning("Disclaimer: Data at extreme conditions is likely to be extrapolated. The neural network has not been trained by the measurement data in the corner cases.")
        
    info_string = f'{material_manufacturers[material]} - {material}, {excitation} excitation, ' \
                  f'f={round(freq / 1e3)} kHz, Bac={round(flux * 1e3)} mT, Bias={round(bias)} A/m'
    if excitation in ["Sinusoidal", "Arbitrary"]:
        st.write(f'{info_string}, '
                 f'T={round(temp)} C')
    if excitation == "Triangular":
        st.write(f'{info_string}, '
                 f'D={round(duty_p, 2)}, '
                 f'T={round(temp)} C')
    if excitation == "Trapezoidal":
        st.write(f'{info_string}, '
                 f'D1={round(duty_p, 2)}, D2={round(duty_0, 2)}, D3={round(duty_n, 2)}, D4={round(duty_0, 2)}, '
                 f'T={round(temp)} C')

    st.warning('''
                Solid curves: interporated prediction. MagNet AI is reproducing the training data. \n 
                Dash curves: extrapolated prediction. MagNet AI is doing its own prediction.
                ''')
    
    if excitation == "Arbitrary":
        col1, col2 = st.columns(2)
        with col1:
            st.write('Please bear in mind that the neural network has been trained using '
                     'trapezoidal, triangular and sinusoidal data. '
                     'The accuracy for waveforms very different from those used for training cannot be guaranteed.')
        with col2:
            bdata = np.interp(np.linspace(0, 1, c.streamlit.n_nn), np.array(duty_vector), np.array(flux_vector + flux_bias)) * 1e3
            output = {'B [mT]': bdata}
            csv = convert_df(pd.DataFrame(output))
            st.download_button(
                f"Download This Waveform as {c.streamlit.n_nn}-Step CSV File",
                data=csv,
                file_name='B-Input.csv',
                mime='text/csv',
            )

    else:
        # Plots for different sweeps
        col1, col2 = st.columns(2)

        if excitation == "Sinusoidal":
            subtitle_plot = ', for a fixed Temperature and Bias'
        if excitation == "Triangular":
            subtitle_plot = f' for a fixed Duty Cycle ({duty}), Temperature and Bias'
        if excitation == "Trapezoidal":
            subtitle_plot = f' for a fixed Temperature, Bias, and the Selected Duty Cycles'

        plot_bar = st.progress(0)
        with st.spinner('MagNet AI is Plotting the Design Graphs, Please Wait...'):

            with col1:  # vs frequency
                d_freq = np.tile(np.array(c.streamlit.core_loss_freq), 3)
                d_flux = np.concatenate(
                    (np.tile(flux, len(c.streamlit.core_loss_freq)),
                     np.tile(flux*1.2, len(c.streamlit.core_loss_freq)),
                     np.tile(flux*0.8, len(c.streamlit.core_loss_freq))))
                d_temp = np.tile(temp, len(c.streamlit.core_loss_freq)*3)
                d_bias = np.tile(bias, len(c.streamlit.core_loss_freq)*3)
                d_duty = [duty]*len(c.streamlit.core_loss_freq)*3
                d_loss, not_extrapolated = core_loss_default(material=material, 
                                         freq=d_freq, flux=d_flux, temp=d_temp, bias=d_bias, duty=d_duty, 
                                         batched = True)               
                d_loss[d_loss < core_loss_range[material][0]] = "NaN"
                d_loss[d_loss > core_loss_range[material][1]] = "NaN"       
                plot_core_loss(
                    st,
                    x=[freq / 1e3 for freq in c.streamlit.core_loss_freq],
                    y=1e-3 * d_loss[0:len(c.streamlit.core_loss_freq)],
                    y_upper=1e-3 * d_loss[len(c.streamlit.core_loss_freq):2*len(c.streamlit.core_loss_freq)],
                    y_lower=1e-3 * d_loss[2*len(c.streamlit.core_loss_freq):3*len(c.streamlit.core_loss_freq)],
                    x0=list([freq / 1e3]),
                    y0=list([1e-3 * loss]),
                    legend=f'{round(flux * 1e3)} mT',
                    legend_upper=f'{round(flux * 1.2 * 1e3)} mT',
                    legend_lower=f'{round(flux * 0.8 * 1e3)} mT',
                    title=f'<b> Core Loss Sweeping Frequency </b>'
                          f'<br> at a Few Flux Densities {subtitle_plot}',
                    x_title='Frequency [kHz]',
                    not_extrapolated = not_extrapolated
                )
                plot_bar.progress(10)

            with col2:  # vs flux density
                d_freq = np.concatenate(
                    (np.tile(freq, len(c.streamlit.core_loss_flux)),
                     np.tile(freq*2.0, len(c.streamlit.core_loss_flux)),
                     np.tile(freq*0.5, len(c.streamlit.core_loss_flux))))
                d_flux = np.tile(np.array(c.streamlit.core_loss_flux), 3)
                d_temp = np.tile(temp, len(c.streamlit.core_loss_flux)*3)
                d_bias = np.tile(bias, len(c.streamlit.core_loss_flux)*3)
                d_duty = [duty]*len(c.streamlit.core_loss_flux)*3
                d_loss, not_extrapolated = core_loss_default(material=material, 
                                         freq=d_freq, flux=d_flux, temp=d_temp, bias=d_bias, duty=d_duty, 
                                         batched = True)
                d_loss[d_loss < core_loss_range[material][0]] = "NaN"
                d_loss[d_loss > core_loss_range[material][1]] = "NaN"
                plot_core_loss(
                    st,
                    x=[flux * 1e3 for flux in c.streamlit.core_loss_flux],
                    y=1e-3 * d_loss[0:len(c.streamlit.core_loss_flux)],
                    y_upper=1e-3 * d_loss[len(c.streamlit.core_loss_flux):2*len(c.streamlit.core_loss_flux)],
                    y_lower=1e-3 * d_loss[2*len(c.streamlit.core_loss_flux):3*len(c.streamlit.core_loss_flux)],
                    x0=list([flux * 1e3]),
                    y0=list([1e-3 * loss]),
                    legend=f'{round(freq * 1e-3)} kHz',
                    legend_upper=f'{round(freq * 2.0 * 1e-3)} kHz',
                    legend_lower=f'{round(freq * 0.5 * 1e-3)} kHz',
                    title=f'<b> Core Loss Sweeping Flux Density </b>'
                          f'<br> at a fixed Frequency {subtitle_plot}',
                    x_title='AC Flux Density [mT]',
                    not_extrapolated = not_extrapolated
                )
                plot_bar.progress(20)

                if excitation == "Triangular":
                    with col1:  # vs duty at a few flux densities
                        d_freq = np.tile(freq, len(c.streamlit.core_loss_duty)*3)
                        d_flux = np.concatenate(
                            (np.tile(flux, len(c.streamlit.core_loss_duty)),
                             np.tile(flux*1.2, len(c.streamlit.core_loss_duty)),
                             np.tile(flux*0.8, len(c.streamlit.core_loss_duty))))
                        d_temp = np.tile(temp, len(c.streamlit.core_loss_duty)*3)
                        d_bias = np.tile(bias, len(c.streamlit.core_loss_duty)*3)
                        d_duty = np.tile(np.array(c.streamlit.core_loss_duty), 3)
                        d_loss, not_extrapolated = core_loss_default(material=material, 
                                                 freq=d_freq, flux=d_flux, temp=d_temp, bias=d_bias, duty=d_duty, 
                                                 batched = True)
                        d_loss[d_loss < core_loss_range[material][0]] = "NaN"
                        d_loss[d_loss > core_loss_range[material][1]] = "NaN"
                        plot_core_loss(
                            st,
                            x=c.streamlit.core_loss_duty,
                            y=1e-3 * d_loss[0:len(c.streamlit.core_loss_duty)],
                            y_upper=1e-3 * d_loss[len(c.streamlit.core_loss_duty):2*len(c.streamlit.core_loss_duty)],
                            y_lower=1e-3 * d_loss[2*len(c.streamlit.core_loss_duty):3*len(c.streamlit.core_loss_duty)],
                            x0=list([duty_p]),
                            y0=list([1e-3 * loss]),
                            legend=f'{round(freq * 1e-3)} kHz, {round(flux * 1e3)} mT',
                            legend_upper=f'{round(freq * 1e-3)} kHz, {round(flux * 1.2 * 1e3)} mT',
                            legend_lower=f'{round(freq * 1e-3)} kHz, {round(flux * 0.8 * 1e3)} mT',
                            title=f'<b> Core Loss Sweeping Duty Cycle </b>'
                                  f'<br> at a fixed Frequency, Temperature, and Bias',
                            x_title='Duty Cycle',
                            x_log=False,
                            y_log=True,
                            not_extrapolated = not_extrapolated
                        )
                        plot_bar.progress(30)
                        
                    with col2: # vs duty at a few frequencies
                        d_freq = np.concatenate(
                            (np.tile(freq, len(c.streamlit.core_loss_duty)),
                             np.tile(freq*2.0, len(c.streamlit.core_loss_duty)),
                             np.tile(freq*0.5, len(c.streamlit.core_loss_duty))))
                        d_flux = np.tile(flux, len(c.streamlit.core_loss_duty)*3)
                        d_temp = np.tile(temp, len(c.streamlit.core_loss_duty)*3)
                        d_bias = np.tile(bias, len(c.streamlit.core_loss_duty)*3)
                        d_duty = np.tile(np.array(c.streamlit.core_loss_duty), 3)
                        d_loss, not_extrapolated = core_loss_default(material=material, 
                                                 freq=d_freq, flux=d_flux, temp=d_temp, bias=d_bias, duty=d_duty, 
                                                 batched = True)
                        d_loss[d_loss < core_loss_range[material][0]] = "NaN"
                        d_loss[d_loss > core_loss_range[material][1]] = "NaN"
                        plot_core_loss(
                            st,
                            x=c.streamlit.core_loss_duty,
                            y=1e-3 * d_loss[0:len(c.streamlit.core_loss_duty)],
                            y_upper=1e-3 * d_loss[len(c.streamlit.core_loss_duty):2*len(c.streamlit.core_loss_duty)],
                            y_lower=1e-3 * d_loss[2*len(c.streamlit.core_loss_duty):3*len(c.streamlit.core_loss_duty)],
                            x0=list([duty_p]),
                            y0=list([1e-3 * loss]),
                            legend=f'{round(freq * 1e-3)} kHz, {round(flux * 1e3)} mT',
                            legend_upper=f'{round(freq * 2.0 * 1e-3)} kHz, {round(flux * 1e3)} mT',
                            legend_lower=f'{round(freq * 0.5 * 1e-3)} kHz, {round(flux * 1e3)} mT',
                            title=f'<b> Core Loss Sweeping Duty Cycle </b>'
                                  f'<br> at a fixed Flux Density, Temperature, and Bias',
                            x_title='Duty Cycle',
                            x_log=False,
                            y_log=True,
                            not_extrapolated = not_extrapolated
                        )
                        plot_bar.progress(40)

                with col1:  # vs dc bias at a few flux densities
                    d_freq = np.tile(freq, len(c.streamlit.core_loss_bias)*3)
                    d_flux = np.concatenate(
                        (np.tile(flux, len(c.streamlit.core_loss_bias)),
                         np.tile(flux*1.2, len(c.streamlit.core_loss_bias)),
                         np.tile(flux*0.8, len(c.streamlit.core_loss_bias))))
                    d_temp = np.tile(temp, len(c.streamlit.core_loss_bias)*3)
                    d_bias = np.tile(np.array(c.streamlit.core_loss_bias), 3)
                    d_duty = [duty]*len(c.streamlit.core_loss_bias)*3
                    d_loss, not_extrapolated = core_loss_default(material=material, 
                                             freq=d_freq, flux=d_flux, temp=d_temp, bias=d_bias, duty=d_duty, 
                                             batched = True)
                    d_loss[d_loss < core_loss_range[material][0]] = "NaN"
                    d_loss[d_loss > core_loss_range[material][1]] = "NaN"
                    plot_core_loss(
                        st,
                        x=c.streamlit.core_loss_bias,
                        y=1e-3 * d_loss[0:len(c.streamlit.core_loss_bias)],
                        y_upper=1e-3 * d_loss[len(c.streamlit.core_loss_bias):2*len(c.streamlit.core_loss_bias)],
                        y_lower=1e-3 * d_loss[2*len(c.streamlit.core_loss_bias):3*len(c.streamlit.core_loss_bias)],
                        x0=list([bias]),
                        y0=list([1e-3 * loss]),
                        legend=f'{round(freq * 1e-3)} kHz, {round(flux * 1e3)} mT',
                        legend_upper=f'{round(freq * 1e-3)} kHz, {round(flux * 1.2 * 1e3)} mT',
                        legend_lower=f'{round(freq * 1e-3)} kHz, {round(flux * 0.8 * 1e3)} mT',
                        title=f'<b> Core Loss Sweeping DC Bias </b>'
                              f'<br> at a fixed Frequency',
                        x_title='DC Bias [A/m]',
                        x_log=False,
                        y_log=True,
                        not_extrapolated = not_extrapolated
                    )
                    plot_bar.progress(50)

                with col2:  # vs dc bias at a few frequencies
                    d_freq = np.concatenate(
                        (np.tile(freq, len(c.streamlit.core_loss_bias)),
                         np.tile(freq*2.0, len(c.streamlit.core_loss_bias)),
                         np.tile(freq*0.5, len(c.streamlit.core_loss_bias))))
                    d_flux = np.tile(flux, len(c.streamlit.core_loss_bias)*3)
                    d_temp = np.tile(temp, len(c.streamlit.core_loss_bias)*3)
                    d_bias = np.tile(np.array(c.streamlit.core_loss_bias), 3)
                    d_duty = [duty]*len(c.streamlit.core_loss_bias)*3
                    d_loss, not_extrapolated = core_loss_default(material=material, 
                                             freq=d_freq, flux=d_flux, temp=d_temp, bias=d_bias, duty=d_duty, 
                                             batched = True)
                    d_loss[d_loss < core_loss_range[material][0]] = "NaN"
                    d_loss[d_loss > core_loss_range[material][1]] = "NaN"
                    plot_core_loss(
                        st,
                        x=c.streamlit.core_loss_bias,
                        y=1e-3 * d_loss[0:len(c.streamlit.core_loss_bias)],
                        y_upper=1e-3 * d_loss[len(c.streamlit.core_loss_bias):2*len(c.streamlit.core_loss_bias)],
                        y_lower=1e-3 * d_loss[2*len(c.streamlit.core_loss_bias):3*len(c.streamlit.core_loss_bias)],
                        x0=list([bias]),
                        y0=list([1e-3 * loss]),
                        legend=f'{round(freq * 1e-3)} kHz, {round(flux * 1e3)} mT',
                        legend_upper=f'{round(freq*2.0 * 1e-3)} kHz, {round(flux * 1e3)} mT',
                        legend_lower=f'{round(freq*0.5 * 1e-3)} kHz, {round(flux * 1e3)} mT',
                        title=f'<b> Core Loss Sweeping DC Bias </b>'
                              f'<br> at a fixed Flux Density',
                        x_title='DC Bias [A/m]',
                        x_log=False,
                        y_log=True,
                        not_extrapolated = not_extrapolated
                    )
                    plot_bar.progress(60)

                with col1:  # vs temperature at a few flux densities
                    d_freq = np.tile(freq, len(c.streamlit.core_loss_temp)*3)
                    d_flux = np.concatenate(
                        (np.tile(flux, len(c.streamlit.core_loss_temp)),
                         np.tile(flux*1.2, len(c.streamlit.core_loss_temp)),
                         np.tile(flux*0.8, len(c.streamlit.core_loss_temp))))
                    d_temp = np.tile(np.array(c.streamlit.core_loss_temp), 3)
                    d_bias = np.tile(bias, len(c.streamlit.core_loss_temp)*3)
                    d_duty = [duty]*len(c.streamlit.core_loss_temp)*3
                    d_loss, not_extrapolated = core_loss_default(material=material, 
                                             freq=d_freq, flux=d_flux, temp=d_temp, bias=d_bias, duty=d_duty, 
                                             batched = True)
                    d_loss[d_loss < core_loss_range[material][0]] = "NaN"
                    d_loss[d_loss > core_loss_range[material][1]] = "NaN"
                    plot_core_loss(
                        st,
                        x=c.streamlit.core_loss_temp,
                        y=1e-3 * d_loss[0:len(c.streamlit.core_loss_temp)],
                        y_upper=1e-3 * d_loss[len(c.streamlit.core_loss_temp):2*len(c.streamlit.core_loss_temp)],
                        y_lower=1e-3 * d_loss[2*len(c.streamlit.core_loss_temp):3*len(c.streamlit.core_loss_temp)],
                        x0=list([temp]),
                        y0=list([1e-3 * loss]),
                        legend=f'{round(freq * 1e-3)} kHz, {round(flux * 1e3)} mT',
                        legend_upper=f'{round(freq * 1e-3)} kHz, {round(1.2 * flux * 1e3)} mT',
                        legend_lower=f'{round(freq * 1e-3)} kHz, {round(flux *0.8 * 1e3)} mT',
                        title=f'<b> Core Loss Sweeping Temperature </b>'
                              f'<br> at a fixed Frequency',
                        x_title='Temperature [C]',
                        x_log=False,
                        y_log=True,
                        not_extrapolated = not_extrapolated
                    )
                    plot_bar.progress(70)

                with col2:  # vs temperature at a few frequencies
                    d_freq = np.concatenate(
                        (np.tile(freq, len(c.streamlit.core_loss_temp)),
                         np.tile(freq*2.0, len(c.streamlit.core_loss_temp)),
                         np.tile(freq*0.5, len(c.streamlit.core_loss_temp))))
                    d_flux = np.tile(flux, len(c.streamlit.core_loss_temp)*3)
                    d_temp = np.tile(np.array(c.streamlit.core_loss_temp), 3)
                    d_bias = np.tile(bias, len(c.streamlit.core_loss_temp)*3)
                    d_duty = [duty]*len(c.streamlit.core_loss_temp)*3
                    d_loss, not_extrapolated = core_loss_default(material=material, 
                                             freq=d_freq, flux=d_flux, temp=d_temp, bias=d_bias, duty=d_duty, 
                                             batched = True)
                    d_loss[d_loss < core_loss_range[material][0]] = "NaN"
                    d_loss[d_loss > core_loss_range[material][1]] = "NaN"
                    plot_core_loss(
                        st,
                        x=c.streamlit.core_loss_temp,
                        y=1e-3 * d_loss[0:len(c.streamlit.core_loss_temp)],
                        y_upper=1e-3 * d_loss[len(c.streamlit.core_loss_temp):2*len(c.streamlit.core_loss_temp)],
                        y_lower=1e-3 * d_loss[2*len(c.streamlit.core_loss_temp):3*len(c.streamlit.core_loss_temp)],
                        x0=list([temp]),
                        y0=list([1e-3 * loss]),
                        legend=f'{round(freq * 1e-3)} kHz, {round(flux * 1e3)} mT',
                        legend_upper=f'{round(freq * 2.0 * 1e-3)} kHz, {round(flux * 1e3)} mT',
                        legend_lower=f'{round(freq * 0.5 * 1e-3)} kHz, {round(flux * 1e3)} mT',
                        title=f'<b> Core Loss Sweeping Temperature </b>'
                              f'<br> at a fixed Flux Density',
                        x_title='Temperature [C]',
                        x_log=False,
                        y_log=True,
                        not_extrapolated = not_extrapolated
                    )
                    plot_bar.progress(100)
                    
            st.success('Done!')

    st.markdown("""---""")
