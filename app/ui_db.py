import streamlit as st
import numpy as np

from magnet import config as c
from magnet.constants import material_names, material_manufacturers, excitations_db
from magnet.io import load_dataframe, load_dataframe_datasheet, load_metadata
from magnet.plots import scatter_plot, waveform_visualization_2axes, cycle_points_sinusoidal, cycle_points_trapezoidal


def ui_core_loss_dbs(n=1):
    for i in range(int(n)):
        ui_core_loss_db(chr(ord('A') + i))


def ui_core_loss_db(m):
    st.sidebar.header(f'Information: Case {m}')
    excitation = st.sidebar.selectbox(
        f'Excitation:',
        excitations_db,
        key=f'excitation {m}',
        index=1)

    if excitation == 'Datasheet':  # No datasheet info for 3E6 or N30
        material = st.sidebar.selectbox(
            f'Material:',
            [elem for elem in material_names if elem not in {'N30', '3E6'}],
            key=f'material {m}')
    else:
        material = st.sidebar.selectbox(
            f'Material:',
            material_names,
            key=f'material {m}')

    [freq_min_aux, freq_max_aux] = st.sidebar.slider(
        f'Frequency Range (kHz)',
        round(c.streamlit.freq_min / 1e3),
        round(c.streamlit.freq_max / 1e3),
        (round(c.streamlit.freq_min / 1e3), round(c.streamlit.freq_max / 1e3)),
        step=round(c.streamlit.freq_step_db / 1e3),
        key=f'freq {m}')
    freq_min = freq_min_aux * 1e3
    freq_max = freq_max_aux * 1e3
    freq_avg = (freq_max + freq_min) / 2

    [flux_min_aux, flux_max_aux] = st.sidebar.slider(
        f'AC Flux Density Range (mT)',
        round(c.streamlit.flux_min * 1e3),
        round(c.streamlit.flux_max * 1e3),
        (round(c.streamlit.flux_min * 1e3), round(c.streamlit.flux_max * 1e3)),
        step=round(c.streamlit.flux_step_db * 1e3),
        key=f'flux {m}',
        help=f'Amplitude of the AC signal, not peak to peak')
    flux_min = flux_min_aux / 1e3
    flux_max = flux_max_aux / 1e3
    flux_avg = (flux_max + flux_min) / 2

    flux_bias = st.sidebar.slider(
        f'DC Flux Density (mT) coming soon!',
        round(-c.streamlit.flux_max * 1e3),
        round(c.streamlit.flux_max * 1e3),
        0,
        step=round(1e9),
        key=f'bias {m}',
        help=f'Fixed at 0 mT for now') / 1e3  # 1e9 step to fix it

    if excitation == 'Triangular':
        duty_p = st.sidebar.slider(
            f'Duty Ratio',
            c.streamlit.duty_min,
            c.streamlit.duty_max,
            (c.streamlit.duty_min + c.streamlit.duty_max) / 2,
            step=c.streamlit.duty_step_db,
            key=f'duty {m}')
        duty_n = 1.0 - duty_p  # For triangular excitation, there are no flat parts
        duty_0 = 0.0
    if excitation == 'Trapezoidal':
        duty_p = st.sidebar.slider(
            f'Duty Ratio (D1)',
            c.streamlit.duty_min,
            c.streamlit.duty_max - 2 * c.streamlit.duty_step_db,
            (c.streamlit.duty_min + c.streamlit.duty_max) / 2 - c.streamlit.duty_step_db,
            step=c.streamlit.duty_step_db,
            key=f'dutyP {m}',
            help=f'Rising part with the highest slope')
        duty_n_max = 1.0 - duty_p - 0.2
        if duty_p in [0.1, 0.3, 0.5, 0.7]:  # TODO: probably there is a more elegant way to implement this
            duty_n_min = 0.1
        elif duty_p in [0.2, 0.4, 0.6]:
            duty_n_min = 0.2

        if duty_n_max <= duty_n_min+0.01:  # In case they are equal but implemented for floats
            duty_n = st.sidebar.slider(
                f'Duty Ratio (D3) Fixed',
                duty_n_max - 0.01,
                duty_n_max + 0.01,
                duty_n_max,
                step=1.0,
                key=f'dutyN {m}',
                help=f'Falling part with the highest slope, fixed by D1')  # Step outside the range to fix the variable
        else:
            duty_n = st.sidebar.slider(
                f'Duty Ratio (D3)',
                duty_n_min,
                duty_n_max,
                duty_n_max,
                step=2 * c.streamlit.duty_step_db,
                key=f'dutyN {m}',
                help=f'Falling part with the highest slope, maximum imposed by D1')
        duty_0 = st.sidebar.slider(
            f'Duty Ratio (D2=D4=(1-D1-D3)/2) Fixed',
            0.0,
            1.0,
            (1-duty_p-duty_n)/2,
            step=1e7,
            key=f'duty0 {m}',
            help=f'Low slope regions, fixed by D1 and D3')  # Step outside the range to fix the variable

    if excitation == 'Datasheet':
        temperature = st.sidebar.slider(
            f'Temperature (C)',
            round(c.streamlit.temp_min),
            round(c.streamlit.temp_max),
            round(c.streamlit.temp_default),
            step=round(c.streamlit.temp_step),
            key=f'temp {m}')
    else:
        temperature = st.sidebar.slider(
            f'Temperature (C) coming soon!',
            round(c.streamlit.temp_min),
            round(c.streamlit.temp_max),
            round(c.streamlit.temp_default),
            step=round(1e9),
            key=f'temp {m}',
            help=f'Fixed at 25 C for now')
        out_max = st.sidebar.slider(
            f'Maximum Outlier Factor (%)',
            round(c.streamlit.outlier_min),
            round(c.streamlit.outlier_max),
            round(c.streamlit.outlier_max),
            step=1,
            key=f'outlier {m}',
            help=f'Measures the similarity between the loss of a datapoint and their neighbours ' 
                 f'(in terms of B and f) based on local Steinmetz parameters')

    c_axis = st.sidebar.selectbox(
        f'Select Color-Axis for the Plots:',
        ['Flux Density', 'Frequency', 'Power Loss'],
        key=f'c_axis {m}')

    st.sidebar.markdown("""---""")

    if excitation == 'Triangular':
        read_excitation = 'Trapezoidal'  # Triangular data read from Trapezoidal files
    else:
        read_excitation = excitation

    if read_excitation == 'Datasheet':
        df = load_dataframe_datasheet(material, freq_min, freq_max, flux_min, flux_max, temperature)
    if read_excitation == 'Sinusoidal':
        df = load_dataframe(material, read_excitation, freq_min, freq_max, flux_min, flux_max, None, None, out_max)
    if read_excitation == 'Trapezoidal':
        df = load_dataframe(material, read_excitation, freq_min, freq_max, flux_min, flux_max, duty_p, duty_n, out_max)

    col1, col2 = st.columns(2)
    with col1:
        st.title(f'Core Loss Database: Case {m}')
        st.subheader(f'{material_manufacturers[material]} - {material}, '
                     f'{excitation} excitation')
        st.subheader(f'f=[{round(freq_min / 1e3)}~{round(freq_max / 1e3)}] kHz, '
                     f'B=[{round(flux_min * 1e3)}~{round(flux_max * 1e3)}] mT, '
                     f'Bias={round(flux_bias * 1e3)} mT')
        if excitation == "Datasheet":
            st.subheader(f'T={round(temperature)} C')
        else:
            st.subheader(f'Max outlier factor={out_max} %')
        if excitation == "Triangular":
            st.subheader(f'D={round(duty_p, 2)}')
        if excitation == "Trapezoidal":
            st.subheader(f'D1={round(duty_p, 2)}, '
                         f'D2={round(duty_0, 2)}, '
                         f'D3={round(duty_n, 2)}, '
                         f'D4={round(duty_0, 2)}')

        if df.empty:
            st.subheader("Warning: no data in range, please change the range")
        else:
            if excitation != 'Datasheet':
                with st.expander('Measurement details'):
                    metadata = load_metadata(material, read_excitation)
                    st.write(metadata['info_date'])
                    st.write(metadata['info_excitation'])
                    if excitation in ['Sinusoidal', 'Triangular', 'Trapezoidal']:
                        st.write(metadata['info_core'])  # The datasheet is not associated with a specific core
            st.subheader(f'Download data:')

            if read_excitation == 'Datasheet':
                df_csv = df[['Frequency', 'Flux_Density', 'Temperature', 'Power_Loss']]
            if read_excitation == 'Sinusoidal':
                df_csv = df[['Frequency', 'Flux_Density', 'Power_Loss', 'Outlier_Factor']]
            if read_excitation == 'Trapezoidal':
                df_csv = df[['Frequency', 'Flux_Density', 'Power_Loss', 'Duty_1', 'Duty_2', 'Duty_3', 'Duty_4', 'Outlier_Factor']]
            file = df_csv.to_csv().encode('utf-8')
            st.download_button(
                'Download CSV',
                file,
                material + '-' + excitation + '.csv',
                'text/csv',
                key=m,
                help='Download a CSV file containing the flux, frequency, duty cycle,'
                     'power loss and outlier factor for the depicted data points')

    with col2:
        if excitation in ['Datasheet', 'Sinusoidal']:
            [cycle_list, flux_list, volt_list] = cycle_points_sinusoidal(c.streamlit.n_points_plot)
        if excitation in ['Triangular', 'Trapezoidal']:
            [cycle_list, flux_list, volt_list] = cycle_points_trapezoidal(duty_p, duty_n, duty_0)
        flux_vector = np.add(np.multiply(flux_list, flux_avg), flux_bias)

        waveform_visualization_2axes(
            st,
            x1=np.multiply(cycle_list, 1e6 / freq_avg),  # In us
            x2=cycle_list,  # Percentage
            y1=np.multiply(flux_vector, 1e3),  # In mT
            y2=volt_list,  # Per unit
            x1_aux=cycle_list,  # Percentage
            y1_aux=flux_list,
            title=f"<b>Waveform visualization</b> <br>"
                  f"f={format(freq_avg / 1e3, '.0f')} kHz, B={format(flux_avg * 1e3, '.0f')} mT")

    if df.empty or excitation == 'Datasheet':  # Second column not required
        col1, col2 = st.columns([5, 1])
    else:
        col1, col2 = st.columns([3, 3])

    if not df.empty:
        with col1:
            st.plotly_chart(scatter_plot(
                df,
                x='Frequency_kHz' if c_axis == 'Flux Density' else
                'Flux_Density_mT',
                y='Frequency_kHz' if c_axis == 'Power Loss' else
                'Power_Loss_kW/m3',
                c='Flux_Density_mT' if c_axis == 'Flux Density' else
                'Frequency_kHz' if c_axis == 'Frequency' else
                'Power_Loss_kW/m3'),
                use_container_width=True,)
        if excitation != 'Datasheet':
            with col2:
                st.plotly_chart(scatter_plot(
                    df,
                    x='Frequency_kHz' if c_axis == 'Flux Density' else
                    'Flux_Density_mT',
                    y='Frequency_kHz' if c_axis == 'Power Loss' else
                    'Power_Loss_kW/m3',
                    c='Outlier_Factor'),
                    use_container_width=True)

    st.markdown("""---""")
