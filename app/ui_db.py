import streamlit as st
import numpy as np

from magnet import config as c
from magnet.constants import material_names, material_manufacturers
from magnet.io import load_dataframe, load_metadata
from magnet.plots import scatter_plot, waveform_visualization_2axes, cycle_points_sinusoidal, cycle_points_trapezoidal


def ui_core_loss_dbs(n=1):

    for i in range(int(n)):
        ui_core_loss_db(chr(ord('A') + i))


def ui_core_loss_db(m):
    st.sidebar.header(f'Information: Case {m}')
    excitation = st.sidebar.selectbox(
        f'Excitation:',
        ('Sinusoidal', 'Triangular', 'Trapezoidal'),
        key=f'excitation {m}',
        index=1)

    material = st.sidebar.selectbox(
        f'Material:',
        material_names,
        key=f'material {m}')

    [freq_min_aux, freq_max_aux] = st.sidebar.slider(
        f'Frequency Range (kHz)',
        50,
        500,
        (50, 500),
        step=1,
        key=f'freq {m}')
    freq_min = freq_min_aux * 1e3
    freq_max = freq_max_aux * 1e3
    freq_avg = (freq_max + freq_min) / 2

    [flux_min_aux, flux_max_aux] = st.sidebar.slider(
        f'AC Flux Density Range (mT)',
        10,
        300,
        (10, 300),
        step=5,
        key=f'flux {m}',
        help=f'Amplitude of the AC signal, not peak to peak')
    flux_min = flux_min_aux / 1e3
    flux_max = flux_max_aux / 1e3
    flux_avg = (flux_max + flux_min) / 2

    df = load_dataframe(material)
    if len(df['DC_Bias']) == 0:
        dc_bias = 0
        st.sidebar.markdown(f'##### Only data without DC bias is measured'),
    else:
        dc_bias = st.sidebar.slider(
            f'DC bias (A/m)',
            0,
            round(round(max(df['DC_Bias']) / 15) * 15),
            0,
            step=15,
            key=f'bias {m}',
            help=f'DC bias in the H field')

    if excitation == 'Triangular':
        duty_p = st.sidebar.slider(
            f'Duty Cycle',
            0.1,
            0.9,
            0.5,
            step=0.1,
            key=f'duty {m}')
        duty_n = 1.0 - duty_p  # For triangular excitation, there are no flat parts
        duty_0 = 0.0

    if excitation == 'Trapezoidal':
        duty_p = st.sidebar.slider(
            f'Duty Ratio (D1)',
            0.1,
            0.9 - 2 * 0.1,
            0.5 - 0.1,
            step=0.1,
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

        duty_0 = (1-duty_p-duty_n)/2
        st.sidebar.markdown(f'##### Duty cycle D2=D4=(1-D1-D3)/2)={round(duty_0 * 10) / 10}'),

    # temperature = st.sidebar.radio(
    #     f'Temperature (C)',
    #     [25, 50, 70, 90],
    #     key=f'temp {m}',
    #     horizontal=False)  # TODO Switch to radio buttons once we update the Streamlit version

    temperature = st.sidebar.selectbox(
        f'Temperature (C)',
        [25, 50, 70, 90],
        key=f'temp {m}')

    c_axis = st.sidebar.selectbox(
        f'Select Color-Axis for the Plots:',
        ['Flux Density', 'Frequency', 'Power Loss'],
        key=f'c_axis {m}')

    st.sidebar.markdown("""---""")

    if excitation == 'Sinusoidal':
        df = load_dataframe(material, freq_min, freq_max, flux_min, flux_max, dc_bias, -1.0, -1.0, temperature)
    if excitation in ['Triangular', 'Trapezoidal']:
        df = load_dataframe(material, freq_min, freq_max, flux_min, flux_max, dc_bias, duty_p, duty_n, temperature)

    col1, col2, col3 = st.columns([2, 3, 4])
    with col1:
        st.header(f'Core Loss Database: Case {m}')
        st.write(f'{material_manufacturers[material]} - {material}, '
                     f'{excitation} excitation')
        st.write(f'f=[{round(freq_min / 1e3)}-{round(freq_max / 1e3)}] kHz, '
                     f'B=[{round(flux_min * 1e3)}-{round(flux_max * 1e3)}] mT, '
                     f'Bias={round(dc_bias)} A/m')
        if excitation == "Sinusoidal":
            st.write(f'T={round(temperature)} C')
        if excitation == "Triangular":
            st.write(f'D={round(duty_p, 2)}, '
                         f'T={round(temperature)} C')
        if excitation == "Trapezoidal":
            st.write(f'D1={round(duty_p, 2)}, '
                         f'D2={round(duty_0, 2)}, '
                         f'D3={round(duty_n, 2)}, '
                         f'D4={round(duty_0, 2)}, '
                         f'T={round(temperature)} C')

        if df.empty:
            st.subheader("Warning: no data in range, please change the range")
        else:
            with st.expander('Measurement details'):
                metadata = load_metadata(material)
                st.write('Core information: ', metadata['info_core'])
                st.write('Setup information: ', metadata['info_setup'])
                st.write('Data-processing information: ', metadata['info_processing'])
                st.write(
                    'Note: the dc bias, duty cycles and temperature have small variations with respect to the data '
                    'reported here, this data has been rounded for visualization purposes. '
                    'The measurements can be obtain from the download section.')

            st.subheader(f'Download data:')
            df_csv = df[['Frequency', 'Flux_Density', 'Power_Loss']]
            file = df_csv.to_csv(index=False).encode('utf-8')
            if excitation == "Sinusoidal":
                csv_name = material + '-' + excitation + '_' + str(dc_bias) + 'Am-1(bias)_'\
                           + str(temperature) + 'C(temp).csv'
            if excitation == "Triangular":
                csv_name = material + '-' + excitation + '_' + str(dc_bias) + 'Am-1(bias)_'\
                           + str(duty_p) + '(duty)_' + str(temperature) + 'C(temp).csv'
            if excitation == "Trapezoidal":
                csv_name = material + '-' + excitation + '_' + str(dc_bias) + 'Am-1(bias)_'\
                           + str(duty_p) + '(D1)_' + str(duty_n) + '(D3)_' + str(temperature) + 'C(temp).csv'

            st.download_button(
                'Download CSV',
                file,
                csv_name,
                'text/csv',
                key=m,
                help='Download a .csv file containing the flux, frequency, and '
                     'power loss for the depicted data points')

    with col2:
        if excitation == 'Sinusoidal':
            [cycle_list, flux_list, volt_list] = cycle_points_sinusoidal(c.streamlit.n_points_plot)
        if excitation in ['Triangular', 'Trapezoidal']:
            [cycle_list, flux_list, volt_list] = cycle_points_trapezoidal(duty_p, duty_n, duty_0)
        flux_vector = np.multiply(flux_list, flux_avg)

        waveform_visualization_2axes(
            st,
            x1=np.multiply(cycle_list, 1e6 / freq_avg),  # In us
            x2=cycle_list,  # Percentage
            y1=np.multiply(flux_vector, 1e3),  # In mT
            y2=volt_list,  # Per unit
            x1_aux=cycle_list,  # Percentage
            y1_aux=flux_list,
            title=f"<b>Waveform visualization</b>"
                  f"<br>f={format(freq_avg / 1e3, '.0f')} kHz, B={format(flux_avg * 1e3, '.0f')} mT")
    with col3:
        if not df.empty:
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

    st.markdown("""---""")


def ui_core_loss_db_inter(m):

    excitation = st.sidebar.selectbox(
        f'Excitation:',
        ('Datasheet', 'Sinusoidal', 'Triangular', 'Trapezoidal (symmetric)'),
        key=f'excitation {m}',
        index=1)

    material = st.sidebar.selectbox(
        f'Material:',
        [elem for elem in material_names if elem not in {'N30', '3E6'}]  # No datasheet info for 3E6 or N30
        if excitation == 'Datasheet' else material_names,
        key=f'material {m}')

    if excitation == 'Datasheet':
        var_interpolation = 'Temperature'
    if excitation == 'Sinusoidal':
        var_interpolation = st.sidebar.selectbox(
            f'Variable to interpolate:',
            ['Temperature', 'DC Bias'],
            key=f'var_inter {m}')
    if excitation == 'Triangular':
        var_interpolation = st.sidebar.selectbox(
            f'Variable to interpolate:',
            ['Temperature', 'DC Bias', 'Duty Cycle'],
            key=f'var_inter {m}')
    if excitation == 'Trapezoidal (symmetric)':
        var_interpolation = st.sidebar.selectbox(
            f'Variable to interpolate:',
            ['Temperature', 'DC Bias', 'Duty 0'],
            key=f'var_inter {m}')

    if var_interpolation == 'Temperature':
        variable = 'temperature'
    if var_interpolation == 'DC Bias':
        variable = 'dc_bias'
    if var_interpolation == 'Duty Cycle':
        variable = 'duty_p'
    if var_interpolation == 'Duty 0':
        variable = 'duty_0'

    [freq_min_aux, freq_max_aux] = st.sidebar.slider(
        f'Frequency Range (kHz)',
        50,
        500,
        (50, 500),
        step=1,
        key=f'freq {m}')
    freq_min = freq_min_aux * 1e3
    freq_max = freq_max_aux * 1e3
    freq_avg = (freq_max + freq_min) / 2

    [flux_min_aux, flux_max_aux] = st.sidebar.slider(
        f'AC Flux Density Range (mT)',
        10,
        300,
        (10, 300),
        step=1,
        key=f'flux {m}',
        help=f'Amplitude of the AC signal, not peak to peak')
    flux_min = flux_min_aux / 1e3
    flux_max = flux_max_aux / 1e3
    flux_avg = (flux_max + flux_min) / 2

    if excitation == 'Datasheet':
        dc_bias = 0
    else:
        dc_bias = st.sidebar.slider(
            f'DC bias (A/m)',
            0,
            90,
            15,
            step=5 if var_interpolation == 'DC Bias' else 15,
            key=f'bias {m}',
            help=f'DC bias in the H field')

    if excitation == 'Triangular':
        duty_p = st.sidebar.slider(
            f'Duty Cycle',
            0.1,
            0.9,
            0.5,
            step=0.05 if var_interpolation == 'Duty Cycle' else 0.1,
            key=f'duty {m}')
        duty_n = 1.0 - duty_p  # For triangular excitation, there are no flat parts
        duty_0 = 0.0

    if excitation == 'Trapezoidal (symmetric)':
        duty_0 = st.sidebar.slider(
            f'Duty Cycle (D0)',
            0.0 if var_interpolation == 'Duty 0' else 0.1,
            0.4,
            0.3,
            step=0.05 if var_interpolation == 'Duty 0' else 0.1,
            key=f'duty0 {m}',
            help=f'Duty cycle of the zero-voltage regions')
        duty_p = (1 - 2 * duty_0)/2
        duty_n = duty_p

        st.sidebar.markdown(f'##### Duty cycle D1=D3=(1-2*D0)/2)={round(duty_p * 10) / 10}'),

    if var_interpolation == 'Temperature':
        temperature = st.sidebar.slider(
            f'Temperature (C)',
            0 if excitation == 'Datasheet' else 25,
            120 if excitation == 'Datasheet' else 90,
            35,
            step=5,
            key=f'temp {m}')
    else:
        temperature = st.sidebar.selectbox(
            f'Temperature (C)',
            [25, 50, 70, 90],
            key=f'temp {m}')

    st.sidebar.markdown("""---""")

    if excitation == 'Datasheet':
        df = load_dataframe_datasheet(material, freq_min, freq_max, flux_min, flux_max, temperature)
    if excitation == 'Sinusoidal':
        df = load_data_inter(material, variable, freq_min, freq_max, flux_min, flux_max, dc_bias, -1.0, -1.0, temperature)
    if excitation in ['Triangular', 'Trapezoidal (symmetric)']:
        df = load_data_inter(material, variable, freq_min, freq_max, flux_min, flux_max, dc_bias, duty_p, duty_n, temperature)

    col1, col2 = st.columns(2)
    with col1:
        st.title(f'Interpolated Core Loss: Case {m}')
        st.subheader(f'{material_manufacturers[material]} - {material}, '
                     f'{excitation} excitation')
        st.subheader(f'f=[{round(freq_min / 1e3)}-{round(freq_max / 1e3)}] kHz, '
                     f'B=[{round(flux_min * 1e3)}-{round(flux_max * 1e3)}] mT, '
                     f'Bias={round(dc_bias)} A/m')
        if excitation == "Triangular":
            st.subheader(f'D={round(duty_p, 2)}')
        if excitation == "Trapezoidal (symmetric)":
            st.subheader(f'D1={round(duty_p, 2)}, '
                         f'D2={round(duty_0, 2)}, '
                         f'D3={round(duty_n, 2)}, '
                         f'D4={round(duty_0, 2)}')

        st.subheader(f'T={round(temperature)} C')

        if df.empty:
            st.subheader("Warning: no data in range, please change the range")
        else:
            with st.expander('Interpolation details'):
                st.write('Data linearly interpolated for Bac and f using the selected third variable (T, Hdc, D, or D0).'
                         ' log10(PV), log10(f), log10(Bac), T, Hdc, D, and D0 used')

    with col2:
        if excitation in ['Datasheet', 'Sinusoidal']:
            [cycle_list, flux_list, volt_list] = cycle_points_sinusoidal(c.streamlit.n_points_plot)
        if excitation in ['Triangular', 'Trapezoidal (symmetric)']:
            [cycle_list, flux_list, volt_list] = cycle_points_trapezoidal(duty_p, duty_n, duty_0)
        flux_vector = np.multiply(flux_list, flux_avg)

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
                x='Flux_Density_mT',
                y='Frequency_kHz',
                c='Power_Loss_kW/m3'),
                use_container_width=True,)

    st.markdown("""---""")


def ui_BH_db_meas(m):

    excitation = st.sidebar.selectbox(
        f'Excitation:',
        ('Sinusoidal', 'Triangular', 'Trapezoidal (symmetric)'),
        key=f'excitation {m}',
        index=1)

    material = st.sidebar.selectbox(
        f'Material:',
        material_names,
        key=f'material {m}')

    [freq_min_aux, freq_max_aux] = st.sidebar.slider(
        f'Frequency Range (kHz)',
        50,
        500,
        (50, 500),
        step=1,
        key=f'freq {m}')
    freq_min = freq_min_aux * 1e3
    freq_max = freq_max_aux * 1e3

    [flux_min_aux, flux_max_aux] = st.sidebar.slider(
        f'AC Flux Density Range (mT)',
        10,
        300,
        (10, 300),
        step=5,
        key=f'flux {m}',
        help=f'Amplitude of the AC signal, not peak to peak')
    flux_min = flux_min_aux / 1e3
    flux_max = flux_max_aux / 1e3

    [dc_bias_min, dc_bias_max] = st.sidebar.slider(
        f'DC bias (A/m)',
        0,
        150,
        (0, 30),
        step=15,
        key=f'bias {m}',
        help=f'DC bias in the H field')

    if excitation == 'Triangular':
        [duty_p_max, duty_p_min] = st.sidebar.slider(
            f'Duty Cycle',
            0.1,
            0.9,
            (0.4, 0.6),
            step=0.1,
            key=f'duty {m}')
        duty_p_avg = (duty_p_max + duty_p_min) / 2

        duty_n_max = 1.0 - duty_p_min
        duty_n_min = 1.0 - duty_p_max
        duty_n_avg = (duty_n_max + duty_n_min) / 2
        duty_0 = 0.0

    if excitation == 'Trapezoidal (symmetric)':
        [duty_0_min, duty_0_max] = st.sidebar.slider(
            f'Duty Cycle (D0)',
            0.0,
            0.4,
            (0.2, 0.3),
            step=0.1,
            key=f'duty0 {m}',
            help=f'Duty cycle of the zero-voltage regions')
        duty_p_max = (1 - 2 * duty_0_min)/2
        duty_p_min = (1 - 2 * duty_0_max)/2
        duty_n_max = duty_p_max
        duty_n_min = duty_p_min
        st.sidebar.markdown(f'##### Duty cycle D1=D3=(1-2*D0)/2)=[{round(duty_p_min * 10) / 10},{round(duty_p_max * 10) / 10}]'),

    [temperature_min, temperature_max] = st.sidebar.slider(
        f'Temperature (C)',
        25,
        90,
        (25, 30),
        step=5,
        key=f'temp {m}')
