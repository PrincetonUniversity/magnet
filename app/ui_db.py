import streamlit as st
import numpy as np

from magnet import config as c
from magnet.constants import material_names, material_manufacturers, excitations
from magnet.io import load_dataframe, load_metadata
from magnet.plots import scatter_plot, waveform_visualization_db


def header(material, excitation, f_min, f_max, b_min, b_max, Bbias=None, Temp=None, DutyP=None, DutyN=None):
    s = f"{material_manufacturers[material]} - {material}, {excitation} Data, f=[{format(f_min/1e3,'.0f')}~{format(f_max/1e3, '.0f')}] kHz, B=[{format(b_min*1e3, '.0f') }~{format(b_max*1e3, '.0f')}] mT"
    if Bbias is not None:
        s += f', Bdc={Bbias*1e3} mT'
    if Temp is not None:
        s += f', T={Temp} C'
    if DutyP is not None:
        if DutyN is None:
            s += f", D={format(DutyP,'.1f')}"
        if DutyN is not None:
            Duty0 = (1.0-DutyP-DutyN)/2.0
            s += f", D1={format(DutyP,'.1f')}, D2={format(Duty0,'.1f')}, D3={format(DutyN,'.1f')}, D4={format(Duty0,'.1f')}"
    return st.subheader(s)


def ui_core_loss_dbs(n=1):
    for i in range(int(n)):
        ui_core_loss_db(chr(ord('A') + i))


def ui_core_loss_db(m):
    st.sidebar.header(f'Information for Material {m}')
    excitation = st.sidebar.selectbox(f'Excitation:', excitations, key="excitation" + m, index=1)
    if excitation == 'Datasheet':  # No datasheet info for 3E6 or N30
        material = st.sidebar.selectbox(f'Material:', [elem for elem in material_names if elem not in {'N30', '3E6'}],
                                        key="material"+m)
    else:
        material = st.sidebar.selectbox(f'Material:', material_names, key="material" + m)
    c_axis = st.sidebar.selectbox(f'Select color-axis for Plotting:', ['Flux Density', 'Frequency', 'Power Loss'],
                                  key="c_axis"+m)

    [Fmin_kHz, Fmax_kHz] = st.sidebar.slider(
        f'Frequency Range (kHz)',
        c.streamlit.freq_min/1e3,
        c.streamlit.freq_max/1e3,
        (c.streamlit.freq_min/1e3, c.streamlit.freq_max/1e3),
        step=c.streamlit.freq_step/1e3,
        key="frequency"+m)

    Fmin = Fmin_kHz * 1e3
    Fmax = Fmax_kHz * 1e3
    Favg = (Fmax + Fmin) / 2

    [Bmin_mT, Bmax_mT] = st.sidebar.slider(
        f'AC Flux Density Amplitude Range (mT)',
        c.streamlit.flux_min*1e3,
        c.streamlit.flux_max*1e3,
        (c.streamlit.flux_min*1e3, c.streamlit.flux_max*1e3),
        step=c.streamlit.flux_step*1e3,
        key="flux"+m)

    Bmin = Bmin_mT / 1e3
    Bmax = Bmax_mT / 1e3
    Bavg = (Bmax + Bmin) / 2

    Bbias = st.sidebar.slider(
        f'DC Flux Density (mT) (coming soon)',
        -c.streamlit.flux_max*1e3,
        c.streamlit.flux_max*1e3,
        0.0,
        step=1e9,
        key="Bbias"+m,
        help="Fixed at 0 mT for now")  # 1e9 step to fix it

    # TODO: add the temperature filter for the datasheet plot
    if excitation == 'Datasheet':
        Temp = st.sidebar.slider(
            f'Temperature (C) (coming soon)',
            c.streamlit.temp_min,
            c.streamlit.temp_max,
            c.streamlit.temp_step,
            step=1e9,
            key="temp"+m,
            help="Fixed at 25 C for now")  # 1e9 step to fix it

    if excitation == 'Triangular':
        DutyP = st.sidebar.slider(
            f'Duty Ratio',
            c.streamlit.duty_min_db,
            c.streamlit.duty_max_db,
            0.5,
            step=c.streamlit.duty_step_db,
            key="duty"+m)
        DutyN = 1.0-DutyP  # For triangular excitation, there are no flat parts
        Duty0 = 0.0
    if excitation == 'Trapezoidal':
        DutyP = st.sidebar.slider(f'Duty Ratio (Rising)',
                                  c.streamlit.duty_min_db,
                                  c.streamlit.duty_max_db-2*c.streamlit.duty_step_db,
                                  0.4,
                                  step=c.streamlit.duty_step_db,
                                  key="dutyP"+m,
                                  help="D1")
        DutyNmax = 1.0-DutyP-0.2
        if DutyP in [0.1, 0.3, 0.5, 0.7]:  # TODO: probably there is a more elegant way to implement this
            DutyNmin = 0.1
        elif DutyP in [0.2, 0.4, 0.6]:
            DutyNmin = 0.2

        if DutyNmax <= DutyNmin+0.01:  # In case they are equal but implemented for floats
            DutyN = st.sidebar.slider(
                f'Duty Ratio (Falling) Fixed',
                DutyNmax-0.01,
                DutyNmax+0.01,
                DutyNmax,
                step=1.0,
                key="dutyN"+m,
                help="D3, fixed by D1")  # Step outside the range to fix the variable
        else:
            DutyN = st.sidebar.slider(
                f'Duty Ratio (Falling)',
                DutyNmin,
                DutyNmax,
                DutyNmax,
                step=2*c.streamlit.duty_step_db,
                key="dutyN"+m,
                help="D3, maximum imposed by D1")
        Duty0 = st.sidebar.slider(
            f'Duty Ratio (Flat) Fixed',
            (1.0-DutyP-DutyN)/2.0-0.01,
            (1.0-DutyP-DutyN)/2.0+0.01,
            (1.0-DutyP-DutyN)/2.0,
            step=c.streamlit.duty_step_db,
            key="duty0"+m,
            help="D2=D4=(1-D1-D3)/2")  # Step outside the range to fix the variable

    if excitation in ['Sinusoidal', 'Triangular', 'Trapezoidal']:
        Outmax = st.sidebar.slider(
            f'Maximum Outlier Factor (%)',
            1,
            20,
            20,
            step=1,
            key="outlier" + m,
            help="Measures the similarity between a datapoint and their neighbours (in terms of B and f) based on local Steinmetz parameters")

    if excitation == 'Triangular':
        read_excitation = 'Trapezoidal'  # Triangular data read from Trapezoidal files
    else:
        read_excitation = excitation

    if excitation in ['Datasheet', 'Sinusoidal']:
        cycle_list = np.linspace(0, 1, 101)
        flux_list = np.add(np.multiply(np.sin(np.multiply(cycle_list, np.pi * 2)), Bavg), Bbias)

    if read_excitation == 'Datasheet':
        df = load_dataframe(material, read_excitation, Fmin, Fmax, Bmin, Bmax, None, None, None)
    if read_excitation == 'Sinusoidal':
        df = load_dataframe(material, read_excitation, Fmin, Fmax, Bmin, Bmax, None, None, Outmax)
    if read_excitation == 'Trapezoidal':
        df = load_dataframe(material, read_excitation, Fmin, Fmax, Bmin, Bmax, DutyP, DutyN, Outmax)

    col1, col2 = st.columns([3, 3])
    with col1:
        st.title(f"Core Loss Database {m}:")
        if excitation == 'Datasheet':
            header(material, excitation, Fmin, Fmax, Bmin, Bmax, None, Temp, None, None)
        if excitation == 'Sinusoidal':
            header(material, excitation, Fmin, Fmax, Bmin, Bmax, None, None, None, None)
        if excitation == 'Triangular':
            header(material, excitation, Fmin, Fmax, Bmin, Bmax, None, None, DutyP, None)
        if excitation == 'Trapezoidal':
            header(material, excitation, Fmin, Fmax, Bmin, Bmax, None, None, DutyP, DutyN)

        if df.empty:
            st.write("Warning: no data in range, please change the range")
        else:

            with st.expander("Measurement details"):
                metadata = load_metadata(material, read_excitation)
                st.write(metadata['info_date'])
                st.write(metadata['info_excitation'])
                if excitation in ['Sinusoidal', 'Triangular', 'Trapezoidal']:
                    st.write(metadata['info_core'])  # The datasheet is not associated with a specific core
            st.header(f"Download data:")
            file = df.to_csv().encode('utf-8')
            st.download_button(
                "Download CSV",
                file,
                material + "-" + excitation + ".csv",
                "text/csv",
                key=m,
                help='Download a CSV file containing the flux, frequency, duty cycle,'
                     'power loss and outlier factor for the depicted datapoints')

    with col2:
        if excitation in ['Triangular', 'Trapezoidal']:
            cycle_list = [0, DutyP, DutyP + Duty0, 1 - Duty0, 1]
            if DutyP > DutyN:
                BPplot = 1  # Bpk is proportional to the voltage, which is is proportional to (1-dp+dN) times the dp
                BNplot = -(-1 - DutyP + DutyN) * DutyN / ((1 - DutyP + DutyN) * DutyP)  # Proportional to (-1-dp+dN)*dn
            else:
                BNplot = 1  # Proportional to (-1-dP+dN)*dN
                BPplot = -(1 - DutyP + DutyN) * DutyP / ((-1 - DutyP + DutyN) * DutyN)  # Proportional to (1-dP+dN)*dP
            flux_list = [-BPplot, BPplot, BNplot, -BNplot, -BPplot]

        x_vector = np.multiply(cycle_list, 1e6 / Favg)  # In us
        flux_vector = np.add(np.multiply(flux_list, Bavg), Bbias)
        y_vector = np.multiply(flux_vector, 1e3)  # In mT
        waveform_visualization_db(
            st,
            x=x_vector,
            y=y_vector,
            title=f"Waveform visualization: <br> f={format(Favg / 1e3, '.0f')} kHz, B={format(Bavg * 1e3, '.0f')} mT")

    if df.empty or excitation == 'Datasheet':  # Second column not required
        col1, col2 = st.columns([5, 1])
    else:
        col1, col2 = st.columns([3, 3])

    if not df.empty:
        with col1:
            if c_axis == 'Flux Density':
                st.plotly_chart(scatter_plot(
                    df,
                    x='Frequency_kHz',
                    y='Power_Loss_kW/m3',
                    c='Flux_Density_mT'),
                    use_container_width=True,)
            elif c_axis == 'Frequency':
                st.plotly_chart(scatter_plot(
                    df,
                    x='Flux_Density_mT',
                    y='Power_Loss_kW/m3',
                    c='Frequency_kHz'),
                    use_container_width=True)
            else:
                st.plotly_chart(scatter_plot(
                    df,
                    x='Flux_Density_mT',
                    y='Frequency_kHz',
                    c='Power_Loss_kW/m3'),
                    use_container_width=True)

        if excitation in ['Sinusoidal', 'Triangular', 'Trapezoidal']:
            with col2:
                if c_axis == 'Flux Density':
                    st.plotly_chart(scatter_plot(
                        df,
                        x='Frequency_kHz',
                        y='Power_Loss_kW/m3',
                        c='Outlier_Factor'),
                        use_container_width=True)
                elif c_axis == 'Frequency':
                    st.plotly_chart(scatter_plot(
                        df,
                        x='Flux_Density_mT',
                        y='Power_Loss_kW/m3',
                        c='Outlier_Factor'),
                        use_container_width=True)
                else:
                    st.plotly_chart(scatter_plot(
                        df, x='Flux_Density_mT',
                        y='Frequency_kHz',
                        c='Outlier_Factor'),
                        use_container_width=True)

    st.sidebar.markdown("""---""")
    st.markdown("""---""")
