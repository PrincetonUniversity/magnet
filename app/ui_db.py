import streamlit as st

from magnet import config as c
from magnet.constants import material_names, excitations
from magnet.io import load_dataframe, load_metadata
from magnet.plots import scatter_plot


def header(material, excitation, f_min, f_max, b_min, b_max, DutyP=None, DutyN=None):
    s = f'{material}, {excitation}, f=[{f_min/1000}~{f_max/1000}] kHz, B=[{b_min*1000}~{b_max*1000}] mT'
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
    st.sidebar.header(f'Information for Material A')
    material = st.sidebar.selectbox(f'Material:', material_names, key="material"+m)
    excitation = st.sidebar.selectbox(f'Excitation:', excitations, key="excitation"+m)
    c_axis = st.sidebar.selectbox(f'Select color-axis for Plotting:', ['Flux Density', 'Frequency', 'Power Loss'], key="c_axis"+m)

    [Fmin_kHz, Fmax_kHz] = st.sidebar.slider(
        "Frequency Range (kHz)",
        c.streamlit.freq_min/1e3,
        c.streamlit.freq_max/1e3,
        (c.streamlit.freq_min/1e3, c.streamlit.freq_max/1e3),
        step=c.streamlit.freq_step/1e3,
        key="frequency"+m)

    Fmin = Fmin_kHz*1e3
    Fmax = Fmax_kHz*1e3

    [Bmin_mT, Bmax_mT] = st.sidebar.slider(
        "AC Flux Density Amplitude Range (mT)",
        c.streamlit.flux_min*1e3,
        c.streamlit.flux_max*1e3,
        (c.streamlit.flux_min*1e3, c.streamlit.flux_max*1e3),
        step=c.streamlit.flux_step*1e3,
        key="flux"+m)

    Bmin = Bmin_mT/1e3
    Bmax = Bmax_mT/1e3

    # TODO: add the temeperature filter for the datasheet plot
    # if excitation == 'Datasheet':
    # Temperature = st.sidebar.slider(f'Temperature (C) (coming soon)', 0, 120, 25, step=1000, key="temp"+m)

    if excitation == 'Triangular':
        DutyP = st.sidebar.slider(f'Duty Ratio', 0.1, 0.9, 0.5, step=0.1, key="duty"+m)
        DutyN = 1.0-DutyP  # For triangular excitation, there are no flat parts

    if excitation == 'Trapezoidal':
        DutyP = st.sidebar.slider(f'Duty Ratio (Rising)', 0.1, 0.7, 0.4, step=0.1, key="dutyP"+m, help="D1")
        if DutyP == 0.1 or DutyP == 0.3 or DutyP == 0.5 or DutyP == 0.7:
            DutyNmin = 0.1
        elif DutyP == 0.2 or DutyP == 0.4 or DutyP == 0.6:
            DutyNmin = 0.2
        DutyNmax = 1.0-DutyP-0.2
        if DutyNmax <= DutyNmin+0.01:
            DutyN = st.sidebar.slider(f'Duty Ratio (Falling) Fixed', DutyNmax-0.01, DutyNmax+0.01, DutyNmax, step=1.0,
                                      key="dutyN"+m, help="D3, fixed by D1")  # Step outside the range to fix the variable
        else:
            DutyN = st.sidebar.slider(f'Duty Ratio (Falling)', DutyNmin, DutyNmax, DutyNmax, step=0.2,
                                      key="dutyN"+m, help="D3, maximum imposed by D1")
        Duty0 = (1.0-DutyP-DutyN)/2.0
        st.sidebar.slider(f'Duty Ratio (Flat) Fixed', Duty0-0.01, Duty0+0.01, Duty0, step=1.0,
                          key="duty0"+m, help="D2=D4=(1-D1-D3)/2")  # Step outside the range to fix the variable

    if excitation == 'Triangular':
        read_excitation = 'Trapezoidal'  # Triangular data read from Trapezoidal files
    else:
        read_excitation = excitation

    st.title(f"Core Loss Database {m}:")

    if excitation in ('Datasheet', 'Sinusoidal'):
        header(material, excitation, Fmin, Fmax, Bmin, Bmax)
        df = load_dataframe(material, read_excitation, Fmin, Fmax, Bmin, Bmax)

    if excitation == 'Triangular':
        header(material, excitation, Fmin, Fmax, Bmin, Bmax, DutyP)
        df = load_dataframe(material, read_excitation, Fmin, Fmax, Bmin, Bmax, DutyP, DutyN)

    if excitation == 'Trapezoidal':
        header(material, excitation, Fmin, Fmax, Bmin, Bmax, DutyP, DutyN)
        df = load_dataframe(material, read_excitation, Fmin, Fmax, Bmin, Bmax, DutyP, DutyN)

    if df.empty:
        st.write("Warning: No Data in Range")
    else:
        if excitation == 'Datasheet':
            if c_axis == 'Flux_Density':
                st.plotly_chart(scatter_plot(df, x='Frequency_kHz', y='Power_Loss_kW/m3', c='Flux_Density_mT'), use_container_width=True)
            elif c_axis == 'Frequency':
                st.plotly_chart(scatter_plot(df, x='Flux_Density_mT', y='Power_Loss_kW/m3', c='Frequency_kHz'), use_container_width=True)
            else:
                st.plotly_chart(scatter_plot(df, x='Flux_Density_mT', y='Frequency_kHz', c='Power_Loss_kW/m3'), use_container_width=True)
        else:  # For Sinusoidal, Triangular and Trapezoidal, the outlier function is also plotted
            col1, col2 = st.columns(2)
            with col1:
                if c_axis == 'Flux_Density':
                    st.plotly_chart(scatter_plot(df, x='Frequency_kHz', y='Power_Loss_kW/m3', c='Flux_Density_mT'), use_container_width=True)
                elif c_axis == 'Frequency':
                    st.plotly_chart(scatter_plot(df, x='Flux_Density_mT', y='Power_Loss_kW/m3', c='Frequency_kHz'), use_container_width=True)
                else:
                    st.plotly_chart(scatter_plot(df, x='Flux_Density_mT', y='Frequency_kHz', c='Power_Loss_kW/m3'), use_container_width=True)
            with col2:
                st.plotly_chart(scatter_plot(df, x='Flux_Density_mT', y='Frequency_kHz', c='Outlier_Factor'), use_container_width=True)

        metadata = load_metadata(material, read_excitation)
        st.write(metadata['info_date'])
        st.write(metadata['info_excitation'])
        if excitation != 'Datasheet':
            st.write(metadata['info_core'])  # The datasheet is not associated with a specific core

        file = df.to_csv().encode('utf-8')
        st.download_button("Download CSV", file, material + "-" + excitation + ".csv", "text/csv", key=m)

    st.sidebar.markdown("""---""")
    st.markdown("""---""")
