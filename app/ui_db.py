import streamlit as st

from magnet import config as c
from magnet.constants import material_names, excitations
from magnet.io import load_dataframe
from magnet.plots import scatter_plot


def header(material, excitation, f_min, f_max, b_min, b_max, duty_1=None, duty_3=None):
    s = f'{material}, {excitation}, f=[{f_min/1000}~{f_max/1000}] kHz, B=[{b_min*1000}~{b_max*1000}] mT, P: kW/m^3'
    if duty_1 is not None and duty_3 is None:
        s += f', D={duty_1}'
    if duty_3 is not None:
        duty_extra = (1.0-duty_1-duty_3)/2
        s += f', D1={duty_1}, D3={duty_extra}, D3={duty_3}, D3={duty_extra}'
    return st.subheader(s)

  
def ui_core_loss_dbs(n=1):
    for i in range(int(n)):
        ui_core_loss_db(chr(ord('A') + i))

        
def ui_core_loss_db(m):
    st.sidebar.header(f'Information for Material {m}')
    material = st.sidebar.selectbox(f'Material {m}:', material_names)
    excitation = st.sidebar.selectbox(f'Excitation {m}:', excitations)
    caxis = st.sidebar.selectbox(f'Select color-axis for Plotting {m}:', ['Flux Density', 'Frequency', 'Power Loss'])

    [Fmin_kHz, Fmax_kHz] = st.sidebar.slider(
        "Frequency Range (kHz)",
        c.streamlit.freq_min/1e3,
        c.streamlit.freq_max/1e3,
        (c.streamlit.freq_min/1e3, c.streamlit.freq_max/1e3),
        step=c.streamlit.freq_step/1e3)

    Fmin = Fmin_kHz * 1e3
    Fmax = Fmax_kHz * 1e3

    [Bmin_mT, Bmax_mT] = st.sidebar.slider(
        "Flux Density Range (mT)",
        c.streamlit.flux_min*1e3,
        c.streamlit.flux_max*1e3,
        (c.streamlit.flux_min*1e3, c.streamlit.flux_max*1e3),
        step=c.streamlit.flux_step*1e3)

    Bmin = Bmin_mT / 1e3
    Bmax = Bmax_mT / 1e3

    if excitation in ('Datasheet', 'Sinusoidal'):
        st.title(f"Core Loss Database {m}:")
        header(material, excitation, Fmin, Fmax, Bmin, Bmax)
        df = load_dataframe(material, excitation, Fmin, Fmax, Bmin, Bmax)

    if excitation == 'Triangular':
        # Duty = st.sidebar.multiselect(f'Duty Ratio {m}', c.streamlit.duty_ratios_triangle, c.streamlit.duty_ratios_triangle)
        duty_1 = st.sidebar.slider(f'Duty Cycle {m}', 0.1, 0.9, 0.5, step=0.1)
        duty_3 = 1.0-duty_1 # For triangular excitation, there are no flat parts
        # Margin = st.sidebar.slider(f'Duty Ratio Margin {m}', 0.0, 1.0, 0.01, step=0.01)

        header(material, excitation, Fmin, Fmax, Bmin, Bmax, duty_1)
        df = load_dataframe(material, 'Trapezoidal', Fmin, Fmax, Bmin, Bmax, duty_1, duty_3)

    if excitation == 'Trapezoidal':
        # Duty = st.sidebar.multiselect(f'Duty Ratio {m}', c.streamlit.duty_ratios_trapezoid, c.streamlit.duty_ratios_trapezoid)
        # Margin = st.sidebar.slider(f'Duty Ratio Margin {m}', 0.0, 1.0, 0.01, step=0.01)
        duty_1 = st.sidebar.slider(f'Rising Duty Cycle (D1) {m}', 0.1, 0.7, 0.4, step=0.1)
        if duty_1 == 0.1 or duty_1 == 0.3 or duty_1 == 0.5 or duty_1 == 0.7:
            duty_3min = 0.1
        elif duty_1 == 0.2 or duty_1 == 0.4 or duty_1 == 0.6:
            duty_3min = 0.2
        duty_3max = 1 - duty_1-0.2
        if duty_3max <= duty_3min+0.01:
            duty_3 = duty_3max
            st.sidebar.write(f'Falling Duty Cycle (D3) {m}', duty_3,)
        else:
            duty_3 = st.sidebar.slider(f'Falling Duty Cycle (D3) {m}', duty_3min, duty_3max, duty_3max, step=0.2)
        duty_2_4 = (1-duty_1-duty_3)/2
        st.sidebar.write(f' Duty Cycles D2 and D4 {m}', duty_2_4)

        header(material, excitation, Fmin, Fmax, Bmin, Bmax, duty_1, duty_3)
        # st.header("Note: D=0.2332 means **20% Up + 30% Flat + 30% Down + 20% Flat** from left to right")

        df = load_dataframe(material, excitation, Fmin, Fmax, Bmin, Bmax, duty_1, duty_3)

    if df.empty:
        st.write("Warning: No Data in Range")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if caxis == 'Frequency':
                st.subheader('Frequency - Power Loss')
                st.plotly_chart(scatter_plot(df, x='Frequency', y='Power_Loss', c='Flux_Density'), use_container_width=True)
            elif caxis == 'Flux Density':
                st.subheader('Flux Density - Power Loss')
                st.plotly_chart(scatter_plot(df, x='Flux_Density', y='Power_Loss', c='Frequency'), use_container_width=True)
            else:
                st.subheader('Flux Density - Frequency')
                st.plotly_chart(scatter_plot(df, x='Flux_Density', y='Frequency', c='Power_Loss'), use_container_width=True)
        with col2:
            st.subheader('Outlier Factor')
            st.plotly_chart(scatter_plot(df, x='Flux_Density', y='Frequency', c='Outlier_Factor'), use_container_width=True)

        file = df.to_csv().encode('utf-8')
        st.download_button("Download CSV", file, material+"-"+excitation+".csv", "text/csv", key=m)
        st.write("CSV Columns: [Index; Frequency (Hz); Flux Density (T); D1; D2; D3; D4; Outlier factor (%); Power Loss (W/m^3)]")
        st.write("Data Info: Core Shape - ? ; Temperature - ? ; Method - ? ; Winding Turns - ?; Princeton Measured")

    st.sidebar.markdown("""---""")
    st.markdown("""---""")