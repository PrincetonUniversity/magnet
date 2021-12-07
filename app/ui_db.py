import streamlit as st

from magnet import config as c
from magnet.constants import material_names, excitations, input_dir
from magnet.io import load_dataframe
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
    st.sidebar.header(f'Information for Material {m}')
    material = st.sidebar.selectbox(f'Material {m}:', material_names)
    excitation = st.sidebar.selectbox(f'Excitation {m}:', excitations[1:]) # Datasheet disabled until the new JSON files are availabe
    caxis = st.sidebar.selectbox(f'Select color-axis for Plotting {m}:', ['Flux Density', 'Frequency', 'Power Loss'])

    [Fmin_kHz, Fmax_kHz] = st.sidebar.slider(
        "Frequency Range (kHz)",
        c.streamlit.freq_min/1e3,
        c.streamlit.freq_max/1e3,
        (c.streamlit.freq_min/1e3, c.streamlit.freq_max/1e3),
        step=c.streamlit.freq_step/1e3)

    Fmin = Fmin_kHz*1e3
    Fmax = Fmax_kHz*1e3

    [Bmin_mT, Bmax_mT] = st.sidebar.slider(
        "AC Flux Density Amplitude Range (mT)",
        c.streamlit.flux_min*1e3,
        c.streamlit.flux_max*1e3,
        (c.streamlit.flux_min*1e3, c.streamlit.flux_max*1e3),
        step=c.streamlit.flux_step*1e3)

    Bmin = Bmin_mT/1e3
    Bmax = Bmax_mT/1e3

    read_excitation = excitation
    if excitation == 'Triangular':
        read_excitation = 'Trapezoidal'  # Triangular data read from Trapezoidal files

    if excitation in ('Datasheet', 'Sinusoidal'):
        st.title(f"Core Loss Database {m}:")
        header(material, excitation, Fmin, Fmax, Bmin, Bmax)
        df = load_dataframe(material, read_excitation, Fmin, Fmax, Bmin, Bmax)

    if excitation == 'Triangular':
        DutyP = st.sidebar.slider(f'Duty Ratio {m}', 0.1, 0.9, 0.5, step=0.1)
        DutyN = 1.0-DutyP  # For triangular excitation, there are no flat parts

        header(material, excitation, Fmin, Fmax, Bmin, Bmax, DutyP)
        df = load_dataframe(material, read_excitation, Fmin, Fmax, Bmin, Bmax, DutyP, DutyN)

    if excitation == 'Trapezoidal':
        DutyP = st.sidebar.slider(f'Duty Ratio (Rising) {m}', 0.1, 0.7, 0.4, step=0.1, help="D1")
        if DutyP == 0.1 or DutyP == 0.3 or DutyP == 0.5 or DutyP == 0.7:
            DutyNmin = 0.1
        elif DutyP == 0.2 or DutyP == 0.4 or DutyP == 0.6:
            DutyNmin = 0.2
        DutyNmax = 1.0-DutyP-0.2
        if DutyNmax <= DutyNmin+0.01:
            DutyN = st.sidebar.slider(f'Duty Ratio (Falling) {m} (Fixed)', DutyNmax-0.01, DutyNmax+0.01, DutyNmax, step=1.0, help="D3") # Step outside the range to fix the variable
        else:
            DutyN = st.sidebar.slider(f'Duty Ratio (Falling) {m}', DutyNmin, DutyNmax, DutyNmax, step=0.2, help="D3")
        Duty0 = (1.0-DutyP-DutyN)/2.0
        st.sidebar.slider(f'Duty Ratio (Flat) {m} (Fixed)', Duty0-0.01, Duty0+0.01, Duty0, step=1.0, help="D2=D4")

        header(material, excitation, Fmin, Fmax, Bmin, Bmax, DutyP, DutyN)
        # st.header("Note: D=0.2332 means **20% Up + 30% Flat + 30% Down + 20% Flat** from left to right")

        df = load_dataframe(material, read_excitation, Fmin, Fmax, Bmin, Bmax, DutyP, DutyN)

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

        file_txt = input_dir + material + "_" + read_excitation + "_Test_Info.txt"
        txt_info_file = open(file_txt)
        txt_info_lines = txt_info_file.readlines()
        st.write(txt_info_lines[0])  # Excitation information (first line of the txt file)
        st.write(txt_info_lines[1])  # Core information (second line of the txt file)
        st.write(txt_info_lines[-1])  # Date information (last line of the txt file)

        file = df.to_csv().encode('utf-8')
        st.download_button("Download CSV", file, material + "-" + excitation + ".csv", "text/csv", key=m)
        st.write("CSV Columns: [Index; Frequency (Hz); Flux Density (T); D1; D2; D3; D4; Outlier factor (%); Power Loss (W/m^3)]")

    st.sidebar.markdown("""---""")
    st.markdown("""---""")