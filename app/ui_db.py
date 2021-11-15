import streamlit as st

from magnet import config as c
from magnet.constants import material_names, excitations
from magnet.io import load_dataframe
from magnet.plots import power_loss_scatter_plot


def header(material, excitation, f_min, f_max, b_min, b_max, duty=None):
    s = f'{material}, {excitation}, f=[{f_min}~{f_max}] Hz, B=[{b_min}~{b_max}] mT, P: kW/m^3'
    if duty is not None:
        s += f', D={duty}'
    return st.subheader(s)


def ui_core_loss_db(m):
    st.sidebar.header(f'Information for Material {m}')
    material = st.sidebar.selectbox(f'Material {m}:', material_names)
    excitation = st.sidebar.selectbox(f'Excitation {m}:', excitations)
    xaxis = st.sidebar.selectbox(f'Select X-axis for Plotting {m}:', ['Flux Density', 'Frequency'])

    [Fmin, Fmax] = st.sidebar.slider(
        f'Frequency Range {m} (Hz)',
        c.streamlit.freq_min,
        c.streamlit.freq_max,
        (c.streamlit.freq_min, c.streamlit.freq_max),
        step=c.streamlit.freq_step
    )
    [Bmin, Bmax] = st.sidebar.slider(
        f'Flux Density Range {m} (mT)',
        c.streamlit.flux_min,
        c.streamlit.flux_max,
        (c.streamlit.flux_min, c.streamlit.flux_max),
        step=c.streamlit.flux_step
    )

    if excitation in ('Datasheet', 'Sinusoidal'):
        st.title(f"Core Loss Database {m}:")
        header(material, excitation, Fmin, Fmax, Bmin, Bmax)
        df = load_dataframe(material, excitation, Fmin,Fmax, Bmin, Bmax)

        if df.empty:
            st.write("Warning: No Data in Range")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(xaxis+' - Power Loss')
                if xaxis == 'Frequency':
                    st.plotly_chart(power_loss_scatter_plot(df, x='Frequency', color_prop='Flux_Density'), use_container_width=True)
                else:
                    st.plotly_chart(power_loss_scatter_plot(df, x='Flux_Density', color_prop='Frequency'), use_container_width=True)
            with col2:          # to be replaced with data driven error graphs
                st.subheader(xaxis+' - Estimated Error')    
                if xaxis == 'Frequency':
                    st.plotly_chart(power_loss_scatter_plot(df, x='Frequency', color_prop='Flux_Density'), use_container_width=True)
                else:
                    st.plotly_chart(power_loss_scatter_plot(df, x='Flux_Density', color_prop='Frequency'), use_container_width=True)
            file = df.to_csv().encode('utf-8')
            st.download_button("Download CSV",file, material+"-"+excitation+".csv","text/csv",key=m)
            st.write("CSV Column: [Index; Frequency (Hz); Flux Density (mT); Duty Ratio; Power Loss (kW/m^3)]")
            st.write("Data Info: Core Shape - ? ; Temperature - ? ; Method - ? ; Winding Turns - ?; Princeton Measured")

    if excitation == 'Triangle':
        Duty = st.sidebar.multiselect(f'Duty Ratio {m}', c.streamlit.duty_ratios_triangle, c.streamlit.duty_ratios_triangle)
        Margin = st.sidebar.slider(f'Duty Ratio Margin {m}', 0.0, 1.0, 0.01, step=0.01)

        header(material, excitation, Fmin, Fmax, Bmin, Bmax, Duty)
        df = load_dataframe(material, excitation, Fmin,Fmax, Bmin, Bmax, Duty, Margin)

        if df.empty:
            st.write("Warning: No Data in Range")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(xaxis+' - Power Loss')
                if xaxis == 'Frequency':
                    st.plotly_chart(power_loss_scatter_plot(df, x='Frequency', color_prop='Duty_Ratio'), use_container_width=True)
                else:
                    st.plotly_chart(power_loss_scatter_plot(df, x='Flux_Density', color_prop='Duty_Ratio'), use_container_width=True)
            with col2:
                st.subheader(xaxis+' - Estimated Error')    
                if xaxis == 'Frequency':
                    st.plotly_chart(power_loss_scatter_plot(df, x='Frequency', color_prop='Duty_Ratio'), use_container_width=True)
                else:
                    st.plotly_chart(power_loss_scatter_plot(df, x='Flux_Density', color_prop='Duty_Ratio'), use_container_width=True)
            file = df.to_csv().encode('utf-8')
            st.download_button("Download CSV",file,material+"-"+excitation+".csv","text/csv",key=m)
            st.write("CSV Column: [Index; Frequency (Hz); Flux Density (mT); Duty Ratio; Power Loss (kW/m^3)]")
            st.write("Data Info: Core Shape - ? ; Temperature - ? ; Method - ? ; Winding Turns - ?; Princeton Measured")

    if excitation == 'Trapezoidal':
        Duty = st.sidebar.multiselect(f'Duty Ratio {m}', c.streamlit.duty_ratios_trapezoid, c.streamlit.duty_ratios_trapezoid)
        Margin = st.sidebar.slider(f'Duty Ratio Margin {m}', 0.0, 1.0, 0.01, step=0.01)

        header(material, excitation, Fmin, Fmax, Bmin, Bmax, Duty)
        st.header("Note: D=0.2332 means **20% Up + 30% Flat + 30% Down + 20% Flat** from left to right")

        df = load_dataframe(material, excitation, Fmin,Fmax, Bmin, Bmax, Duty, Margin)

        if df.empty:
            st.write("Warning: No Data in Range")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(xaxis+' - Power Loss')
                if xaxis == 'Frequency':
                    st.plotly_chart(power_loss_scatter_plot(df, x='Frequency', color_prop='Duty_Ratio'), use_container_width=True)
                else:
                    st.plotly_chart(power_loss_scatter_plot(df, x='Flux_Density', color_prop='Duty_Ratio'), use_container_width=True)
            with col2:
                st.subheader(xaxis+' - Estimated Error')    
                if xaxis == 'Frequency':
                    st.plotly_chart(power_loss_scatter_plot(df, x='Frequency', color_prop='Duty_Ratio'), use_container_width=True)
                else:
                    st.plotly_chart(power_loss_scatter_plot(df, x='Flux_Density', color_prop='Duty_Ratio'), use_container_width=True)
            file = df.to_csv().encode('utf-8')
            st.download_button("Download CSV",file,material+"-"+excitation+".csv","text/csv",key=m)
            st.write("CSV Column: [Index; Frequency (Hz); Flux Density (mT); Duty Ratio; Power Loss (kW/m^3)]")
            st.write("Data Info: Core Shape - ? ; Temperature - ? ; Method - ? ; Winding Turns - ?; Princeton Measured")

    st.sidebar.markdown("""---""")
    st.markdown("""---""")