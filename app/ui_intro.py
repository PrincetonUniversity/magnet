import os.path
from PIL import Image
import pandas as pd
import streamlit as st
from magnet.constants import material_names, materials, materials_extra, material_manufacturers, \
    material_applications, material_core_tested
from magnet.io import load_dataframe

STREAMLIT_ROOT = os.path.dirname(__file__)


def ui_intro(m):

    st.header('Introduction of MagNet')
    st.write("""
        MagNet is a large-scale dataset designed to enable researchers modeling magnetic core loss using machine learning to accelerate the design process of power electronics.
        The dataset contains a large amount of voltage and current data of different magnetic components with different shapes of waveforms and different properties measured in the real world.
        Researchers may use these data as pairs of excitations and responses to build up dynamic magnetic models or calculate the core loss to derive static models.
        MagNet is continuously being maintained and updated with new data.
        
        With this webpage, you can visualize and download the collected data for different magnetic materials and excitations or calculate core losses for your specific design conditions using Neural Networks integrated into the webpage.
        For more information, the FAQ section details how the data is captured and processed and how losses are calculated.
        
        Please select one of these functions on the left menu to start exploring the webpage.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('Core Loss Database')
        st.write("""
            In this section, the core loss data can be visualized.
            
            Select the desired material and excitation on the left to visualize the core loss as a function of the frequency and flux density.
            The desired range for the plot and specific conditions such as temperature, DC bias, or duty cycle can also be selected from the left sliders.
            On the right, you can see the shape of the waveform you have selected.
            
            For each case, a plot will represent the volumetric loss, frequency, and flux density, where the variable in the colorbar can be selected on the left.
            When selecting Datasheet excitation, the data provided is the interpolation of the values provided in the material datasheet from the manufacturer.
            
            Finally, for measured data, a plot shows the Outlier Factor, which provides information on the quality of the data; for more information please check the FAQ section.
            
            Additionally, all the data points in the selected range can be easily downloaded as a .csv file by clicking the download button.
            Click on "Measurement details" to see the core and specific conditions for the test.
            
            Currently we are working on adding measurements at different temperatures and DC bias.
        """)
    with col2:
        st.subheader('Core Loss Analysis')
        st.write("""
            In this section, volumetric core losses are calculated for any desired operation point.
            
            The material and operation point can be configured with the menu on the left.
            On the right, the voltage and flux as a function of time are depicted for the selected conditions. 
            
            For the selected operation point, losses are calculated using two methods:
            1) improved Generalized Steinmetz Equations (iGSE).
            2) Machine Learning (ML) models, which is a Neural Networks trained with the measured database are deployed on the webpage.
            Further information on the iGSE and ML models can be found in the FAQ section.
            Additionally, the interpolated values for the measurement and datasheet are provided when available for comparison purposes.
            
            For the selected material and conditions, additional plots show how losses change when sweeping one of the variables (such as frequency, flux, or duty cycle) and keeping the others fixed.
            The results include both the iGSE and ML methods.
            
            Besides the calculation for conventional excitations, we are working on NN models for arbitrary waveforms.
            In the future, it will also be possible to obtain the losses for simulated waveforms using a PLECs toolbox.
        """)
    with col3:
        st.subheader('Download Waveform Data')
        st.write("""
            In this section, the measurement and post-processed data are available for download.
            
            For each material and excitation, there are two .zip files available:
            
            1) The raw voltage and current data from the oscilloscope of each measured waveform are provided for download.
            Each data point contains 2.000 samples, the first 20 us out of the 100 us of the total sample, which ensures at least a switching cycle information while saving space.
            The voltage and current are provided as two separated .csv files. 
            An additional .txt file includes the information regarding how the test has been performed.
            
            2) The B and H waveforms for a single switching cycle.
            This information is post-processed from the raw voltage and current waveform as detailed in the FAQ section.
            Again, two .csv files are generated, one for B and one for H, and another .csv file contains the information of the frequency of each data point.
            A .txt with information on the test is also incldued.
            
            These files are intended for researchers to build their own core loss models.
        """)

    st.markdown("""---""")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header('Webpage Status')
        st.write("")
        st.write("")
        n_sine = 0
        n_trap = 0
        for material in material_names:
            n_sine = n_sine + len(load_dataframe(material, 'Sinusoidal'))
            n_trap = n_trap + len(load_dataframe(material, 'Trapezoidal'))
        st.subheader(f'Total number of data points: {n_sine + n_trap}')
        st.write(f'{n_sine} Sinusoidal points and {n_trap} Triangular-Trapezoidal points.')
        st.write("")
        st.write("")
        st.subheader(f'Number of materials added: {len(material_names)}')
        st.write(f'Tested for 25 C and no DC bias so far.')
    with col2:
        st.header('How to Cite')
        st.write("""
            If you used MagNet, please cite us with the following:
            
            [1] H. Li, D. Serrano, T. Guillod, E. Dogariu, A. Nadler, S. Wang, M. Luo, V. Bansal, Y. Chen, C. R. Sullivan, and M. Chen, 
            "MagNet: an Open-Source Database for Data-Driven Magnetic Core Loss Modeling," 
            IEEE Applied Power Electronics Conference (APEC), Houston, 2022.
            
            [2] E. Dogariu, H. Li, D. Serrano, S. Wang, M. Luo and M. Chen, 
            "Transfer Learning Methods for Magnetic Core Loss Modeling,” 
            IEEE Workshop on Control and Modeling of Power Electronics (COMPEL), Cartagena de Indias, Colombia, 2021.
            
            [3] H. Li, S. R. Lee, M. Luo, C. R. Sullivan, Y. Chen and M. Chen, 
            "MagNet: A Machine Learning Framework for Magnetic Core Loss Modeling,” 
            IEEE Workshop on Control and Modeling of Power Electronics (COMPEL), Aalborg, Denmark, 2020.
        """)

    df = pd.DataFrame({'Manufacturer': material_manufacturers})
    df['Material'] = materials.keys()
    df['Applications'] = pd.DataFrame({'Applications': material_applications})
    df_extra = pd.DataFrame(materials_extra)
    df['mu_i_0'] = df_extra.iloc[0]
    df['f_min'] = df_extra.iloc[1]
    df['f_max'] = df_extra.iloc[2]
    df_params = pd.DataFrame(materials)
    df['k_i*'] = df_params.iloc[0]
    df['alpha*'] = df_params.iloc[1]
    df['beta*'] = df_params.iloc[2]
    df['Tested Core'] = pd.DataFrame({'Tested Core': material_core_tested})
    # Hide the index column
    hide_table_row_index = """
                <style>
                tbody th {display:none}
                .blank {display:none}
                </style>
                """  # CSS to inject contained in a string
    st.markdown(hide_table_row_index, unsafe_allow_html=True)  # Inject CSS with Markdown
    st.table(df)

    st.write(f'*iGSE parameters obtained from the sinusoidal measurements at 25 C and data '
             f'between 50 kHz and 500 kHz and 10 mT and 300 mT; '
             f'with Pv, f, and B in W/m^3, Hz and T respectively')

    st.markdown("""---""")
