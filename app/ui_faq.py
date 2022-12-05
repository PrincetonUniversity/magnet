import os.path
from PIL import Image
import streamlit as st
import pandas as pd
from magnet.constants import material_list, material_manufacturers, material_applications, material_extra, \
    material_core_tested, material_steinmetz_param
from magnet.io import load_dataframe

STREAMLIT_ROOT = os.path.dirname(__file__)


def ui_faq(m):

    st.title('MagNet AI Overview')
    st.caption('MagNet is a large-scale dataset designed to enable researchers modeling power magnetics with real measurement data. MagNet AI is a pre-trained smart agent that can predict the behavior of power magnetics with neural networks. The dataset contains a large amount of voltage and current data (B and H) of different magnetic components with different shapes of waveforms and different properties measured in the real world. Researchers may use these data as pairs of excitations and responses to build up dynamic magnetic models or calculate the core loss in design. MagNet is continuously being maintained and updated with new data. With this webpage, you can visualize and download the collected data for different magnetic materials and excitations or calculate core losses for your specific design conditions using Neural Networks integrated into the webpage. Please select one of these functions on the left menu to start exploring the webpage.')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader('MagNet Database')
        st.caption('In this section, the MagNet database can be visualized in terms of core losses. Select the desired material and excitation to visualize the core loss as a function of the frequency and flux density. The desired range for the plot and specific conditions such as temperature, DC bias, or duty cycle can also be selected. You can also see the shape of the waveform you have selected. For each case, a plot will represent the volumetric loss, frequency, and flux density, where the variable in the colorbar can be selected. Additionally, all the data points in the selected range can be easily downloaded as a .csv file by clicking the download button. Click on "Measurement details" to see the core and specific conditions for the test.')
    with col2:
        st.subheader('MagNet Smartsheet')
        st.caption('In this section, volumetric core losses are calculated using Machine Learning for any desired operation point. The material and operation point can be configured with the menu and the voltage and flux as a function of time are depicted for the selected conditions. For the selected operation point, losses are calculated using Machine Learning (ML) models, which are Neural Networks trained with the measured database are deployed on the webpage. Further information models can be found in the FAQ section below. For the selected material and conditions, additional plots show how losses change when sweeping one of the variables (such as frequency, flux, or duty cycle) and keeping the others fixed. Finally, Machine Learning can also be used to compute the core loss of any arbitrary waveform.')
    with col3:
        st.subheader('MagNet Simulation')
        st.caption('This PLECs toolbox allows you to simulate conventional power converters with the selected material and core shape to obtain the desired waveforms. The waveforms are then used to compute the core losses using iGSE and Machine Learning. We are working on including the DC bias into the calculations.')
    with col4:
        st.subheader('MagNet Download')
        st.caption('In this section, the measurement and post-processed data are available for download. For each material and excitation, there are two .zip files available: 1) The raw voltage and current data from the oscilloscope of each measured waveform are provided for download. The voltage and current are provided as two separated .csv files. Each data point contains 10.000 samples for a total sampling time of 80 us. 2) The B and H waveforms for a single switching cycle. This information is post-processed from the raw voltage and current waveform as detailed in the FAQ section below. Two .csv files are generated, one for B and one for H with 128 samples at variable sample time to save a single switching cycle.')

    st.markdown("""---""")

    st.header('MagNet Status')
    st.write("")
    st.write("Materials added so far")
    df = pd.DataFrame({'Manufacturer': material_manufacturers})
    df['Material'] = material_steinmetz_param.keys()
    df['Applications'] = pd.DataFrame({'Applications': material_applications})
    df_extra = pd.DataFrame(material_extra)
    df['mu_i_r'] = df_extra.iloc[0]
    df['f_min [Hz]'] = df_extra.iloc[1]
    df['f_max [Hz]'] = df_extra.iloc[2]
    df_params = pd.DataFrame(material_steinmetz_param)
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

    st.write(f'*iGSE parameters obtained from sinusoidal measurements at 25 C without bias using a least-squares curve fitting method for log10(Pv); with Pv, f, and B in W/m^3, Hz and T respectively.')

    st.markdown("""---""")

    st.title('Frequently Asked Questions')
    st.write('')
    st.write("This section includes some explanations and limitations regarding how data is captured and processed, and how the implemented neural network works.")
    st.write('If you have any other questions, please let us know so we can add them to the list. Please use the "contact us" form on the left.')
    st.write('')
    st.subheader('Measurements:')
    with st.expander("1. How is the data measured?"):
        st.write("The magnetic core loss measurement is supported by an automated data acquisition system as shown in the figures. The voltage and current are directly measured using the two-winding method (also referred to as the V–I method or the voltamperometric method), which has high simplicity and flexibility of operation for fast measurements. Since the voltage is measured across the secondary winding, winding losses are excluded from the measurements.")
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'system-overview.jpg')), width=500)
        st.write("The automated data acquisition system consists of a power stage/power amplifier to generate the desired excitations, a magnetic core as the device-under-test, a wide-band coaxial shunt for current measurement, a passive probe for voltage measurement, an oscilloscope for data acquisition, a DC injection circuitry, and an oil bath with a heater for temperature control.")
        st.write("The hardware system is controlled and coordinated by a Python-based program on the host PC, which enables fully automated equipment setting and measurement under pre-programmed excitations. Please refer to [this paper](https://doi.org/10.36227/techrxiv.21340998.v2) for more details about the data acquisition system.")
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'system-photo.jpg')), width=500)
        st.warning("The main limitation of the voltamperometric method is the impact of delays between the voltage and current measurement (phase discrepancy) for the calculation of losses. This is especially concerning at high frequency and high quality factor materials. For a complete discussion on this issue please check [this paper](https://ieeexplore.ieee.org/abstract/document/7479554). Further details and limitations of this method can be found in [this paper](https://ieeexplore.ieee.org/document/8403009). To ensure that the error is small, the maximum frequency of the tests is limited to 500 kHz. The error analysis for this setup can be found in [this paper](https://doi.org/10.36227/techrxiv.21340989.v2).")
    with st.expander("2. How are sinusoidal excitations generated?"):
        st.write("A signal generator connected to a power amplifier (Amplifier Research 25A250AM6) is used. The load for the power amplifier is the series connection of a blocking capacitor and the core under test. The computer is used to apply the desired frequency and amplitude to the signal generator in each test")
        st.warning("Due to the operation of the power amplifier, distorted voltage waveforms because of the nonlinear behavior of the core-under-test or its low impedance are discarded (THD>5%).")
    with st.expander("3. How are triangular and trapezoidal excitations generated?"):
        st.write("The desired waveform is applied using a T-type inverter where the load is the core under test and a blocking capacitor. This T-Type inverter is supplied by two power supplies controlled by the computer. The voltage of the power supply sets the amplitude of the waveform. To obtain the different shapes, the PWM gate signals are controlled by a DSP through the computer. With this inverter, the load can be connected to the positive power supply, the negative power supply, or to ground, so the amplitude, frequency, and shape of the output waveform can be controlled. For these types of waveforms, four duty cycles are defined, d1 to d4, corresponding to different parts of the switching cycle. Please note that d2=d4 in all the tests performed so far. In the following figure, examples of different waveforms and their duty cycles are shown.")
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'trapezoidal-waveforms.jpg')), width=500)
        st.warning("Because of the dead time, soft and hard-switching depending on the conditions, and the shunt resistor and other parasitic resistances and inductances, the voltage waveforms applied are not ideal. Moreover, the DSP used to control the gate signals has a resolution of 10 ns and a minimum dead time is required, currently set to 70 ns. These parameters set a limit on the accuracy of the duty cycle and frequency, becoming more significant as the frequency increases.")
        st.error("Since GaN switches are employed, the dv/dt and ringing during the transitions between different voltage levels is high. These switching transitions impact the data. Capacitances (either as a part of the core under test or parasitic) generate dips in the current waveform during the transition affecting the shape of B-H loopa and measured losses. The bandwidth in the oscilloscope is set to 20 MHz for the measurements to limit the impact of ringing. Please note that these effects are intrinsic to power converters and are present in the real operation of power transformers and inductors.")
        st.write("The voltage and current measurements are available for download for users interested in the nonidealities of the excitation.")
    with st.expander("4. How is the DC bias applied?"):
        st.write("With this setup, the bias can be controlled by adding a constant current in the primary side. This current is added using a voltage supply using in current mode control connected between the DC blocking capacitor and the device under test. Additionally, a mirror transformer and a series inductor are added to minimize unwanted current ripples. Details will be available in a future paper (APEC 2023).")
        st.warning("The bias is applied in the current, leading to a dc value in the H field rather the B field. For ungapped cores, the relation between Bdc and Hdc is not unique. Bdc is not included in the data as it is not measured. Please refer to [this paper](https://ieeexplore.ieee.org/document/9383254) for further discussion.")
    with st.expander("5. How is the temperature controlled?"):
        st.write("The device under test is placed in an oil bath. The oil is continuously moved using a magnetic stirrer to improve heat dissipation. To maintain the oil temperature constant, it is placed inside a water tank. The temperature of the water is controlled using a water heater to set the desired temperature for the tests.")
        st.warning("The temperature of the core is not measured, the temperature reported in this database is the temperature at which the water tank is set.")
        st.error("The temperature of the core can still rise if the average power loss during testing is high. The core temperature before and after testing showed an increase increased by 5 C for the N87 50% duty cycle triangular excitation. Moreover, the oil temperature is about 2 C below the set temperature when set to 90 C.")
    with st.expander("6. How are the cores under test designed?"):
        st.write("The cores of each material are selected based on availability and how well they match the range of measurement of the system. The effective area, effective length, and number of turns are selected to maximize the amount of data collected. As a result, no standard core shape or size is used. Litz wire is used for the primary winding, to minimize high frequency resistance, while solid wire is used for the winding for voltage measurements.")
        st.warning("B, H, and losses are calculated from the measured voltage and currents and the effective dimensions in the datasheet. Unfortunately, the physical dimensions vary from core to core, and the variations of the dimensions with respect to the specified values are translated into an error in the reported B, H, and losses. This is a common problem that designers must face and leads to variations in inductance and core loss between samples.")
        st.error("The data provided in this webpage and used to train the neural networks are obtained from a single set of measurements. However, core-to-core variation can impact the measurements. Refer to [this paper](https://ieeexplore.ieee.org/document/9829998) for further discussion.")
    with st.expander("7. What is the range of the data collected?"):
        st.write("Data is measured for different waveform shapes amplitude, frequency, DC bias, and temperature. For the waveform shape, the duty cycle is swept in 10% steps, from 10% to 90% for triangular waveforms, and changing d2 and d4 from 10% to 40% for trapezoidal waveforms. The data is collected by setting the water tank temperature at 25 C, 50 C, 70 C, and 90 C. The DC bias is iterated in 15 A/m steps from 0 A/m to saturation. The frequency and flux density are acquired in a logarithmic range at 10 or 20 points per decade depending on the material. The data collection is limited by the data acquisition system employed and the specific limitations of each material. The figure shows the measurement range and limitations in terms of amplitude, frequency, and DC bias.")
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'measurement-range.jpg')), width=350)

    st.subheader('Data Processing:')
    with st.expander("1. What information is processed?"):
        st.write("The voltage and current measurements and the time stamps from each data point measured using the oscilloscope are saved (an example of those waveforms is shown in the figure below), together with the information regarding the operating conditions for each test and the core under test. These files are processed using Matlab. First, the raw measurements are downsampled from 100,000 samples to 10,000 samples using a box car averaging. This process barely penalizes the quality of the data while reducing the size of the database by a factor of 10. The data available for download is this downsamples voltage and current measurements, with the sampling time instead of a vector for the time stamps.")
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'volt-curr.jpg')), width=500)
    with st.expander("2. How is the frequency identified?"):
        st.write("Frequency plays a critical role on all other algorithms, so using the commanded frequency is not good enough. To obtain the frequency, a fast Fourier transform is not accurate enough due to the low frequency resolution. The power spectral density using Welch method is used instead to obtain the frequency instead, searching nearby the commanded frequency with a resolution of 10 Hz.")
    with st.expander("3. How are the single-cycle waveforms generated?"):
        st.write("To plot the B-H loop and train the neural networks, a single switching cycle for the B and H waveforms is needed. To obtain it, first, a single switching cycle for the measured voltage and current waveforms is obtained. The raw voltage and current waveform are split into the different switching cycles based on the calculated switching frequency, then, for each switching cycle, the waveform is interpolated into 1024 samples. Finally, the 1024 samples for each cycle are averaged into a single switching cycle waveform. The following simplified figure clarifies the process for obtaining a single voltage waveform; the same process is applied to the current waveform.")
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'single-cycle.jpg')), width=500)
        st.warning("Post-processing the data in this way has some limitations. If steady-state conditions are not reached or the frequency of the waveform is not accurately identified, some unwanted averaging effect will appear in the single-cycle waveform, creating artifacts that can affect the B-H loops and losses. Nevertheless, to ensure the quality of the data, data points where the single-cycle waveform from one switching cycle (before averaging) differs too much from the resulting average waveform are discarded. More information will be included in a future paper (APEC 2023).")
    with st.expander("4. How do we get B and H?"):
        st.write("The instantaneous flux density is calculated as the integral of the single-cycle voltage waveform divided by the number of turns of the secondary winding used to measure the voltage, and also divided by the effective area specified in the datasheet. One of the problems of this method is that any offset in the voltage will be integrated over time. Any small offset in the average voltage is removed first, additionally, the average value of B is removed as the initial value for B is unknown. Bdc cannot be inferred with this method.")
        st.write("The magnetic field strength H is directly calculated as the single-cycle current multiplied by the number of turns in primary and divided by the effective length listed in the datasheet.")
        st.write("This processed H and B data is also available for download, together with the resulting sampling time for the 1024 evenly spaced samples.")
    with st.expander("5. How are losses and other parameters obtained?"):
        st.write("For data visualization, core losses are obtained and plotted against Bac, f, d1, d2 or d4, d3, Hdc, T.")
        st.write("The instantaneous power can be calculated directly as the product of the single-cycle voltage and current waveforms. The volumetric losses are the average value of the instantaneous power divided by the effective volume listed in the datasheet. The instantaneous power can be calculated directly as the product of the single-cycle voltage and current waveforms. The volumetric losses are the average value of the instantaneous power divided by the effective volume listed in the datasheet. For the calculation of core losses using the sequence-to-sequence neural network, volumetric core losses are calculated as the integral of the single-cycle H(t) over B(t) multiplied by the frequency.")
        st.write("Parameters other than the temperature are obtained from the single-cycle waveform instead of the commanded parameters for improved data quality. The amplitude of the flux density is calculated as (max(B)-min(B))/2. Hdc is directly the average of the H waveform. The duty cycles are calculated by checking the voltage waveform. The percentage of the waveform above any voltage levels between min(V) and max(V) is calculated. Because d2=d4, and the waveform has only three levels, after filtering the noise, d1 directly corresponds to the fraction of the voltage waveform above (3(max(V)-min(V))/4) while d3 is the fraction of the voltage below (max(V)-min(V))/4. For sinusoidal waveforms, since there is no duty cycle, d1 to d4 are set to -1 or NaN.")
        st.warning("For data visualization, Hdc is rounded to 1A/m, the duty cycles are rounded in 10% steps, and temperature is reported using the water set temperature directly")
    st.subheader('Neural Networks:')
    with st.expander("1. What machine learning algorithm is used?"):
        st.write("A single type of neural network is used for all inferences on the webpage. The model calculates a single cycle for the H waveform based on the input B waveform and other scalars. This sequence-to-sequence neural network is a transformer-based encoder-projector-decoder architecture with B(t), T, f, and Hdc as the inputs and H(t) as the output as shown in the figure. The single cycle B and H waveforms post-processed from the measurements are downsampled from 1024 samples to 128 and are used to train the neural network for each material, using data augmentation and all the data points available in the database. This trained network is built into the webpage and used to obtain the B-H loop for any desired input. Volumetric core losses are calculated based on the generated B-H loop. All the details regarding the implementation will be available in a future paper (APEC 2023).")
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'transformer.jpg')), width=500)
    with st.expander("2. What are the limitations of this method?"):
        st.write("The quality of the output of a well-trained neural network is bounded by the quality and availability of the data used for training. As a result, all the errors in the measurement and processing of the data directly affect the output of the neural network. Only measured data is used for training, but the data is limited to certain ranges of operating conditions and waveform types. The quality of the output outside those ranges, or with waveform differing much from sinusoidal, triangular, or trapezoidal waveforms cannot be guaranteed. Other than that, the dimension of the neural network and training process has been designed to minimize the error in the H waveform and avoid overfitting. Finally, the network has been trained to minimize the rms error of the H waveform rather than losses, as a result, some discrepancy in the inferred core loss is also expected.")

st.markdown("""---""")
