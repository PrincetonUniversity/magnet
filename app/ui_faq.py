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
    st.write("")
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

    st.write(f'*iGSE parameters obtained from the sinusoidal measurements at 25 C without bias; '
             f'with Pv, f, and B in W/m^3, Hz and T respectively.')

    st.markdown("""---""")

    st.title('Frequently Asked Questions')
    st.write('')
    st.write("""
        This section includes some explanations regarding how data is captured, processed, and how losses are calculated.
        
        If you have any other questions, please let us know so we can add them to the list.
        You can either write us to any of the e-mails listed on the left or open a new issue in our GitHub repository.
    """)
    st.write('')
    st.subheader('Data Acquisition System:')
    with st.expander("1. How is the data measured?"):
        st.write("""
            The magnetic core loss measurement is supported by an automated data acquisition system as shown in the figure.
            The magnetic core loss is directly measured and calculated via voltamperometric method (also referred to as the two winding method), 
            which has high simplicity and flexibility of operation.        
            
            The automated data acquisition system consists of a power stage to generate the desired excitations, 
            a magnetic core as the device-under-test, a wide-band coaxial shunt for current measurement, 
            an oscilloscope for data acquisition, and an oil bath with a heater for temperature control.
            
            The hardware system is controlled and coordinated by a Python-based program on the host PC, 
            which enables fully automated equipment setting and core loss measurement under pre-programmed excitations.
            Please refer to [1] for more details about the core loss data acquisition.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'system-overview.jpg')), width=500)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'system-photo.jpg')), width=500)
        st.write("""          
            [1] H. Li, D. Serrano, T. Guillod, E. Dogariu, A. Nadler, S. Wang, M. Luo, V. Bansal, Y. Chen, C. R. Sullivan, and M. Chen, "MagNet: an Open-Source Database for Data-Driven Magnetic Core Loss Modeling," IEEE Applied Power Electronics Conference (APEC), Houston, 2022.
        """)
    with st.expander("2. How are Sinusoidal excitations generated?"):
        st.write("""
            For sinusoidal excitation, a signal generator connected to a power amplifier (Amplifier Research 25A250AM6) is used.
            The load for the power amplifier is the series connection of a blocking capacitor and the core under test.
            The computer is used to apply the desired frequency and amplitude to the signal generator in each test.
        """)
    with st.expander("3. How are Triangular and Trapezoidal excitations generated?"):
        st.write("""
            For Triangular and Trapezoidal excitations, the desired waveform is applied using a T-Type inverter where the load is the core under test and a blocking capacitor.
            This T-Type inverter is supplied by two power supplies controlled by the computer. The voltage of the power supply sets the amplitude of the waveform.
            To obtain the different shapes, the PWM gate signals are controller by a DSP through the computer.
            With this inverter, the load can be connected to the positive power supply, the negative power supply or to ground, so the amplitude, frequency, and shape of the output waveform can be controlled.
            For these types of waveform, four duty cycles are defined, d1 to d4, corresponding to different parts of the switching cycle.
            Please note that d2=d4 in all the tests performed so far. 
            
            In the following figure, examples of different waveforms and their duty cycles are shown.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'trapezoidal-waveforms.jpg')), width=500)
    with st.expander("4. How is the DC bias applied?"):
        st.write("""
        With this setup, Hdc can be controlled by adding a constant current in the primary side.
        This current is added using a voltage supplied using in current mode control connected between the DC blocking cpacitor and the device under test.
        Additionally, a mirror transformer and a series inductor are added to minimize unwanted current ripples.
        """)
    with st.expander("5. How is the temperature controlled?"):
        st.write("""
        The device under test is placed in an oil bath. The oil is continously moved using a magnetic stirrer to improve heat disipation.
        To maintain the oil temperature constant, it is placed inside a water tank. The temperature of the water is controlled using a water heater to set the desired temperature for the test.
        """)

    st.subheader('Data Processing:')
    with st.expander("1. What information is processed?"):
        st.write("""
            The information processed is the voltage, current and time sequences collected with the oscilloscope (Tektronix DPO4054 in our setup).
            These files are saved as .cvs files with a datapoint per row, and a time sample in each row.
            The data contains 10.000 samples per datapoint, with a sampling time of 10 ns.
            Each datapoint is a test conducted with a different core and waveform (shape, flux density, and frequency); a few switching cycles are saved.
            
            The figure below shows an example of the voltage and current recorded in a Trapezoidal test for N87 material.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'volt-curr.jpg')), width=500)
        st.write("""
            Then, these files are processed using Matlab.
            The Matlab scripts can be found at GitHub -> Princeton University -> MagNet -> Main -> Scripts -> Matlab Scripts.
            Besides the raw data from the oscilloscope, the Matlab script also needs the core under test information and points from the volumetric losses plot in the datasheet.
        """)
    with st.expander("2. How is the frequency identified?"):
        st.write("""
            To obtain the frequency, a Fast Fourier Transform (FFT) is not enough due to the low frequency resolution.
            
            The power spectral density using Welch method is used instead to obtain the frequency instead.
            
            In the picture below the FFT of the current in the previous example is shown; for this example, the fundamental frequency is 50 kHz.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'fft.jpg')), width=500)
    with st.expander("8. How are the single-cycle waveforms generated?"):
        st.write("""
            Besides the processed data and the raw files, the webpage also offers post-processed voltage and current waveforms containing a single switching cycle with 100 evenly spaced samples.
            These waveforms can be useful to analyze B-H loops, generate physic-based models, or train Neural Networks.

            To generate the single-cycle waveforms, the raw voltage and current waveform are split into the different switching cycles, then, for each switching cycle, 100 samples are obtained by interpolation, and, finally, the 100 samples for each cycle are averaged into a single switching cycle waveform.
            The following simplified figure clarifies the process for obtaining a single voltage waveform; the same process is applied to the current waveform.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'single-cycle.jpg')), width=500)
    with st.expander("4. How do we get B and H?"):
        st.write("""
            The instantaneous flux density is the integral of the voltage waveform divided by the number of turns of the secondary winding used to measure the voltage, and also divided by the effective area specified in the datasheet.

            One of the problems of this method is that any offset in the voltage will be integrated over time, as shown in red in the picture below.
            Moreover, this "raw" integral of the voltage would have a different average value depending on the initial time the waveform is integrated.
            These effects can be removed by subtracting the "average" flux density, or offset, from the raw signal.
            For this purpose a moving-mean filter with a switching period length is employed, this variable represents the offset of the flux density (in blue).
            Since the moving-mean does not use the complete period at the beginning and end of the waveform, the first and last switching cycles are removed.

            Once the offset is removed, the corrected shape for the instantaneous flux density is obtained (in black)
            The flux density amplitude is calculated from the maximum (or minimum) of the corrected waveform.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'flux.jpg')), width=500)

    with st.expander("3. How are losses obtained?"):
        st.write("""
            The instantaneous power can be calculated directly as the product of the voltage and current.
            The losses are the average value of the instantaneous power.
            
            In fact, the drawback of the voltamperometric method used for the tests is that the reactive power is much higher than the losses.
            Both the instantaneous power and the losses are depicted in the figure below. 
            
            To improve accuracy, the offset in the voltage and current are removed before computing the losses.
            Additionally, only an entire number of switching cycles is used for this calculation, as losses may not be constant along the switching cycle.
            Finally, the loss density is calculated as the power divided by the effective volume specified in the datasheet.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'power.jpg')), width=500)
    with st.expander("3. How are duty cycles calculated?"):
        st.write("""

        """)
    with st.expander("5. How are the results plotted?"):
        st.write("""
            Besides the frequency, volumetric loss, and flux density amplitude, the different duty cycles for Trapezoidal and Triangular waveforms are identified too (d1 to d4).
            For Sinusoidal waveforms, since there is no duty cycle, d1 to d4 are set to -1.
            For Triangular and Trapezoidal, a scrip processes the voltage waveform to obtain the duty cycles.

            With these values, the following plot depicting the loss density against frequency and flux density can be obtained for each material and combination of duty cycles
            For this example, the material N87 and Sinusoidal excitation are used.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'results-example.jpg')), width=500)
    with st.expander("6. How are the Steinmetz parameters obtained?"):
        st.write("""
            In this webpage, the Steinmetz parameters for each materials are required to estimate the core losses with the iGSE method.
            The parameters are obtained from the Sinusoidal results for each material.
            
            For each material, the parameters k, α, and β that best fit the measured data are obtained.
            For such purpose, a least-squares curve fitting method is used ("lsqcurvefit" Matlab function).
            To ensure that data points with large losses have the same impact as those with reduced losses, the logarithm of losses is used.
            
            Once k, α, and β are obtained, ki for iGSE is calculated.
            Please note that the ki, α, and β are obtained from Sinusoidal tests, even when used for Trapezoidal or Triangular loss calculations.
        """)

    st.subheader('Core Loss Prediction:')
    with st.expander("1. What is the algorithm that used in the core loss analysis?"):
        st.write("""
            Two types of algorithms are currently used: the iGSE (improved Generalized Steinmetz Equation)  model and the ML (Machine Learning) based model.
        """)
        st.write("""        
            The iGSE model is calculated based on the following expressions,
            where the Steinmetz parameters are extracted by the global fitting of entire sinusoidal excitation dataset of each material.
            More advanced analytical model will be deployed in the future release.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'model-igse.jpg')), width=300)
        st.write("""          
            The ML-based model is currently based on a feedforward neural network (FNN), as shown in the following figure. 
            This network is a scalar-to-scalar model, which takes the fundamental frequency, AC flux density amplitude 
            and the duty ratio (to describe the shape of the waveform) as the input, and the core loss per volume as the output.
            It is able to calculate the core loss under sinusoidal, triangular and trapezoidal wave excitations.
            More sequence-to-scalar and sequence-to-sequence models (based on LSTM or transformer, for example) will be deployed in the future release,
            which are expected to predict the core loss under arbitrary waveforms without the limitation of waveform shapes.
            Please refer to [1] for more details about the ML-based magnetic core loss modeling.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'model-fnn.jpg')), width=400)
        st.write("""          
            [1] H. Li, D. Serrano, T. Guillod, E. Dogariu, A. Nadler, S. Wang, M. Luo, V. Bansal, Y. Chen, C. R. Sullivan, and M. Chen, "MagNet: an Open-Source Database for Data-Driven Magnetic Core Loss Modeling," IEEE Applied Power Electronics Conference (APEC), Houston, 2022.
        """)

    st.subheader('Limitations:')
    with st.expander("1. Frequency limit of voltamperometric measurements"):
        st.write("""
            The main limitation of the voltamperometric method is the impact that a small delay between the voltage and current measurement (phase discrepancy) can have in the calculation of losses.
            This is especially concerning at high frequency and for materials with high quality factor.
            For a complete discussion on this issue please check [[1]](https://ieeexplore.ieee.org/abstract/document/7479554). Further details and limitations of this method can be found in [[2]](https://ieeexplore.ieee.org/document/8403009).
            
            To ensure that the error is small compared with the measurements, the maximum frequency of the tests is limited to 500 kHz.
            Unfortunately, no other method can be readily automated as this one. For this reason, we don't plan to measure data at higher frequency.
            
            [[1]](https://ieeexplore.ieee.org/abstract/document/7479554) D. Hou, M. Mu, F. C. Lee and Q. Li, "New High-Frequency Core Loss Measurement Method With Partial Cancellation Concept," in IEEE Transactions on Power Electronics, vol. 32, no. 4, pp. 2987-2994, April 2017.
            
            [[2]](https://ieeexplore.ieee.org/document/8403009) E.Stenglein, D.Kuebrich, M.Albach and T.Duerbaum, "Guideline for Hysteresis Curve Measurements with Arbitrary Excitation: Pitfalls to Avoid and Practices to Follow," PCIM Europe 2018; International Exhibition and Conference for Power Electronics, Intelligent Motion, Renewable Energy and Energy Management, 2018, pp.1-8.
        """)
    with st.expander("2. Quality of the excitation"):
        st.write("""
           Obtaining the desired waveform for the excitation is challenging.

            For Sinusoidal waveforms, the linearity of the power amplifier can be a problem in certain conditions.
            Power amplifier are designed to supply 50 Ω loads.
            However, in this setup, the impedance depends on the transformer under test and the frequency, so it is never 50 Ω.
            Moreover, there is a limit to the reactive power the power amplifier can provide, which limits the range of the tests that can be performed.
            Additionally, when the voltage or the current is too high, the waveforms start to distort.
            Finally, for tests with a large flux density excursion, the variations of the inductance along the cycle become a problem.
            When the device is close to saturation the current is no longer sinusoidal, and, since the power amplifier provides sinusoidal power, the voltage is also distorted.
            
            For Triangular and Trapezoidal excitations, the quality of the excitation depends on the power stage used to generate the voltage.
            A limitation is the dv/dt and ringing during switching transitions.
            Even though the T-Type inverter used employs GaN devices with low drain-to-source capacitance, the shape of the voltage during the transition is affected by the current and parasitic inductance.
            Please note that these effects are intrinsic to power converters and are present in the real operation of power transformers and inductors.
            Moreover, the DSP used to control the gate signals has a resolution of 10 ns and a minimum dead time is required, currently set to 70 ns.
            These parameters set a limit on the accuracy of the duty cycle and frequency, becoming more significant as the frequency increases.
            
            The voltage and current measurements are available for download for those interested in the nonidealities of the excitation.
        """)
    with st.expander("3. Temperature stability"):
        st.write("""
            A drawback of a system taking one measurement each 1 to 2 seconds is the heating of the core under test due to core losses.
            Since temperature affects the core loss, it is important to keep the core at the desired temperature during testing.
            To do so, the core is submerged into a mineral oil tank controlled to be at a fixed temperature.
            Nevertheless, the temperature of the core can still rise if the average power loss during testing is high.

            To mitigate temperature increase in the core for future measurements, we are:
            
            1) Adding a stirrer for the oil tank, which improves the thermal conductivity.
            
            2) Skipping points where losses are too high to be practical.
            
            3) Including a waiting time between tests to let the core cool down.
            
            Initial tests with the core N87 R22.1x13.7x7.9 and sinusoidal excitation show an increase in losses of 10~15% when the temperature is controlled in a better way.
            Measurements with better temperature control will be captured and added to the webpage in the future.
        """)
    st.write(f'During the tests, the core temperature may increase 5 C ~ 10 C in worst case conditions.')
    with st.expander("4. Effect of variations in core dimensions"):
        st.write("""
            Please note that B, H, and losses are calculated from the measured voltage and currents and the effective dimensions in the datasheet (effective length, effective area, and effective volume).
            Unfortunately, the physical dimensions vary from core to core, and the variations of the dimensions with respect to the specified values are translated into an error in the reported B, H, and losses.
            As most toroids used for testing are coated, their real dimensions cannot be measured.
            This is a common problem that designers must face and leads to variations in inductance and core loss between samples.
            
            This effect can be considerable. For instance, for the core N87 R22.1x13.7x7.9, variations in B of up to -10.8%/+7.6% and variations in losses of up to -7.3%/+10.4% would be obtained due to the change in the effective area when comparing the biggest and smallest possible core (based on tolerances) with respect to the "nominal" core.
        
            For shapes other than toroidal, the gap is minimum given by the surface roughness. Its effects are neglected in the calculations.
        """)
    with st.expander("4. Effect of switching speed and parasitics"):
        st.write("""
        """)
    st.write(f'dv/dt during the transitions is not controlled but might impact the data.')

    with st.expander("2. Postprocessing artifacts"):
        st.write("""
            Post-processing the data in this way has some limitations.
            First, if the frequency of the waveform is not accurately identified, some unwanted averaging effect will appear in the single-cycle waveform, leading to an inaccurate capture of the waveform.
            Nevertheless, to ensure the quality of the data, data points where the single-cycle (before averaging) differs too much from one switching cycle to the next are discarded.
        """)
    st.write(f'Errors in the fundamental frequency detection affect the quality of the single-cycle data reported.')

    st.markdown("""---""")
