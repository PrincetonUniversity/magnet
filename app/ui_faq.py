import os.path
from PIL import Image
import streamlit as st

STREAMLIT_ROOT = os.path.dirname(__file__)


def ui_faq(m):
    
    st.subheader('Introduction of MagNet')
    
    st.write("""
         MagNet is a large-scale dataset designed to enable researchers modeling 
         magnetic core loss using machine learning to accelerate the design process 
         of power electronics. The dataset contains a large amount of voltage and 
         current data of different magnetic components with different shapes of 
         waveforms and different properties measured in the real world. Researchers 
         may use these data as pairs of excitations and responses to build up dynamic 
         magnetic models or calculate the core loss to derive static models. MagNet 
         is continuously being maintained and updated with new data.
    """)
     
    st.subheader('How to Cite')
    
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

    st.subheader('Frequently Asked Questions')

    st.write('Data Acquisition System:')
    with st.expander("1. How is the magnetic core loss data measured?"):
        st.write("""
            The magnetic core loss measurement is supported by an automated data acquisition system as shown in the figure.
            The magnetic core loss is directly measured and calculated via voltamperometric method, 
            which has high simplicity and flexibility of operation.
            
            
            This method consist on...
            
            
            
            The automated data acquisition system consists of a power stage to generate the desired excitations, 
            a magnetic core as the device-under-test, a wide-band coaxial shunt for current measurement, 
            an oscilloscope for data acquisition, and an oil bath with heater for temperature control.
            
            The hardware system is controlled and coordinated by a Python-based program on the host PC, 
            which enables fully automated equipment setting and core loss measurement under pre-programmed excitations.
            Please refer to [1] for more details about the core loss data acquisition.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'system-overview.jpg')), width=500)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'system-photo.jpg')), width=500)
    with st.expander("2. How are Sinusoidal excitations generated?"):
        st.write("""
        For sinusoidal excitation, a signal generator is used.
        The desired frequency and amplitude are controlled through the computer.
        The signal generator is connected to the power amplifier (Amplifier Research 25A250AMB)
        """)

    with st.expander("3. How are Triangular and Trapezoidal excitations generated?"):
        st.write("""
        For Triangular and Trapezoidal, the core under test is excited with a power converter.
        The power converter is a T-Type inverter where the load is the core under test and a series capacitor.
        The gate signals are controller by a DSP commanded by the computer and voltage of the power supplies is also controlled by the computer.
        By controlling the gate signals in the inverter, the load can be connected to the positive power supply, the negative power supply or to ground,
        so the amplitude, frequency and shape of the output waveform can be controlled.
        """)

    st.write('Data Processing:')
    with st.expander("1. How are the frequency, flux and losses obtained from the measured waveforms?"):
        st.write("""
            First, the voltage, current and time sequences for each test are collected with the oscilloscope (Tektronix DPO4054 in our setup) and saved as .cvs files.
            Each test is conducted with a different core and waveform (shape, flux density and frequency), and a few switching cycles are saved.
            Then, these files are processed using Matlab. The Matlab scrips used can be found at https://github.com/PrincetonUniversity/magnet/tree/main/scripts
            The figure below shows an example of the voltage and current recorded in a Trapezoidal test for N87 material.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'volt-curr.png')), width=500)
        st.write("""
            To obtain the frequency, a Fast Fourier Transform (FFT) of the current is performed.
            Please note that this script only works for constant time steps samples so far.
            The tests recorded by our Lab contain 10.000 samples with a sampling time of 10 ns.
            As a result, the frequency resolution of the FFT is 10 kHz.
            Only tests with a switching frequency of 10 kHz are performed, to have an entire number of switching cycles in the total sample.
            In the picture below the FFT of the current of the previous example is shown.
            The fundamental frequency is 50 kHz.  
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'fft.png')), width=500)
        st.write("""
            The instantaneous power can be calculated as the product of the voltage and current.
            The losses are the average value of the instantaneous power.
            In fact, the drawback of the voltamperometric method used for the tests is that the reactive power is much higher than the losses.
            Both the instantaneous power and the losses are depicted in the figure below. 
            To improve accuracy, the offset in the voltage and current are removed before computing the losses.
            Additionally, only and an entire number of switching cycles is used, as losses may not be constant along the switching cycle.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'power.png')), width=500)
        st.write("""
            The flux density amplitude is calculated from the the voltage signal.
            The flux density is the integral of the voltage waveform divided by the number of turns of the secondary winding used to measure the voltage, and also divided by the effective area (which can be found in the datasheet of the core)
            One of the problems of this method is that any offset in the voltage will be integrated over time, as shown in red in the picture below.
            This effect can be removed by filtering the flux density (in blue) and removing it from the original signal.
            For this purpose a moving mean filter at the fundamental frequency is employed, and the first and last switching cycles are removed.
            The amplitude then is calculated as the maximum (or minimum) of the resulting flux density waveform (in black).
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'flux.png')), width=500)
#    with st.expander("1. How is the duty cycle obtained?"):
#        st.write("""
#            Finally, for Triangular and Trapezoidal waveforms, the waveforms are divided in four segments with duty cycles d1, d2 ,d3 and d4, where d1 (or dP) is the rising part of the waveform and d3 (or DN) is the falling part.
#            The other segments, d2 and d4 (or d0) are equal in all the waveforms obtained so far.
#            Please note that Triangular waveforms are an specific type of Trapezoidal waveform where d2=d4=0.
#            Additionally, note that the segments for d2 and d4 not always correspond to flat regions, this hold only in the cases where d1=d3, as in all other cases the series capacitor will block a certain voltage.
#            To calculate the different duty cycles from the voltage waveform, the idea is to do a sweep of "threshold" voltages between the maximum and minimum, and for each threshold, count how many voltage samples are above the threshold, then, dividing this number over the total number of samples, an idea of the shape of the waveform is obtained.
#            Finding the the value at 1/4 of the treshold length and at 3/4th of the threshold length gives the value of d1 and d3, however this is only valid for d2=d4.
#        """)
#        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'duty.gif')), width=500)
#        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'duty.png')), width=500)

    with st.expander("2. What's the definition of the outlier factor?"):
        st.write("""
            The outlier factor reflects the smoothness of the measured data. 
            
            For each point in the dataset, the estimated power losses are calculated based on the Steinmetz
            parameters inferred from the nearby points that are close in terms of frequency and flux density to the considered point. 
            And the outlier factor calculates the relative discrepancy between the estimated value and the measured value.
            A data point whose measured core loss is far from its estimated value will get a high outlier factor,
            and can be considered an outlier. 
            
            By setting the maximum outlier factor, you set your tolerance for the smoothness of the dataset, 
            where the data points with outlier factor that higher than the threshold will be removed from the dataset
            to improve the overall data quality.
        """)
    with st.expander("2. How are the datasheet points generated?"):
        st.write("""
            The datasheet datapoints are interpolated from the manufacturer's datasheet for each specific material.
            Plase note that the datasheets for materials 3E6(3E10) and N30 do not contain power loss plots, so no these materials are not selectable.
            
        """)
    st.write('Core Loss Prediction:')
    with st.expander("3. What's the algorithm that used in the core loss analysis?"):
        st.write("""
            Two types of algorithms are currently used: the iGSE (improved Generalized Steinmetz Equation)  model and the ML (Machine Learning) based model.
        """)
        st.write("""        
            The iGSE model is calculated based on the following expressions,
            where the Steinmetz parameters are extracted by the global fitting of entire sinusoidal excitation dataset of each material.
            More advanced analytical model will be deployed in the future release.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'model-igse.jpg')), width=400)
        st.write("""          
            The ML-based model is currently based on a feedforward neural network (FNN), as shown in the following figure. 
            This network is a scalar-to-scalar model, which takes the fundamental frequency, AC flux density amplitude 
            and the duty ratio (to describe the shape of the waveform) as the input, and the core loss per volume as the output.
            It is able to calculate the core loss under sinusoidal, triangular and trapezoidal wave excitations.
            More sequence-to-scalar and sequence-to-sequence models (based on LSTM or transformer, for example) will be deployed in the future release,
            which are expected to predict the core loss under arbitrary waveforms without the limitation of waveform shapes.
            Please refer to [1] for more details about the ML-based magnetic core loss modeling.
        """)
        st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'model-fnn.jpg')), width=500)

    st.markdown("""---""")
