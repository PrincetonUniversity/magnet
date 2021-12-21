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
        
    with st.expander("1. How's the magnetic core loss data measured?"):
     st.write("""
        The magnetic core loss measurement is supported by an automated data acquisition system as shown in the figure.
        The magnetic core loss is directly measured and calculated via voltamperometric method, 
        which has high simplicity and flexibility of operation.

        The automated data acquisition system consists of a power stage to generate the desired excitations, 
        a magnetic core as the device-under-test, a wide-band coaxial shunt for current measurement, 
        an oscillpscope for data acquisition, and an oil bath with heater for temperature control.
        
        The hardware system is controlled and coordinated by a Python-based program on the host PC, 
        which enables fully automated equipment setting and core loss measurement under pre-programmed excitations.
        Please refer to [1] for more details about the core loss data acquisition.
        

     """)
     st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'system-overview.jpg')), width=500)
     st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'system-photo.jpg')), width=500)
     
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
     
    with st.expander("3. What's the algorithm that used in the core loss analysis?"):
     st.write("""
        Two types of algorithms are currently used: the iGSE model and the ML-based model.
     """)       
     
     st.write("""        
        The iGSE (improved generalized Steinmetz equation) model is calculated based on the following expressions,
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
