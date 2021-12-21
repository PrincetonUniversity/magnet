import streamlit as st

def ui_faq(m):
    
    st.subheader('Introduction of MagNet')
    
    st.write('\
         MagNet is a large-scale dataset designed to enable researchers modeling \
         magnetic core loss using machine learning to accelerate the design process \
         of power electronics. The dataset contains a large amount of voltage and \
         current data of different magnetic components with different shapes of \
         waveforms and different properties measured in the real world. Researchers \
         may use these data as pairs of excitations and responses to build up dynamic \
         magnetic models or calculate the core loss to derive static models. MagNet \
         is continuously being maintained and updated with new data.\
    ')
     
    st.subheader('How to Cite')
    
    st.write('\
        If you used MagNet, please cite us with the following:\
        \
        \n [1] H. Li, D. Serrano, T. Guillod, E. Dogariu, A. Nadler, S. Wang, M. Luo, V. Bansal, Y. Chen, C. R. Sullivan, and M. Chen, \
        "MagNet: an Open-Source Database for Data-Driven Magnetic Core Loss Modeling," \
        IEEE Applied Power Electronics Conference (APEC), Houston, 2022.\
        \
        \n [2] E. Dogariu, H. Li, D. Serrano, S. Wang, M. Luo and M. Chen, \
        "Transfer Learning Methods for Magnetic Core Loss Modeling,” \
        IEEE Workshop on Control and Modeling of Power Electronics (COMPEL), Cartagena de Indias, Colombia, 2021.\
        \
        \n [3] H. Li, S. R. Lee, M. Luo, C. R. Sullivan, Y. Chen and M. Chen, \
        "MagNet: A Machine Learning Framework for Magnetic Core Loss Modeling,” \
        IEEE Workshop on Control and Modeling of Power Electronics (COMPEL), Aalborg, Denmark, 2020.\
    ')


    st.subheader('Frequently Asked Questions')
        
    with st.expander("1. How does xxx?"):
     st.write("""
              How does xxx?
     """)
     st.image("https://static.streamlit.io/examples/dice.jpg")
     
    with st.expander("1. How does xxx?"):
     st.write("""
              How does xxx?
     """)
        
    st.markdown("""---""")
