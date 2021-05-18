import streamlit as st
import json
import plotly.express as px
import numpy as np
import re
from search import search

# setup the user interface 
st.set_page_config(layout="wide")
st.image('pulogo.jpg', width=600)


st.sidebar.image('magnetlogo.jpg', width=300)

st.sidebar.header('Princeton MagNet v1.0 Beta')
function_list = ("Core Loss Database", "Core Loss Prediction")
function_select = st.sidebar.radio("Select one of the two functions",
    function_list
)

if function_select == "Core Loss Database":
    
    st.title("Princeton MagNet - Core Loss Database")
    st.header("Princeton Power Electronics Research Lab, Princeton University")
    st.markdown("""---""")
    @st.cache
    def load_data(data_dir):
        with open(data_dir) as jsonfile:
            Data = json.load(jsonfile)
        return Data
    
    # read the necessary information for display
    material_listA = ("N27","N49","N87")  
    material_typeA = st.sidebar.selectbox(
        "Material A:",
        material_listA
    )
    
    excitation_listA = ("Datasheet","Sinusoidal","Triangle","Trapezoidal",
        "Arbitrary")
    excitation_typeA = st.sidebar.selectbox(
        "Excitation A:",
        excitation_listA
    ) 
    
    [FminA,FmaxA] = st.sidebar.slider('Frequency Range A (Hz)', 
                                      10000, 300000, (10000,300000),step=1000)
    [BminA,BmaxA] = st.sidebar.slider('Flux Density Range A (mT)', 
                                      10, 300, (10,300),step=1)
    
    if excitation_typeA == "Datasheet" or excitation_typeA == "Sinusoidal":
        st.header(material_typeA+", "+excitation_typeA+", f=["+str(FminA)+"~"+str(FmaxA)+"] Hz"
                 +", B=["+str(BminA)+"~"+str(BmaxA)+"] mT")
        
        data_dirA="./Data/Data_" + material_typeA + "_" + excitation_typeA + "_light.json"
        # create a subset of data that meet the rules
        DataA = load_data(data_dirA)
        SubsetA = search(DataA,FminA,FmaxA,BminA,BmaxA,0,1)
        
        if not SubsetA['Frequency']:
            st.write("Warning: No Data in Range")
        else:
            col1, col2 = st.beta_columns(2)
            with col1:
                fig1 = px.scatter(x=SubsetA['Frequency'],y=SubsetA['Power_Loss'],
                                  log_x=True,log_y=True,
                                  labels={'x':'Frequency [Hz]', 'y':'Power Loss [kW/m^3]'})
                fig1.update_traces(marker=dict(size=5,color='darkred'),
                              selector=dict(mode='markers'))
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.scatter(x=SubsetA['Flux_Density'],y=SubsetA['Power_Loss'],
                                  log_x=True,log_y=True,
                                  labels={'x':'Flux_Density [mT]', 'y':'Power Loss [kW/m^3]'})
                fig2.update_traces(marker=dict(size=5,color='darkred'),
                              selector=dict(mode='markers'))
                st.plotly_chart(fig2, use_container_width=True)
    
    if excitation_typeA == "Triangle":
        [DminA,DmaxA] = st.sidebar.slider('Duty Ratio Range A', 
                                      0.0, 1.0, (0.0,1.0),step=0.05)

        st.header(material_typeA+", "+excitation_typeA+", f=["+str(FminA)+"~"+str(FmaxA)+"] Hz"
                 +", B=["+str(BminA)+"~"+str(BmaxA)+"] mT"+", D=["+str(DminA)+"~"+str(DmaxA)+"]")
        
        # read the corresponding data
        data_dirA="./Data/Data_" + material_typeA + "_" + excitation_typeA + "_light.json"
        # create a subset of data that meet the rules
        DataA = load_data(data_dirA)
        SubsetA = search(DataA,FminA,FmaxA,BminA,BmaxA,DminA,DmaxA)
        
        if not SubsetA['Frequency']:
            st.write("Warning: No Data in Range")
        else:
            col1, col2 = st.beta_columns(2)
            with col1:
                fig1 = px.scatter(x=SubsetA['Frequency'],y=SubsetA['Power_Loss'],
                                  log_x=True,log_y=True,
                                  labels={'x':'Frequency [Hz]', 'y':'Power Loss [kW/m^3]'})
                fig1.update_traces(marker=dict(size=5,color='darkred'),
                              selector=dict(mode='markers'))
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.scatter(x=SubsetA['Flux_Density'],y=SubsetA['Power_Loss'],
                                  log_x=True,log_y=True,
                                  labels={'x':'Flux_Density [mT]', 'y':'Power Loss [kW/m^3]'})
                fig2.update_traces(marker=dict(size=5,color='darkred'),
                              selector=dict(mode='markers'))
                st.plotly_chart(fig2, use_container_width=True)
    
    if excitation_typeA == "Trapezoidal":
        [DminA1,DmaxA1] = st.sidebar.slider('Duty Ratio Range A1', 
                                      0.0, 1.0, (0.0,1.0),step=0.05)
        [DminA2,DmaxA2] = st.sidebar.slider('Duty Ratio Range A2', 
                                      0.0, 1.0, (0.0,1.0),step=0.05)
        st.write("Warning: Function Not Supported")
    
    ################################################################
    # read the necessary information for display
    material_listB = ("N27","N49","N87")  
    material_typeB = st.sidebar.selectbox(
        "Material B:",
        material_listB
    )
    
    excitation_listB = ("Datasheet","Sinusoidal","Triangle","Trapezoidal",
        "Arbitrary")
    excitation_typeB = st.sidebar.selectbox(
        "Excitation B:",
        excitation_listB
    ) 
    
    [FminB,FmaxB] = st.sidebar.slider('Frequency Range B (Hz)', 
                                      10000, 300000, (10000,300000),step=1000)
    [BminB,BmaxB] = st.sidebar.slider('Flux Density Range B (mT)', 
                                      10, 300, (10,300),step=1)
    
    if excitation_typeB == "Datasheet" or excitation_typeB == "Sinusoidal":
        st.header(material_typeB+", "+excitation_typeB+", f=["+str(FminB)+"~"+str(FmaxB)+"] Hz"
                 +", B=["+str(BminB)+"~"+str(BmaxB)+"] mT")
        
        # read the corresponding data
        data_dirB="./Data/Data_" + material_typeB + "_" + excitation_typeB + "_light.json"
        # create a subset of data that meet the rules
        DataB = load_data(data_dirB)
        SubsetB = search(DataB,FminB,FmaxB,BminB,BmaxB,0,1)
        
        if not SubsetB['Frequency']:
            st.write("Warning: No Data in Range")
        else:
            col1, col2 = st.beta_columns(2)
            with col1:
                fig3 = px.scatter(x=SubsetB['Frequency'],y=SubsetB['Power_Loss'],
                                  log_x=True,log_y=True,
                                  labels={'x':'Frequency [Hz]', 'y':'Power Loss [kW/m^3]'})
                fig3.update_traces(marker=dict(size=5,color='darkblue'),
                              selector=dict(mode='markers'))
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                fig4 = px.scatter(x=SubsetB['Flux_Density'],y=SubsetB['Power_Loss'],
                                  log_x=True,log_y=True,
                                  labels={'x':'Flux_Density [mT]', 'y':'Power Loss [kW/m^3]'})
                fig4.update_traces(marker=dict(size=5,color='darkblue'),
                              selector=dict(mode='markers'))
                st.plotly_chart(fig4, use_container_width=True)
    
    if excitation_typeB == "Triangle":
        [DminB,DmaxB] = st.sidebar.slider('Duty Ratio Range B', 
                                      0.0, 1.0, (0.0,1.0),step=0.05)
        
        st.header(material_typeB+", "+excitation_typeB+", f=["+str(FminB)+"~"+str(FmaxB)+"] Hz"
                 +", B=["+str(BminB)+"~"+str(BmaxB)+"] mT"+", D=["+str(DminB)+"~"+str(DmaxB)+"]")
        
        # read the corresponding data
        data_dirB="./Data/Data_" + material_typeB + "_" + excitation_typeB + "_light.json"
        # create a subset of data that meet the rules
        DataB = load_data(data_dirB)
        
        if not SubsetB['Frequency']:
            st.write("Warning: No Data in Range")
        else:
            SubsetB = search(DataB,FminB,FmaxB,BminB,BmaxB,DminB,DmaxB)
            col1, col2 = st.beta_columns(2)
            with col1:
                fig3 = px.scatter(x=SubsetB['Frequency'],y=SubsetB['Power_Loss'],
                                  log_x=True,log_y=True,
                                  labels={'x':'Frequency [Hz]', 'y':'Power Loss [kW/m^3]'})
                fig3.update_traces(marker=dict(size=5,color='darkblue'),
                              selector=dict(mode='markers'))
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                fig4 = px.scatter(x=SubsetB['Flux_Density'],y=SubsetB['Power_Loss'],
                                  log_x=True,log_y=True,
                                  labels={'x':'Flux_Density [mT]', 'y':'Power Loss [kW/m^3]'})
                fig4.update_traces(marker=dict(size=5,color='darkblue'),
                              selector=dict(mode='markers'))
                st.plotly_chart(fig4, use_container_width=True)

    if excitation_typeB == "Trapezoidal":
        [DminA1,DmaxB1] = st.sidebar.slider('Duty Ratio Range B1', 
                                      0.0, 1.0, (0.0,1.0),step=0.05)
        [DminB2,DmaxB2] = st.sidebar.slider('Duty Ratio Range B2', 
                                      0.0, 1.0, (0.0,1.0),step=0.05)
        st.write("Warning: Function Not Supported")

if function_select == "Core Loss Prediction":
    
    st.title("Princeton MagNet - Core Loss Prediction")
    st.header("Princeton Power Electronics Research Lab, Princeton University")
    st.markdown("""---""")
    st.header("Please provide waveform information")
    # read the necessary information for display
    material_list = ("N27","N49","N87")  
    material_type = st.sidebar.selectbox(
        "Material:",
        material_list
    )
    
    excitation_list = ("Datasheet","Sinusoidal","Triangle","Trapezoidal",
        "Arbitrary")
    excitation_type = st.sidebar.selectbox(
        "Excitation:",
        excitation_list
    )
    
    algorithm_list = ("iGSE","Advanced Analytical","Machine Learning")
    algorithm_type = st.sidebar.selectbox(
        "Algorithm:",
        algorithm_list
    ) 
    
    
    if excitation_type == "Sinusoidal" or excitation_type == "Datasheet":
        Freq = st.slider('Frequency (Hz)', 10000, 300000, step=1000)
        Flux = st.slider('Peak to Peak Flux Density (mT)', 10, 300, step=1)
        Bias = st.slider('DC Bias (mT)', -300, 300, 0, step=10)
        duty_list = np.linspace(0,1,101)
        flux_read = np.multiply(np.sin(np.multiply(duty_list,np.pi*2)),Flux/2)
        flux_list = np.add(flux_read,Bias)
        st.header("Waveform Visualization")
        fig5 = px.line(x=duty_list,y=flux_list,
                      labels={'x':'Duty in a Cycle [%]', 'y':'Flux Density [mT]'})
        st.plotly_chart(fig5, use_container_width=True)
        st.header(material_type+", "+excitation_type+", f="+str(Freq)+" Hz"
                 +", \u0394B="+str(Flux)+" mT"+", Bias="+str(Bias)+" mT")
    
    if excitation_type == "Triangle":
        Freq = st.slider('Frequency (Hz)', 10000, 300000, step=1000)
        Flux = st.slider('Peak to Peak Flux Density (mT)', 10, 300, step=10)
        Duty = st.slider('Duty Ratio', 0.0, 1.0, 0.5, step=0.01)
        Bias = st.slider('DC Bias (mT)', -300, 300, 0, step=10)
        duty_list = [0, Duty, 1]
        flux_read = [0, Flux, 0]
        flux_mean = Flux/2
        flux_diff = Bias-flux_mean
        flux_list = np.add(flux_read,flux_diff)
        
        st.header("Waveform Visualization")
        fig5 = px.line(x=duty_list,y=flux_list,
                      labels={'x':'Duty in A Cycle [%]', 'y':'Flux Density [mT]'})
        st.plotly_chart(fig5, use_container_width=True)
        
        st.header(material_type+", "+excitation_type+", f="+str(Freq)+" Hz"
                 +", \u0394B="+str(Flux)+" mT"+", D="+str(Duty)+", Bias="+str(Bias)+" mT")
            
    if excitation_type == "Trapezoidal":
        Freq = st.slider('Frequency (Hz)', 10000, 300000, step=1000)
        Flux = st.slider('Peak to Peak Flux Density (mT)', 10, 300, step=10)
        Duty1 = st.slider('Duty Ratio 1', 0.0, 1.0, 0.25, step=0.01)
        Duty2 = st.slider('Duty Ratio 2', 0.0, 1.0, 0.5, step=0.01)
        Duty3 = st.slider('Duty Ratio 3', 0.0, 1.0, 0.75, step=0.01)
        Bias = st.slider('DC Bias (mT)', -300, 300, 0, step=10)
        duty_list = [0, Duty1, Duty2, Duty3, 1]
        flux_read = [0, Flux, Flux, 0,  0]
        flux_mean = Flux/2
        flux_diff = Bias-flux_mean
        flux_list = np.add(flux_read,flux_diff)
        
        st.header("Waveform Visualization")
        fig5 = px.line(x=duty_list,y=flux_list,
                      labels={'x':'Duty in a Cycle [%]', 'y':'Flux Density [mT]'})
        st.plotly_chart(fig5, use_container_width=True)
        
        st.header(material_type+", "+excitation_type+", f="+str(Freq)+" Hz"
                 +", \u0394B="+str(Flux)+" mT"+", D1="+str(Duty1)+", D2="+str(Duty2)+", D3="+str(Duty3)+", Bias="+str(Bias)+" mT")
    
    if excitation_type == "Arbitrary":
        Freq = st.slider('Cycle Frequency (Hz)', 10000, 300000, step=1000)
        duty_string = st.text_input('Waveform Pattern Duty in a Cycle (%)', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        flux_string = st.text_input('Waveform Pattern Relative Flux Density (mT)', [0, 10, 20, 10, 20, 30, -10, -30, 10, -10, 0])
        Bias = st.slider('DC Bias (mT)', -300, 300, 0, step=10)
        
        duty_split = re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", duty_string)
        flux_split = re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", flux_string)
        duty_list = [float(i) for i in duty_split]
        # calculate flux based on input value and dc-bias
        flux_read = [float(i) for i in flux_split]
        flux_mean = np.average(flux_read)
        flux_diff = Bias-flux_mean
        flux_list = np.add(flux_read,flux_diff)
        st.header("Waveform Visualization")
        fig5 = px.line(x=duty_list,y=flux_list,
                      labels={'x':'Duty in A Cycle [%]', 'y':'Flux Density [mT]'})
        st.plotly_chart(fig5, use_container_width=True)
    
    core_loss=np.random.rand()
    st.title("Core Loss: "+str(round(core_loss,3))+" kW/m^3")
    
st.markdown("""---""")
st.title("Research Team")
st.image('magnetteam.jpg', width=1000)
st.title("Sponsors")
st.image('sponsor.jpg', width=1000)
st.title("Contact")
st.header("Minjie Chen (minjie@princeton.edu), Haoran Li (haoranli@princeton.edu)")