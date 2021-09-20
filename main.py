import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import pandas as pd
from search import sinesearch
from search import taglsearch
import torch
import torch.nn as nn

from SimPLECS.SimFunctions import *

# Neural network setup
NN_ARCHITECTURE = [3, 15, 15, 9, 1]  # Number of neurons in each layer


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define a fully connected layers model with three inputs (frequency, flux density, duty ratio)
        # and one output (power loss).
        self.layers = nn.Sequential(
            nn.Linear(NN_ARCHITECTURE[0], NN_ARCHITECTURE[1]),
            nn.ReLU(),
            nn.Linear(NN_ARCHITECTURE[1], NN_ARCHITECTURE[2]),
            nn.ReLU(),
            nn.Linear(NN_ARCHITECTURE[2], NN_ARCHITECTURE[3]),
            nn.ReLU(),
            nn.Linear(NN_ARCHITECTURE[3], NN_ARCHITECTURE[4])
        )

    def forward(self, x):
        return self.layers(x)

    # Returns number of trainable parameters in a network
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# setup the user interface
st.set_page_config(page_title="PMagNet", layout="wide")
st.image('pulogo.jpg', width=600)
st.sidebar.image('magnetlogo.jpg', width=300)

st.sidebar.header('PMagNet v1.1 Beta')
function_list = ("Core Loss Database", "Core Loss Prediction")
function_select = st.sidebar.radio("Select one of the two functions",
                                   function_list
                                   )

if function_select == "Core Loss Database":

    st.title("Princeton MagNet - Core Loss Database")
    st.header("Princeton Power Electronics Research Lab, Princeton University")
    st.markdown("""---""")


    @st.cache(allow_output_mutation=True)
    def load_data(data_dir):
        with open(data_dir) as jsonfile:
            Data = json.load(jsonfile)
            NewData = dict()
            NewData['Frequency'] = Data['Frequency']
            NewData['Power_Loss'] = Data['Power_Loss']
            NewData['Flux_Density'] = Data['Flux_Density']
            NewData['Duty_Ratio'] = Data['Duty_Ratio']
        return NewData


    st.sidebar.header("Information for Material A")
    # read the necessary information for display
    material_listA = ("N27", "N49", "N87", "3C90", "3C94")
    material_typeA = st.sidebar.selectbox(
        "Material A:",
        material_listA
    )

    excitation_listA = ("Datasheet", "Sinusoidal", "Triangle", "Trapezoidal")
    excitation_typeA = st.sidebar.selectbox(
        "Excitation A:",
        excitation_listA
    )

    [FminA, FmaxA] = st.sidebar.slider('Frequency Range A (Hz)',
                                       10000, 500000, (10000, 500000), step=1000)
    [BminA, BmaxA] = st.sidebar.slider('Flux Density Range A (mT)',
                                       10, 300, (10, 300), step=1)
    BiasA = st.sidebar.slider('DC Bias A (mT)', -300, 300, 0, step=1)

    if excitation_typeA == "Datasheet" or excitation_typeA == "Sinusoidal":
        st.header("**" + material_typeA + ", " + excitation_typeA + ", f=[" + str(FminA) + "~" + str(FmaxA) + "] Hz"
                  + ", B=[" + str(BminA) + "~" + str(BmaxA) + "] mT" + "**")

        data_dirA = "./Data/Data_" + material_typeA + "_" + excitation_typeA + "_light.json"
        # create a subset of data that meet the rules
        DataA = load_data(data_dirA)
        SubsetA = sinesearch(DataA, FminA, FmaxA, BminA, BmaxA)

        if not SubsetA['Frequency']:
            st.write("Warning: No Data in Range")
        else:
            df = pd.DataFrame(SubsetA)
            col1, col2 = st.beta_columns(2)
            with col1:
                fig1 = px.scatter(df, x=SubsetA['Frequency'], y=SubsetA['Power_Loss'], color=SubsetA['Flux_Density'],
                                  log_x=True, log_y=True, color_continuous_scale=px.colors.sequential.Turbo,
                                  labels={'x': 'Frequency [Hz]', 'y': 'Power Loss [kW/m^3]',
                                          'color': 'Flux Density [mT]'})
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.scatter(df, x=SubsetA['Flux_Density'], y=SubsetA['Power_Loss'], color=SubsetA['Frequency'],
                                  log_x=True, log_y=True, color_continuous_scale=px.colors.sequential.Turbo,
                                  labels={'x': 'Flux Density [mT]', 'y': 'Power Loss [kW/m^3]',
                                          'color': 'Frequency [Hz]'})
                st.plotly_chart(fig2, use_container_width=True)

    if excitation_typeA == "Triangle":

        DutyA = st.sidebar.multiselect("Duty Ratio A", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                       [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        MarginA = st.sidebar.slider('Duty Ratio Margin A', 0.0, 1.0, 0.01, step=0.01)

        st.header("**" + material_typeA + ", " + excitation_typeA + ", f=[" + str(FminA) + "~" + str(FmaxA) + "] Hz"
                  + ", B=[" + str(BminA) + "~" + str(BmaxA) + "] mT" + ", D=" + str(DutyA) + "**")

        # read the corresponding data
        data_dirA = "./Data/Data_" + material_typeA + "_" + excitation_typeA + "_light.json"
        # create a subset of data that meet the rules
        # read the corresponding data
        DataA = load_data(data_dirA)
        SubsetA = taglsearch(DataA, FminA, FmaxA, BminA, BmaxA, DutyA, MarginA)

        if not SubsetA['Frequency']:
            st.write("Warning: No Data in Range")
        else:
            df = pd.DataFrame(SubsetA)
            col1, col2 = st.beta_columns(2)
            with col1:
                fig1 = px.scatter(df, x=SubsetA['Frequency'], y=SubsetA['Power_Loss'], color=SubsetA['Duty_Ratio'],
                                  log_x=True, log_y=True, color_continuous_scale=px.colors.sequential.Turbo,
                                  hover_data=['Flux_Density'],
                                  labels={'x': 'Frequency [Hz]', 'y': 'Power Loss [kW/m^3]', 'color': 'Duty Ratio'})
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.scatter(df, x=SubsetA['Flux_Density'], y=SubsetA['Power_Loss'], color=SubsetA['Duty_Ratio'],
                                  log_x=True, log_y=True, color_continuous_scale=px.colors.sequential.Turbo,
                                  hover_data=['Frequency'],
                                  labels={'x': 'Flux Density [mT]', 'y': 'Power Loss [kW/m^3]', 'color': 'Duty Ratio'})
                st.plotly_chart(fig2, use_container_width=True)

    if excitation_typeA == "Trapezoidal":
        DutyA = st.sidebar.multiselect("Duty Ratio A",
                                       [0.1414, 0.2323, 0.3232, 0.3313, 0.4141, 0.4222, 0.5131, 0.5212, 0.6121],
                                       [0.1414, 0.2323, 0.3232, 0.3313, 0.4141, 0.4222, 0.5131, 0.5212, 0.6121])
        MarginA = st.sidebar.slider('Duty Ratio Margin A', 0.0, 1.0, 0.01, step=0.01)

        st.header("**" + material_typeA + ", " + excitation_typeA + ", f=[" + str(FminA) + "~" + str(FmaxA) + "] Hz"
                  + ", B=[" + str(BminA) + "~" + str(BmaxA) + "] mT" + ", D=" + str(DutyA) + "**")
        st.header("Note: D=0.2332 means **20% Up + 30% Flat + 30% Down + 20% Flat** from left to right")

        # read the corresponding data
        data_dirA = "./Data/Data_" + material_typeA + "_" + "Trapezoidal" + "_light.json"
        # create a subset of data that meet the rules
        # read the corresponding data
        DataA = load_data(data_dirA)
        SubsetA = taglsearch(DataA, FminA, FmaxA, BminA, BmaxA, DutyA, MarginA)

        if not SubsetA['Frequency']:
            st.write("Warning: No Data in Range")
        else:
            df = pd.DataFrame(SubsetA)
            col1, col2 = st.beta_columns(2)
            with col1:
                fig1 = px.scatter(df, x=SubsetA['Frequency'], y=SubsetA['Power_Loss'], color=SubsetA['Duty_Ratio'],
                                  log_x=True, log_y=True, color_continuous_scale=px.colors.sequential.Turbo,
                                  hover_data=['Flux_Density'],
                                  labels={'x': 'Frequency [Hz]', 'y': 'Power Loss [kW/m^3]', 'color': 'Duty Ratio'})
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.scatter(df, x=SubsetA['Flux_Density'], y=SubsetA['Power_Loss'], color=SubsetA['Duty_Ratio'],
                                  log_x=True, log_y=True, color_continuous_scale=px.colors.sequential.Turbo,
                                  hover_data=['Frequency'],
                                  labels={'x': 'Flux Density [mT]', 'y': 'Power Loss [kW/m^3]', 'color': 'Duty Ratio'})
                st.plotly_chart(fig2, use_container_width=True)

    st.sidebar.markdown("""---""")
    st.markdown("""---""")
    ################################################################
    # read the necessary information for display
    st.sidebar.header("Information for Material B")
    material_listB = ("N27", "N49", "N87", "3C90", "3C94")
    material_typeB = st.sidebar.selectbox(
        "Material B:",
        material_listB
    )

    excitation_listB = ("Datasheet", "Sinusoidal", "Triangle", "Trapezoidal")
    excitation_typeB = st.sidebar.selectbox(
        "Excitation B:",
        excitation_listB
    )

    [FminB, FmaxB] = st.sidebar.slider('Frequency Range B (Hz)',
                                       10000, 500000, (10000, 500000), step=1000)
    [BminB, BmaxB] = st.sidebar.slider('Flux Density Range B (mT)',
                                       10, 300, (10, 300), step=1)
    BiasB = st.sidebar.slider('DC Bias B (mT)', -300, 300, 0, step=1)

    if excitation_typeB == "Datasheet" or excitation_typeB == "Sinusoidal":
        st.header("**" + material_typeB + ", " + excitation_typeB + ", f=[" + str(FminB) + "~" + str(FmaxB) + "] Hz"
                  + ", B=[" + str(BminB) + "~" + str(BmaxB) + "] mT" + "**")

        # read the corresponding data
        data_dirB = "./Data/Data_" + material_typeB + "_" + excitation_typeB + "_light.json"
        # create a subset of data that meet the rules
        DataB = load_data(data_dirB)
        SubsetB = sinesearch(DataB, FminB, FmaxB, BminB, BmaxB)

        if not SubsetB['Frequency']:
            st.write("Warning: No Data in Range")
        else:
            df = pd.DataFrame(SubsetB)
            col1, col2 = st.beta_columns(2)
            with col1:
                fig3 = px.scatter(df, x=SubsetB['Frequency'], y=SubsetB['Power_Loss'], color=SubsetB['Flux_Density'],
                                  log_x=True, log_y=True, color_continuous_scale=px.colors.sequential.Turbo,
                                  labels={'x': 'Frequency [Hz]', 'y': 'Power Loss [kW/m^3]',
                                          'color': 'Flux Density [mT]'})
                st.plotly_chart(fig3, use_container_width=True)
            with col2:
                fig4 = px.scatter(df, x=SubsetB['Flux_Density'], y=SubsetB['Power_Loss'], color=SubsetB['Frequency'],
                                  log_x=True, log_y=True, color_continuous_scale=px.colors.sequential.Turbo,
                                  labels={'x': 'Flux Density [mT]', 'y': 'Power Loss [kW/m^3]',
                                          'color': 'Frequency [Hz]'})
                st.plotly_chart(fig4, use_container_width=True)

    if excitation_typeB == "Triangle":
        DutyB = st.sidebar.multiselect("Duty Ratio B", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                       [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        MarginB = st.sidebar.slider('Duty Ratio Margin B', 0.0, 1.0, 0.01, step=0.01)

        st.header("**" + material_typeB + ", " + excitation_typeB + ", f=[" + str(FminB) + "~" + str(FmaxB) + "] Hz"
                  + ", B=[" + str(BminB) + "~" + str(BmaxB) + "] mT" + ", D=" + str(DutyB) + "**")

        # read the corresponding data
        data_dirB = "./Data/Data_" + material_typeB + "_" + excitation_typeB + "_light.json"
        # create a subset of data that meet the rules
        # read the corresponding data
        DataB = load_data(data_dirB)
        SubsetB = taglsearch(DataB, FminB, FmaxB, BminB, BmaxB, DutyB, MarginB)

        if not SubsetB['Frequency']:
            st.write("Warning: No Data in Range")
        else:
            df = pd.DataFrame(SubsetB)
            col1, col2 = st.beta_columns(2)
            with col1:
                fig3 = px.scatter(df, x=SubsetB['Frequency'], y=SubsetB['Power_Loss'], color=SubsetB['Duty_Ratio'],
                                  log_x=True, log_y=True, color_continuous_scale=px.colors.sequential.Turbo,
                                  hover_data=['Flux_Density'],
                                  labels={'x': 'Frequency [Hz]', 'y': 'Power Loss [kW/m^3]', 'color': 'Duty Ratio'})
                st.plotly_chart(fig3, use_container_width=True)
            with col2:
                fig4 = px.scatter(df, x=SubsetB['Flux_Density'], y=SubsetB['Power_Loss'], color=SubsetB['Duty_Ratio'],
                                  log_x=True, log_y=True, color_continuous_scale=px.colors.sequential.Turbo,
                                  hover_data=['Frequency'],
                                  labels={'x': 'Flux Density [mT]', 'y': 'Power Loss [kW/m^3]', 'color': 'Duty Ratio'})
                st.plotly_chart(fig4, use_container_width=True)

    if excitation_typeB == "Trapezoidal":
        DutyB = st.sidebar.multiselect("Duty Ratio B",
                                       [0.1414, 0.2323, 0.3232, 0.3313, 0.4141, 0.4222, 0.5131, 0.5212, 0.6121],
                                       [0.1414, 0.2323, 0.3232, 0.3313, 0.4141, 0.4222, 0.5131, 0.5212, 0.6121])
        MarginB = st.sidebar.slider('Duty Ratio Margin B', 0.0, 1.0, 0.01, step=0.01)

        st.header("**" + material_typeB + ", " + excitation_typeB + ", f=[" + str(FminB) + "~" + str(FmaxB) + "] Hz"
                  + ", B=[" + str(BminB) + "~" + str(BmaxB) + "] mT" + ", D=" + str(DutyB) + "**")
        st.header("Note: D=0.1414 means **10% Up + 40% Flat + 10% Down + 40% Flat** from left to right")

        # read the corresponding data
        data_dirB = "./Data/Data_" + material_typeB + "_" + "Trapezoidal" + "_light.json"
        # create a subset of data that meet the rules
        # read the corresponding data
        DataB = load_data(data_dirB)
        SubsetB = taglsearch(DataB, FminB, FmaxB, BminB, BmaxB, DutyB, MarginB)

        if not SubsetB['Frequency']:
            st.write("Warning: No Data in Range")
        else:
            df = pd.DataFrame(SubsetB)
            col1, col2 = st.beta_columns(2)
            with col1:
                fig3 = px.scatter(df, x=SubsetB['Frequency'], y=SubsetB['Power_Loss'], color=SubsetB['Duty_Ratio'],
                                  log_x=True, log_y=True, color_continuous_scale=px.colors.sequential.Turbo,
                                  hover_data=['Flux_Density'],
                                  labels={'x': 'Frequency [Hz]', 'y': 'Power Loss [kW/m^3]', 'color': 'Duty Ratio'})
                st.plotly_chart(fig3, use_container_width=True)
            with col2:
                fig4 = px.scatter(df, x=SubsetB['Flux_Density'], y=SubsetB['Power_Loss'], color=SubsetB['Duty_Ratio'],
                                  log_x=True, log_y=True, color_continuous_scale=px.colors.sequential.Turbo,
                                  hover_data=['Frequency'],
                                  labels={'x': 'Flux Density [mT]', 'y': 'Power Loss [kW/m^3]', 'color': 'Duty Ratio'})
                st.plotly_chart(fig4, use_container_width=True)

#######################################################################
#######################################################################

if function_select == "Core Loss Prediction":
    # core_loss=np.random.rand()
    st.title("Princeton MagNet - Core Loss Prediction")
    st.header("Princeton Power Electronics Research Lab, Princeton University")
    st.markdown("""---""")
    
    fluxplot_list = [10, 12, 15, 17, 20, 25, 30, 40, 50, 60, 80, 100, 120, 160, 200, 240, 260, 300]
    freqplot_list = [10000, 15000, 20000, 30000, 40000, 50000, 60000, 80000, 100000, 120000, 160000, 200000, 240000, 320000, 400000, 480000]
    dutyplot_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # function to calculate the iGSE core loss
    def calc_iGSE_sine(Freq,Flux,ki,alpha,beta):
        time = np.linspace(0, 10e-9 * 10000, 10001)
        B = np.multiply( np.sin(np.multiply(np.multiply(time, Freq), np.pi * 2)), Flux / 2)
        dt = time[1] - time[0]
        dBdt = np.gradient(B, dt)
        T = time[-1] - time[0]
        core_loss = 1 / T * np.trapz(ki * (np.abs(dBdt) ** alpha) * (Flux ** (beta - alpha)), time)
        return core_loss
    
        # function to calculate the iGSE core loss
    def calc_ML_sine(Freq,Flux):
        core_loss = 10.0 ** neural_network(torch.from_numpy(np.array([np.log10(float(Freq)), np.log10(float(Flux/2)),
                                                                          0.5]))).item()
        return core_loss
    
        # function to calculate the iGSE core loss
    def calc_iGSE_tagl(Freq,Flux,flux_list,duty_list,ki,alpha,beta):
        time = np.linspace(0, 1 / Freq, 10001)
        T = time[-1] - time[0]
        B = np.interp(time, np.multiply(duty_list, T), flux_list)
        dt = time[1] - time[0]
        dBdt = np.gradient(B, dt)
        core_loss = 1 / T * np.trapz(ki * (np.abs(dBdt) ** alpha) * (Flux ** (beta - alpha)), time)
        return core_loss
    
    def calc_ML_tagl(Freq,Flux,Duty):
        core_loss = 10.0 ** neural_network(torch.from_numpy(np.array([np.log10(float(Freq)), np.log10(float(Flux/2)),
                                                                          Duty]))).item()
        return core_loss
    
    def calc_iGSE_trapz(Freq,Flux,flux_list,duty_list,ki,alpha,beta):
        time = np.linspace(0, 1 / Freq, 10001)
        T = time[-1] - time[0]
        B = np.interp(time, np.multiply(duty_list, T), flux_list)
        dt = time[1] - time[0]
        dBdt = np.gradient(B, dt)
        core_loss = 1 / T * np.trapz(ki * (np.abs(dBdt) ** alpha) * (Flux ** (beta - alpha)), time)
        return core_loss
    
    def calc_ML_trapz(Freq,Flux,flux_list,duty_list):
        core_loss = np.random.rand()  # TBD
        return core_loss
    
    def calc_iGSE_arbit(Freq,Flux,flux_list,duty_list,ki,alpha,beta):
        time = np.linspace(0, 1 / Freq, 10001)
        T = time[-1] - time[0]
        B = np.interp(time, np.multiply(duty_list, T), flux_list)
        dt = time[1] - time[0]
        dBdt = np.gradient(B, dt)
        core_loss = 1 / T * np.trapz(ki * (np.abs(dBdt) ** alpha) * (flux_delta ** (beta - alpha)), time)
        return core_loss
    
    def calc_ML_arbit(Freq,Flux,flux_list,duty_list):
        core_loss = np.random.rand()  # TBD
        return core_loss
    
    # read the necessary information for display
    material_list = ("N27", "N49", "N87", "3C90", "3C94")
    material_type = st.sidebar.selectbox(
        "Material:",
        material_list
    )

    excitation_list = ("Datasheet", "Sinusoidal", "Triangle", "Trapezoidal",
                       "Arbitrary", "Simulated")
    excitation_type = st.sidebar.selectbox(
        "Excitation:",
        excitation_list
    )

    algorithm_list = ("iGSE", "Advanced Analytical", "Machine Learning")
    algorithm_type = st.sidebar.selectbox(
        "Algorithm:",
        algorithm_list
    )


    # Load the iGSE parameters and NN models for different materials
    if material_type == "N27":
        alpha = 1.09
        beta = 2.44
        ki = 4.88e-10

    if material_type == "N87":
        alpha = 1.43
        beta = 2.49
        ki = 5.77e-12

    if material_type == "N49":
        alpha = 1.27
        beta = 3.17
        ki = 1.18e-12

    use_GPU = False
    if use_GPU and not torch.cuda.is_available():
        raise ValueError("GPU not detected but CONFIG.USE_GPU is set to True.")
    device = torch.device("cuda" if use_GPU else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_GPU else {}

    neural_network = Net().double().to(device)
    state_dict = torch.load('Models/' + material_type + '.sd')
    neural_network.load_state_dict(state_dict, strict=True)
    neural_network.eval()

    if excitation_type == "Sinusoidal" or excitation_type == "Datasheet":
        
        col1, col2 = st.beta_columns(2)
        with col1:
            st.header("Please provide waveform information")
            Freq = st.slider('Frequency (Hz)', 10000, 500000, 250000, step=1000)
            Flux = st.slider('Peak to Peak Flux Density (mT)', 10, 300, 150, step=1)
            Bias = st.slider('DC Bias (mT)', -300, 300, 0, step=10)
            duty_list = np.linspace(0, 1, 101)
            flux_read = np.multiply(np.sin(np.multiply(duty_list, np.pi * 2)), Flux / 2)
            flux_list = np.add(flux_read, Bias)
            
        with col2:
            st.header("Waveform Visualization")
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=duty_list, y=flux_list,
                                  line=dict(color='firebrick', width=4)))
            fig5.update_layout(xaxis_title='Duty in a Cycle',
                           yaxis_title='Flux Density [mT]')
            st.plotly_chart(fig5, use_container_width=True)
        st.header(material_type + ", " + excitation_type + ", f=" + str(Freq) + " Hz"
                  + ", \u0394B=" + str(Flux) + " mT" + ", Bias=" + str(Bias) + " mT")
        if algorithm_type == "iGSE":
            core_loss = calc_iGSE_sine(Freq,Flux,ki,alpha,beta)
            st.title("iGSE Core Loss: " + str(round(core_loss, 3)) + " kW/m^3")
        if algorithm_type == "Machine Learning":
            core_loss = calc_ML_sine(Freq,Flux)
            st.title("ML Core Loss: " + str(round(core_loss, 3)) + " kW/m^3")
            
        iGSE_coreloss_fluxplot = [calc_iGSE_sine(Freq,i,ki,alpha,beta) for i in fluxplot_list]
        iGSE_coreloss_freqplot = [calc_iGSE_sine(i,Flux,ki,alpha,beta) for i in freqplot_list]
        ML_coreloss_fluxplot = [calc_ML_sine(Freq,i) for i in fluxplot_list]
        ML_coreloss_freqplot = [calc_ML_sine(i,Flux) for i in freqplot_list]
        flux_coreloss = {'Sweep': fluxplot_list, 'iGSE': iGSE_coreloss_fluxplot, 'ML': ML_coreloss_fluxplot}
        freq_coreloss = {'Sweep': freqplot_list, 'iGSE': iGSE_coreloss_freqplot, 'ML': ML_coreloss_freqplot}
        col1, col2 = st.beta_columns(2)
        with col1:
            st.header("Core Loss with Fixed Flux Density " + str(Flux) + " mT")
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(name="iGSE", x=freq_coreloss['Sweep'],  y=freq_coreloss['iGSE'],
                                  line=dict(color='firebrick', width=4)))
            fig5.add_trace(go.Scatter(name="ML",x=freq_coreloss['Sweep'],  y=freq_coreloss['ML'],
                                  line=dict(color='darkslategrey', width=4)))
            fig5.update_layout(xaxis_title='Frequency [Hz]',
                           yaxis_title='Power Loss [kW/m^3]')
            fig5.update_xaxes(type="log")
            fig5.update_yaxes(type="log")
            st.plotly_chart(fig5, use_container_width=True)
        with col2:
            st.header("Core Loss with Fixed Frequency " + str(Freq) + " Hz")
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(name="iGSE", x=flux_coreloss['Sweep'],  y=flux_coreloss['iGSE'],
                                  line=dict(color='firebrick', width=4)))
            fig6.add_trace(go.Scatter(name="ML",x=flux_coreloss['Sweep'],  y=flux_coreloss['ML'],
                                  line=dict(color='darkslategrey', width=4)))
            fig6.update_layout(xaxis_title='Flux Density [mT]',
                           yaxis_title='Power Loss [kW/m^3]')
            fig6.update_xaxes(type="log")
            fig6.update_yaxes(type="log")
            st.plotly_chart(fig6, use_container_width=True)
            
    if excitation_type == "Triangle":
        col1, col2 = st.beta_columns(2)
        with col1:
            st.header("Please provide waveform information")
            Freq = st.slider('Frequency (Hz)', 10000, 500000, 250000, step=1000)
            Flux = st.slider('Peak to Peak Flux Density (mT)', 10, 300, 150, step=10)
            Duty = st.slider('Duty Ratio', 0.0, 1.0, 0.5, step=0.01)
            Bias = st.slider('DC Bias (mT)', -300, 300, 0, step=10)
            duty_list = [0, Duty, 1]
            flux_read = [0, Flux, 0]
            flux_mean = Flux / 2
            flux_diff = Bias - flux_mean
            flux_list = np.add(flux_read, flux_diff)
        with col2:
            st.header("Waveform Visualization")
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=duty_list, y=flux_list,
                                  line=dict(color='firebrick', width=4)))
            fig5.update_layout(xaxis_title='Duty in a Cycle',
                           yaxis_title='Flux Density [mT]')

            st.plotly_chart(fig5, use_container_width=True)

        st.header(material_type + ", " + excitation_type + ", f=" + str(Freq) + " Hz"
                  + ", \u0394B=" + str(Flux) + " mT" + ", D=" + str(Duty) + ", Bias=" + str(Bias) + " mT")
        if algorithm_type == "iGSE":
            core_loss = calc_iGSE_tagl(Freq,Flux,flux_list,duty_list,ki,alpha,beta)
            st.title("iGSE Core Loss: " + str(round(core_loss, 3)) + " kW/m^3")
        if algorithm_type == "Machine Learning":
            core_loss = calc_ML_tagl(Freq,Flux,Duty)
            st.title("ML Core Loss: " + str(round(core_loss, 3)) + " kW/m^3")

        iGSE_coreloss_fluxplot = [calc_iGSE_tagl(Freq,i,flux_list,duty_list,ki,alpha,beta) for i in fluxplot_list]
        iGSE_coreloss_freqplot = [calc_iGSE_tagl(i,Flux,flux_list,duty_list,ki,alpha,beta) for i in freqplot_list]
        iGSE_coreloss_dutyplot = [calc_iGSE_tagl(Freq,Flux,flux_list,[0, i, 1],ki,alpha,beta) for i in dutyplot_list]
        
        
        ML_coreloss_fluxplot = [calc_ML_tagl(Freq,i,Duty) for i in fluxplot_list]
        ML_coreloss_freqplot = [calc_ML_tagl(i,Flux,Duty) for i in freqplot_list]
        ML_coreloss_dutyplot = [calc_ML_tagl(Freq,Flux,i) for i in dutyplot_list]
        
        flux_coreloss = {'Sweep': fluxplot_list, 'iGSE': iGSE_coreloss_fluxplot, 'ML': ML_coreloss_fluxplot}
        freq_coreloss = {'Sweep': freqplot_list, 'iGSE': iGSE_coreloss_freqplot, 'ML': ML_coreloss_freqplot}
        duty_coreloss = {'Sweep': dutyplot_list, 'iGSE': iGSE_coreloss_dutyplot, 'ML': ML_coreloss_dutyplot}
        
        col1, col2, col3 = st.beta_columns(3)
        with col1:
            st.header("Core Loss with F Sweep at " + str(Flux) + " mT and D=" + str(Duty))
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(name="iGSE", x=freq_coreloss['Sweep'],  y=freq_coreloss['iGSE'],
                                  line=dict(color='firebrick', width=4)))
            fig5.add_trace(go.Scatter(name="ML",x=freq_coreloss['Sweep'],  y=freq_coreloss['ML'],
                                  line=dict(color='darkslategrey', width=4)))
            fig5.update_layout(xaxis_title='Frequency [Hz]',
                           yaxis_title='Power Loss [kW/m^3]')
            fig5.update_xaxes(type="log")
            fig5.update_yaxes(type="log")
            st.plotly_chart(fig5, use_container_width=True)
        with col2:
            st.header("Core Loss with B Sweep at " + str(Freq) + " Hz and D=" + str(Duty))
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(name="iGSE", x=flux_coreloss['Sweep'],  y=flux_coreloss['iGSE'],
                                  line=dict(color='firebrick', width=4)))
            fig6.add_trace(go.Scatter(name="ML",x=flux_coreloss['Sweep'],  y=flux_coreloss['ML'],
                                  line=dict(color='darkslategrey', width=4)))
            fig6.update_layout(xaxis_title='Flux Density [mT]',
                           yaxis_title='Power Loss [kW/m^3]')
            fig6.update_xaxes(type="log")
            fig6.update_yaxes(type="log")
            st.plotly_chart(fig6, use_container_width=True)
        with col3:
            st.header("Core Loss with D Sweep at " + str(Freq) + " Hz and " + str(Flux) + " mT")
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(name="iGSE", x=duty_coreloss['Sweep'],  y=duty_coreloss['iGSE'],
                                  line=dict(color='firebrick', width=4)))
            fig7.add_trace(go.Scatter(name="ML",x=duty_coreloss['Sweep'],  y=duty_coreloss['ML'],
                                  line=dict(color='darkslategrey', width=4)))
            fig7.update_layout(xaxis_title='Duty Ratio',
                           yaxis_title='Power Loss [kW/m^3]')
            fig7.update_yaxes(type="log")
            st.plotly_chart(fig7, use_container_width=True)

    if excitation_type == "Trapezoidal":
        col1, col2 = st.beta_columns(2)
        with col1:
            st.header("Please provide waveform information")
            Freq = st.slider('Frequency (Hz)', 10000, 500000, step=1000)
            Flux = st.slider('Peak to Peak Flux Density (mT)', 10, 300, step=10)
            Duty1 = st.slider('Duty Ratio 1', 0.0, 1.0, 0.25, step=0.01)
            Duty2 = st.slider('Duty Ratio 2', 0.0, 1.0, 0.5, step=0.01)
            Duty3 = st.slider('Duty Ratio 3', 0.0, 1.0, 0.75, step=0.01)
            Bias = st.slider('DC Bias (mT)', -300, 300, 0, step=10)
            duty_list = [0, Duty1, Duty2, Duty3, 1]
            flux_read = [0, Flux, Flux, 0, 0]
            flux_mean = Flux / 2
            flux_diff = Bias - flux_mean
            flux_list = np.add(flux_read, flux_diff)
        with col2:
            st.header("Waveform Visualization")
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=duty_list, y=flux_list,
                                  line=dict(color='firebrick', width=4)))
            fig5.update_layout(xaxis_title='Duty in a Cycle',
                           yaxis_title='Flux Density [mT]')

            st.plotly_chart(fig5, use_container_width=True)

        st.header(material_type + ", " + excitation_type + ", f=" + str(Freq) + " Hz"
                  + ", \u0394B=" + str(Flux) + " mT" + ", D1=" + str(Duty1) + ", D2=" + str(Duty2) + ", D3=" + str(
            Duty3) + ", Bias=" + str(Bias) + " mT")
        if algorithm_type == "iGSE":
            core_loss = calc_iGSE_trapz(Freq,Flux,flux_list,duty_list,ki,alpha,beta)
            st.title("iGSE Core Loss: " + str(round(core_loss, 3)) + " kW/m^3")
        if algorithm_type == "Machine Learning":
            core_loss = calc_ML_trapz(Freq,Flux,flux_list,duty_list)  # TBD
            st.title("ML Core Loss: " + str(round(core_loss, 3)) + " kW/m^3")

        iGSE_coreloss_fluxplot = [calc_iGSE_trapz(Freq,i,flux_list,duty_list,ki,alpha,beta) for i in fluxplot_list]
        iGSE_coreloss_freqplot = [calc_iGSE_trapz(i,Flux,flux_list,duty_list,ki,alpha,beta) for i in freqplot_list]
        ML_coreloss_fluxplot = [calc_ML_trapz(Freq,i,flux_list,duty_list) for i in fluxplot_list]
        ML_coreloss_freqplot = [calc_ML_trapz(i,Flux,flux_list,duty_list) for i in freqplot_list]
        flux_coreloss = {'Sweep': fluxplot_list, 'iGSE': iGSE_coreloss_fluxplot, 'ML': ML_coreloss_fluxplot}
        freq_coreloss = {'Sweep': freqplot_list, 'iGSE': iGSE_coreloss_freqplot, 'ML': ML_coreloss_freqplot}
        col1, col2 = st.beta_columns(2)
        with col1:
            st.header("Core Loss with Fixed Flux Density " + str(Flux) + " mT")
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(name="iGSE", x=freq_coreloss['Sweep'],  y=freq_coreloss['iGSE'],
                                  line=dict(color='firebrick', width=4)))
            fig5.add_trace(go.Scatter(name="ML",x=freq_coreloss['Sweep'],  y=freq_coreloss['ML'],
                                  line=dict(color='darkslategrey', width=4)))
            fig5.update_layout(xaxis_title='Frequency [Hz]',
                           yaxis_title='Power Loss [kW/m^3]')
            fig5.update_xaxes(type="log")
            fig5.update_yaxes(type="log")
            st.plotly_chart(fig5, use_container_width=True)
        with col2:
            st.header("Core Loss with Fixed Frequency " + str(Freq) + " Hz")
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(name="iGSE", x=flux_coreloss['Sweep'],  y=flux_coreloss['iGSE'],
                                  line=dict(color='firebrick', width=4)))
            fig6.add_trace(go.Scatter(name="ML",x=flux_coreloss['Sweep'],  y=flux_coreloss['ML'],
                                  line=dict(color='darkslategrey', width=4)))
            fig6.update_layout(xaxis_title='Flux Density [mT]',
                           yaxis_title='Power Loss [kW/m^3]')
            fig6.update_xaxes(type="log")
            fig6.update_yaxes(type="log")
            st.plotly_chart(fig6, use_container_width=True)

    if excitation_type == "Arbitrary":
        col1, col2 = st.beta_columns(2)
        with col1:
            Freq = st.slider('Cycle Frequency (Hz)', 10000, 500000, step=1000)
            duty_string = st.text_input('Waveform Pattern Duty in a Cycle (%)',
                                    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            flux_string = st.text_input('Waveform Pattern Relative Flux Density (mT)',
                                    [0, 10, 20, 10, 20, 30, -10, -30, 10, -10, 0])
            Bias = st.slider('DC Bias (mT)', -300, 300, 0, step=10)

            duty_split = re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", duty_string)
            flux_split = re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", flux_string)
            duty_list = [float(i) for i in duty_split]
            # calculate flux based on input value and dc-bias
            flux_read = [float(i) for i in flux_split]
            flux_mean = np.average(flux_read)
            flux_diff = Bias - flux_mean
            flux_list = np.add(flux_read, flux_diff)
            flux_delta= np.amax(flux_read)-np.amin(flux_read)
        with col2:  
            st.header("Waveform Visualization")
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=duty_list, y=flux_list,
                                  line=dict(color='firebrick', width=4)))
            fig5.update_layout(xaxis_title='Duty in a Cycle',
                           yaxis_title='Flux Density [mT]')
            st.plotly_chart(fig5, use_container_width=True)

        if algorithm_type == "iGSE":
            core_loss = calc_iGSE_arbit(Freq,flux_delta,flux_list,duty_list,ki,alpha,beta)
            st.title("iGSE Core Loss: " + str(round(core_loss, 3)) + " kW/m^3")
        if algorithm_type == "Machine Learning":
            core_loss = np.random.rand()  # TBD
            st.title("ML Core Loss: " + str(round(core_loss, 3)) + " kW/m^3")

    if excitation_type == "Simulated":
        core_loss = SimulationPLECS(material_type, algorithm_type)

st.markdown("""---""")
st.title("Research Collaborators")
st.image('magnetteam.jpg', width=1000)
st.title("Sponsors")
st.image('sponsor.jpg', width=1000)
st.title("Website Contributor")
st.header("Minjie Chen (minjie@princeton.edu), Haoran Li (haoranli@princeton.edu)")
st.header("Evan Dogariu (edogariu@princeton.edu), Vineet Bansal (vineetb@princeton.edu)")
