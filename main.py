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

# from SimPLECS.SimFunctions import *

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

st.sidebar.header('PMagNet v1.0 Beta')
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
    material_listA = ("N27", "N49", "N87")
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
    material_listB = ("N27", "N49", "N87")
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

if function_select == "Core Loss Prediction":
    # core_loss=np.random.rand()
    st.title("Princeton MagNet - Core Loss Prediction")
    st.header("Princeton Power Electronics Research Lab, Princeton University")
    st.markdown("""---""")
    st.header("Please provide waveform information")
    # read the necessary information for display
    material_list = ("N27", "N49", "N87")
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
        Freq = st.slider('Frequency (Hz)', 10000, 500000, step=1000)
        Flux = st.slider('Peak to Peak Flux Density (mT)', 10, 300, step=1)
        Bias = st.slider('DC Bias (mT)', -300, 300, 0, step=10)
        duty_list = np.linspace(0, 1, 101)
        flux_read = np.multiply(np.sin(np.multiply(duty_list, np.pi * 2)), Flux / 2)
        flux_list = np.add(flux_read, Bias)
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
            time = np.linspace(0, 10e-9 * 10000, 10001)
            B = 10 * np.sin(np.multiply(np.multiply(time, Freq), np.pi * 2))
            dt = time[1] - time[0]
            dBdt = np.gradient(B, dt)
            T = time[-1] - time[0]
            core_loss = 1 / T * np.trapz(ki * (np.abs(dBdt) ** alpha) * (Flux ** (beta - alpha)), time)
        if algorithm_type == "Machine Learning":
            core_loss = 10.0 ** neural_network(torch.from_numpy(np.array([np.log10(float(Freq)), np.log10(float(Flux)),
                                                                          0.5]))).item()

    if excitation_type == "Triangle":
        Freq = st.slider('Frequency (Hz)', 10000, 500000, step=1000)
        Flux = st.slider('Peak to Peak Flux Density (mT)', 10, 300, step=10)
        Duty = st.slider('Duty Ratio', 0.0, 1.0, 0.5, step=0.01)
        Bias = st.slider('DC Bias (mT)', -300, 300, 0, step=10)
        duty_list = [0, Duty, 1]
        flux_read = [0, Flux, 0]
        flux_mean = Flux / 2
        flux_diff = Bias - flux_mean
        flux_list = np.add(flux_read, flux_diff)

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
            time = np.linspace(0, 1 / Freq, 10001)
            T = time[-1] - time[0]
            B = np.interp(time, np.multiply(duty_list, T), flux_list)
            dt = time[1] - time[0]
            dBdt = np.gradient(B, dt)
            core_loss = 1 / T * np.trapz(ki * (np.abs(dBdt) ** alpha) * (Flux ** (beta - alpha)), time)
        if algorithm_type == "Machine Learning":
            core_loss = 10.0 ** neural_network(torch.from_numpy(np.array([np.log10(float(Freq)), np.log10(float(Flux)),
                                                                          Duty]))).item()

    if excitation_type == "Trapezoidal":
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
            time = np.linspace(0, 1 / Freq, 10001)
            T = time[-1] - time[0]
            B = np.interp(time, np.multiply(duty_list, T), flux_list)
            dt = time[1] - time[0]
            dBdt = np.gradient(B, dt)
            core_loss = 1 / T * np.trapz(ki * (np.abs(dBdt) ** alpha) * (Flux ** (beta - alpha)), time)
        if algorithm_type == "Machine Learning":
            core_loss = np.random.rand()  # TBD

    if excitation_type == "Arbitrary":
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
        st.header("Waveform Visualization")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=duty_list, y=flux_list,
                                  line=dict(color='firebrick', width=4)))
        fig5.update_layout(xaxis_title='Duty in a Cycle',
                           yaxis_title='Flux Density [mT]')
        st.plotly_chart(fig5, use_container_width=True)

        if algorithm_type == "iGSE":
            time = np.linspace(0, 1 / Freq, 10001)
            T = time[-1] - time[0]
            B = np.interp(time, np.multiply(duty_list, T), flux_list)
            dt = time[1] - time[0]
            dBdt = np.gradient(B, dt)
            core_loss = 1 / T * np.trapz(ki * (np.abs(dBdt) ** alpha) * (Flux ** (beta - alpha)), time)
        if algorithm_type == "Machine Learning":
            core_loss = np.random.rand()  # TBD

    if excitation_type == "Simulated":
        core_loss = SimulationPLECS(material_type, algorithm_type)

    st.title("Core Loss: " + str(round(core_loss, 3)) + " kW/m^3")

st.markdown("""---""")
st.title("Research Collaborators")
st.image('magnetteam.jpg', width=1000)
st.title("Sponsors")
st.image('sponsor.jpg', width=1000)
st.title("Contact")
st.header("Minjie Chen (minjie@princeton.edu), Haoran Li (haoranli@princeton.edu)")
