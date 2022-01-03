import streamlit as st
import numpy as np
import pandas as pd
import os

from magnet.simplecs.classes import CircuitModel, MagModel, CoreMaterial


def SimulationPLECS(material, algorithm):
    path = os.path.dirname(os.path.realpath(__file__))

    # Select topology
    topology_list = ("Buck", "Boost", "Flyback", "DAB")
    topology_type = st.selectbox(
        "Topology:",
        topology_list,
        key=f'Topology {material} {algorithm}'
    )

    # Circuit model instance
    circuit = CircuitModel(topology_type)

    # Display schematic
    circuit.displaySch(path)

    # Circuit parameters
    Param = {
        'Vi': 0,
        'Vo': 0,
        'Ro': 0,
        'Lk': 0,
        'fsw': 0,
        'duty': 0,
        'ph': 0
    }

    st.header("Circuit parameters")
    col1, col2 = st.columns(2)
    with col1:
        Param['Vi'] = st.number_input("Voltage input [V]", min_value=0., max_value=1000., value=400., step=10.,
                                      key=f'Vi {material} {algorithm}')
        Param['R'] = st.number_input("Load resistor [Ω]", min_value=0., max_value=1e6, value=100., step=10.,
                                     key=f'R {material} {algorithm}')
        if topology_type == "DAB":
            Param['Lk'] = st.number_input("Serial inductor [μH]", min_value=0., max_value=1000., value=50., step=1.,
                                          key=f'Lk {material} {algorithm}')*1e-6
    with col2:
        Param['fsw'] = st.number_input("Switching frequency [Hz]", min_value=1e3, max_value=1e6, value=10e3, step=1.,
                                       key=f'fsw {material} {algorithm}')
        if topology_type == "DAB":
            Param['ph'] = st.number_input("Duty cycle [ ]", min_value=0., max_value=1., value=0.5, step=0.1,
                                          key=f'ph {material} {algorithm}')
        else:
            Param['duty'] = st.number_input("Duty cycle [ ]", min_value=0., max_value=1., value=0.5, step=0.1,
                                            key=f'duty {material} {algorithm}')
    # Assign the inputs to the simulation parameter structure
    circuit.setParam(Param)

    # Core parameters
    Param_mag = {
        'lc': 0,
        'Ac': 0,
        'lg': 0,
        'Np': 0,
        'Ns': 0
    }

    if topology_type == "Flyback" or topology_type == "DAB":
        mag = MagModel("Toroid_2W")
    else:
        mag = MagModel("Toroid")

    # Display geometry
    mag.displaySch(path)

    st.header("Core geometry")
    col1, col2 = st.columns(2)
    with col1:
        Param_mag['lc'] = st.number_input("Length of core [mm]", min_value=0., max_value=1000., value=100., step=10.,
                                          key=f'Lc {material} {algorithm}')*1e-3
        Param_mag['Ac'] = st.number_input("Cross section [mm2]", min_value=0., max_value=1000., value=600., step=10.,
                                          key=f'Ac {material} {algorithm}')*1e-6
        Param_mag['lg'] = st.number_input("Length of gap [mm]", min_value=0., max_value=1000., value=1., step=1.,
                                          key=f'lg {material} {algorithm}')*1e-3
    with col2:
        Param_mag['Np'] = st.number_input("Turns number primary [ ]", min_value=0., max_value=1000., value=8., step=1.,
                                          key=f'Np {material} {algorithm}')
        if topology_type == "Flyback" or topology_type == "DAB":
            Param_mag['Ns'] = st.number_input("Turns number secondary [ ]", min_value=0., max_value=1000., value=8.,
                                              step=1., key=f'Ns {material} {algorithm}')

    # Assign the inputs to the simulation parameter structure
    mag.setParam(Param_mag)

    # Steinmetz Parameters
    st.header("Material parameters")

    Param_material = {
        'mu_r': 6500,
        'iGSE_ki': 8.41,
        'iGSE_alpha': 1.09,
        'iGSE_beta': 2.16
    }

    material = CoreMaterial("N87")
    material.setParam(Param_material)

    df = pd.DataFrame(
        np.array([[material.mu_r, material.iGSE_ki, material.iGSE_alpha, material.iGSE_beta]]),
        columns=["μr", "ki", "α", "β"]
    )
    st.table(df)

    # Simulate and obtain the data
    result = st.button("Simulate", key=f'Simulate {material} {algorithm}')
    Ploss = 0
    circuit.setMagModel(mag, material)
    if result:
        Ploss = circuit.steadyRun(path)
        circuit.displayWfm()

    return Ploss
