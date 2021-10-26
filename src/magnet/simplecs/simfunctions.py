import streamlit as st
import numpy as np
import pandas as pd
import os

from magnet.simplecs.classes import CircuitModel, MagModel, CoreMaterial


def SimulationPLECS(material_type, algorithm_type):
    path = os.path.dirname(os.path.realpath(__file__))

    # Select topology
    topology_list = ("Buck","Boost","Flyback","DAB")
    topology_type = st.selectbox(
        "Topology:",
        topology_list
    )

    # Circuit model instance
    circuit = CircuitModel(topology_type)

    # Display schematic
    circuit.displaySch(path)

    # Circuit parameters
    Param = {}
    Param['Vi'] = 0
    Param['Vo'] = 0
    Param['Ro'] = 0
    Param['Lk'] = 0
    Param['fsw'] = 0
    Param['duty'] = 0
    Param['ph'] = 0

    st.header("Circuit parameters")
    col1, col2 = st.beta_columns(2)
    with col1:
        Param['Vi'] = st.number_input("Voltage input [V]", min_value = 0., max_value = 1000., value = 400., step = 10.)
        Param['R'] = st.number_input("Load resistor [Ω]", min_value = 0., max_value = 1e6, value = 100., step = 10.)
        if topology_type == "DAB":
            Param['Lk'] = st.number_input("Serial inductor [μH]", min_value = 0., max_value = 1000., value = 50., step = 1.)*1e-6
    with col2:
        Param['fsw'] = st.number_input("Switching frequency [Hz]", min_value = 1e3, max_value = 1e6, value = 10e3, step = 1.)
        if topology_type == "DAB":
            Param['ph'] = st.number_input("Duty cycle [ ]", min_value = 0., max_value = 1., value = 0.5, step = 0.1)
        else:
            Param['duty'] = st.number_input("Duty cycle [ ]", min_value = 0., max_value = 1., value = 0.5, step = 0.1)
    # Assign the inputs to the simulation parameter structure
    circuit.setParam(Param)

    # Core parameters
    Param_mag = {}
    Param_mag['lc'] = 0
    Param_mag['Ac'] = 0
    Param_mag['lg'] = 0
    Param_mag['Np'] = 0
    Param_mag['Ns'] = 0

    mag = MagModel("")
    if topology_type  == "Flyback" or topology_type  == "DAB":
        mag = MagModel("Toroid_2W")
    else:
        mag = MagModel("Toroid")

    # Display geometry
    mag.displaySch(path)

    st.header("Core geometry")
    col1, col2 = st.beta_columns(2)
    with col1:
        Param_mag['lc'] = st.number_input("Length of core [mm]", min_value = 0., max_value = 1000., value = 100., step = 10.)*1e-3
        Param_mag['Ac'] = st.number_input("Cross section [mm2]", min_value = 0., max_value = 1000., value = 600., step = 10.)*1e-6
        Param_mag['lg'] = st.number_input("Length of gap [mm]", min_value = 0., max_value = 1000., value = 1., step = 1.)*1e-3
    with col2:
        Param_mag['Np'] = st.number_input("Turns number primary [ ]", min_value = 0., max_value = 1000., value = 8., step = 1.)
        if topology_type == "Flyback" or topology_type == "DAB":
            Param_mag['Ns'] = st.number_input("Turns number secondary [ ]", min_value = 0., max_value = 1000., value = 8., step = 1.)

    # Assign the inputs to the simulation parameter structure
    mag.setParam(Param_mag)

    # Steinmetz Parameters
    st.header("Material parameters")

    Param_material = {}
    Param_material["mu_r"] = 6500
    Param_material["iGSE_ki"] = 8.41
    Param_material["iGSE_alpha"] = 1.09
    Param_material["iGSE_beta"] = 2.16

    material = CoreMaterial("N87")
    material.setParam(Param_material)

    df = pd.DataFrame(
    np.array([[material.mu_r, material.iGSE_ki, material.iGSE_alpha, material.iGSE_beta]]),
    columns=["μr","ki","α", "β"])
    st.table(df)

    # Simulate and obtain the data
    result = st.button("Simulate")
    Ploss = 0
    circuit.setMagModel(mag, material)
    if result:
        Ploss = circuit.steadyRun(path)
        circuit.displayWfm()

    return Ploss
