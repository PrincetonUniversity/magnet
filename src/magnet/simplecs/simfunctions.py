import streamlit as st
import numpy as np
import pandas as pd
import os

from magnet.simplecs.classes import CircuitModel, MagModel, CoreMaterial
from magnet.core import loss
from magnet.constants import materials, materials_extra, material_names


def SimulationPLECS(m):
    path = os.path.dirname(os.path.realpath(__file__))

    col1, col2 = st.columns(2)
    with col1:
        # Select topology
        topology_list = ("Buck", "Boost", "Flyback", "DAB")
        topology_type = st.selectbox(
            "Topology:",
            topology_list,
            key='Topology'
        )
        
    # Circuit model instance
    circuit = CircuitModel(topology_type)


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

    st.header("Circuit Parameters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Display schematic
        circuit.displaySch(path)
    with col3:      
        Param['Vi'] = st.number_input("Voltage input [V]", min_value=0., max_value=1000., value=40., step=10.,
                                      key='Vi')
        Param['Ro'] = st.number_input("Load resistor [Ω]", min_value=0., max_value=1e6, value=10., step=10.,
                                     key='R')
        if topology_type == "DAB":
            Param['Lk'] = st.number_input("Series inductor [μH]", min_value=0., max_value=1000., value=12., step=1., key='Lk')*1e-6
    with col4:
        Param['fsw'] = st.number_input("Switching frequency [kHz]", min_value=50., max_value=500., value=100., step=1.,
                                       key='fsw')*1e3
        if topology_type == "DAB":
            Param['ph'] = st.number_input("Phase shift [deg]", min_value=0., max_value=360., value=90., step=1.,
                                          key='ph')/360.
            Param['duty'] = 0.5
        else:
            Param['duty'] = st.number_input("Duty cycle [p.u.]", min_value=0., max_value=1., value=0.5, step=0.01,
                                            key='duty')
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


    st.header("Core Geometry")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Display geometry
        mag.displaySch(path)   
    with col3: 
        Param_mag['lc'] = st.number_input("Length of core [mm]", min_value=0., max_value=1000., value=100., step=1.,
                                          key='Lc')*1e-3
        Param_mag['Ac'] = st.number_input("Cross section [mm2]", min_value=0., max_value=1000., value=100., step=1.,
                                          key='Ac')*1e-6
    with col4:    
        Param_mag['lg'] = st.number_input("Length of gap [mm]", min_value=0., max_value=1000., value=0.1, step=0.01,
                                          key='lg')*1e-3
    
        Param_mag['Np'] = st.number_input("Turns number primary", min_value=0., max_value=100., value=10., step=1.,
                                          key='Np')
        if topology_type == "Flyback" or topology_type == "DAB":
            Param_mag['Ns'] = st.number_input("Turns number secondary", min_value=0., max_value=100., value=10.,
                                              step=1., key='Ns')

    # Assign the inputs to the simulation parameter structure
    mag.setParam(Param_mag)
    
    Vc = (Param_mag['lc']+Param_mag['lg'])*Param_mag['Ac']

    # Steinmetz Parameters
    st.header("Magnetic Material")

    col1, col2 = st.columns(2)
    with col1:
        Material_list = material_names
        Material_type = st.selectbox(
            "Material:",
            Material_list,
            index = 9,
            key='Material'
        )
        
        k_i, alpha, beta = materials[Material_type]
        mu_r_0 = materials_extra[Material_type][0]
        
        Param_material = {
            'mu_r': mu_r_0,
            'iGSE_ki': k_i,
            'iGSE_alpha': alpha,
            'iGSE_beta': beta
        }
        material = CoreMaterial(Material_type)
        material.setParam(Param_material)
    with col2:
        st.write("Reference parameters of the selected magnetic material:")
        df = pd.DataFrame(
            np.array([[material.mu_r, material.iGSE_ki, material.iGSE_alpha, material.iGSE_beta]]),
            columns=["μr", "ki", "α", "β"]
        )
        
        df = df.style.format({"μr": "{:.0f}", "ki": "{:.4f}", "α": "{:.4f}", "β": "{:.4f}"})
        st.table(df)

    # Simulate and obtain the data
    result = st.button("Simulate", key='Simulate')

    circuit.setMagModel(mag, material)
    
    
    if result:
        
        col1, col2 = st.columns(2)
        with col1:
        
            st.header("Simulation Results")
            
            Flux,Time = circuit.steadyRun(path)
            
            Flux = np.array(Flux)
            Time = np.array(Time)
            Duty = np.multiply(Time,Param['fsw'])
            
            temp = (Duty<=1)
            Flux = Flux[temp]
            Duty = Duty[temp]
            
            Flux_amp = (np.max(Flux) - np.min(Flux))/2
    
            Loss_iGSE = loss(
                waveform="Arbitrary", 
                algorithm="iGSE", 
                material=Material_type, 
                freq=Param['fsw'], 
                flux=Flux, 
                duty=Duty) / 1e3
            
            Loss_ML = loss(
                waveform="Arbitrary", 
                algorithm="ML", 
                material=Material_type, 
                freq=Param['fsw'], 
                flux=Flux, 
                duty=Duty) / 1e3
            
        
            st.header("Simulated Core Loss")
            st.subheader(f'{round(Loss_iGSE*Vc*1e3,2)} W  ({round(Loss_iGSE,2)} kW/m^3) - iGSE')
            st.subheader(f'{round(Loss_ML*Vc*1e3,2)} W ({round(Loss_ML,2)} kW/m^3) - ML')
            
            if Flux_amp<0.02:
                st.write("""
                         **Caution**: The simulated amplitude of flux density is **too small** under the given 
                         parameter configurations. The predicted core loss result may be inaccurate!
                         """)
            elif Flux_amp>0.3:
                st.write("""
                         **Caution**: The simulated amplitude of flux density is **too large** under the given 
                         parameter configurations. The predicted core loss result may be inaccurate!
                         """)
            
            if topology_type in ["Buck", "Boost", "Flyback"]:
                st.write(f"""
                         **Note**: The selected {topology_type} topology is very likely to result in a **dc-biased** 
                         flux density, which is not yet taken into consideration by the model.""")
                st.write("""
                         Models for **dc-biased** condition will be coming soon in the next release!""")
            elif topology_type == "DAB":
                st.write(f"""
                         **Note**: This core loss result stands for the loss in the magnetic core of the **transformer**, 
                         while the series auxiliary inductor is assumed lossless.""")
                         
            with col2:
                circuit.displayWfm()
