import streamlit as st
import numpy as np
from magnet import config as c
import os

from magnet.simplecs.classes import CircuitModel, MagModel, CoreMaterial
from magnet.core import core_loss_arbitrary,BH_Transformer,loss_BH
from magnet.constants import material_list, material_extra, material_steinmetz_param


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
        Param['Vi'] = st.number_input("Voltage input [V]", min_value=0.01, max_value=1000., value=24., step=2.,
                                      key='Vi')
        Param['Ro'] = st.number_input("Load resistor [Ω]", min_value=0.01, max_value=1e6, value=10., step=5.,
                                     key='R')
        if topology_type == "DAB":
            Param['Lk'] = st.number_input("Series inductor [μH]", min_value=0.001, max_value=1000., value=10., step=1., key='Lk')*1e-6
    with col4:
        Param['fsw'] = st.number_input("Switching frequency [kHz]", min_value=50., max_value=500., value=200., step=10.,
                                       key='fsw')*1e3
        if topology_type == "DAB":
            Param['ph'] = st.number_input("Phase shift [deg]", min_value=0., max_value=360., value=90., step=1.,
                                          key='ph')/360.
            Param['duty'] = 0.5
        else:
            Param['duty'] = st.number_input("Duty cycle [p.u.]", min_value=0.01, max_value=0.99, value=0.5, step=0.01,
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
        Param_mag['lc'] = st.number_input("Length of core [mm]", min_value=0.01, max_value=1000., value=103., step=1.,
                                          key='Lc')*1e-3
        Param_mag['Ac'] = st.number_input("Cross section [mm2]", min_value=0.01, max_value=1000., value=96., step=1.,
                                          key='Ac')*1e-6
    with col4:    
        Param_mag['lg'] = st.number_input("Length of gap [mm]", min_value=0.0001, max_value=1000., value=0.5, step=0.01,
                                          key='lg')*1e-3
    
        Param_mag['Np'] = st.number_input("Turns number primary", min_value=0., max_value=100., value=8., step=1.,
                                          key='Np')
        if topology_type == "Flyback" or topology_type == "DAB":
            Param_mag['Ns'] = st.number_input("Turns number secondary", min_value=0., max_value=100., value=8.,
                                              step=1., key='Ns')

    # Assign the inputs to the simulation parameter structure
    mag.setParam(Param_mag)
    
    Vc = (Param_mag['lc']+Param_mag['lg'])*Param_mag['Ac']

    # Steinmetz Parameters
    st.header("Magnetic Material")

    col1, col2 =  st.columns(2)
    with col1:
        Material_type = st.selectbox(
            "Material:",
            material_list,
            index=9,
            key='Material'
        )
        
        k_i, alpha, beta = material_steinmetz_param[Material_type]
        mu_r_0 = material_extra[Material_type][0]
        
        Param_material = {
            'mu_r': mu_r_0,
            'iGSE_ki': k_i,
            'iGSE_alpha': alpha,
            'iGSE_beta': beta
        }
        material = CoreMaterial(Material_type)
        material.setParam(Param_material)
        st.write(f"Relative initial permeability: {material.mu_r}, for reference")
        
    with col2:
        Temperature = st.number_input(
            f'Temperature (C)',
            0,
            120,
            25,
            step=5,
            key=f'temp {m}')
    
    # Select a backend
    col1, col2 =  st.columns(2)

    with col1:
        # Select Backend
        backend_list = ("Plecs", "Python")
        backend_type = st.selectbox(
            "Backend:",
            backend_list,
            key='Backend'
        )


    # Simulate and obtain the data

    result = st.button("Simulate", key='Simulate')

    circuit.setMagModel(mag, material)

    if result:
        
        
        st.header("Simulation Results")
       
        if backend_type == "Plecs":
            flux, field, time = circuit.steadyRun(path)
        elif backend_type == "Python":
            flux, field, time = circuit.steadyRun_py(path)
        
        flux = np.array(flux)
        time = np.array(time)
        time_vector = np.multiply(time, Param['fsw'])
        
        temp = (time_vector <= 1)
        flux = flux[temp]

        field = np.array(field)
        field = field[temp]
        
        bias = (np.max(field) + np.min(field)) / 2
        time_vector = time_vector[temp]
           
        flux_amp = (np.max(flux) - np.min(flux)) / 2
        
        duty = time_vector
        bdata_pre = np.interp(np.linspace(0, 1, c.streamlit.n_nn+1), np.array(duty), np.array(flux))
        bdata_pre = bdata_pre[:-1]

        bdata = bdata_pre - np.average(bdata_pre)

        hdata = BH_Transformer(material=Material_type, 
                               freq=Param['fsw'],
                               temp=Temperature, 
                               bias=bias, 
                               bdata=bdata)
        loss = loss_BH(bdata, hdata, freq=Param['fsw'])
        circuit.Binterp = bdata
        circuit.Hinterp = hdata
        circuit.bias = bias

        st.header("Core Loss Based on Simulated Waveform at 25 C")
        st.subheader(f'{round(loss*Vc,2)} W ({round(loss / 1e3,2)} kW/m^3)')
        
        if flux_amp < 0.01:
            st.warning("""
                     The simulated amplitude of flux density is **too small** under the given 
                     parameter configurations. The predicted core loss result may be inaccurate!
                     """)
        elif flux_amp > 0.3:
            st.warning("""
                     The simulated amplitude of flux density is **too large** under the given 
                     parameter configurations. The predicted core loss result may be inaccurate!
                     """)
        elif topology_type == "DAB":
            st.write(f"""
                     **Note**: This core loss result stands for the loss in the magnetic core of the **transformer**, 
                     while the series auxiliary inductor is assumed lossless.""")
                     
        
                
        col1, col2 = st.columns(2)
        with col1:
                circuit.displayWfm()
        with col2:
                circuit.displayBH()
