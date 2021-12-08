import re
import numpy as np
import streamlit as st

from magnet import config
from magnet.constants import material_names, excitations
from magnet.plots import waveform_visualization, core_loss_multiple
from magnet.core import loss
from magnet.simplecs.simfunctions import SimulationPLECS


def header(material, excitation):
    s = f'Core Loss Analysis - {material} Material, {excitation} '
    return st.title(s)


def ui_core_loss_predict(m):
    st.sidebar.header(f'Information for Material {m}')
    material = st.sidebar.selectbox(f'Material {m}:', material_names)
    excitation = st.sidebar.selectbox(f'Excitation {m}:', excitations[1:] # Remove "Datasheet" from Prediction
                                      + ("Arbitrary", "Simulated"))  #TBD
    st.sidebar.markdown("""---""")

    if excitation == "Sinusoidal":
        header(material, excitation)
        col1, col2 = st.columns(2)
        with col1:
            st.header(f'Waveform Information for Material {m}')
            Freq = st.slider("Frequency (kHz)", # Use kHz for front-end demonstration while Hz for underlying calculation
                             config.streamlit.freq_min/1e3, 
                             config.streamlit.freq_max/1e3, 
                             config.streamlit.freq_max/2/1e3, 
                             step=config.streamlit.freq_step/1e3,key = f'Freq {m}')   *1e3
            Flux = st.slider("AC Flux Density Amplitude (mT)", # Use mT for front-end demonstration while T for underlying calculation
                             config.streamlit.flux_min*1e3,  
                             config.streamlit.flux_max*1e3, 
                             config.streamlit.flux_max/2*1e3, 
                             step=config.streamlit.flux_step*1e3,key = f'Flux {m}')   /1e3
            Bias = st.slider("DC Bias (mT)   (Coming Soon, Default as 0mT)", -300, 300, 0, step=int(1e7),key = f'Bias {m}') # Use step=Inf to disable this slider  #TBD
            duty_list = np.linspace(0, 1, 101)
            flux_read = np.multiply(np.sin(np.multiply(duty_list, np.pi * 2)), Flux)
            flux_list = np.multiply(np.add(flux_read, Bias),1e3)

        with col2:
            waveform_visualization(st, x=duty_list, y=flux_list)

        col1, col2 = st.columns(2)
        with col1:
            core_loss_multiple(
                st,
                x=[freq/1e3 for freq in config.streamlit.core_loss_freq],
                y1=[1e-3*loss(waveform='sine', algorithm='iGSE', material=material, freq=i, flux=Flux) for i in config.streamlit.core_loss_freq],
                y2=[1e-3*loss(waveform='sine', algorithm='ML', material=material, freq=i, flux=Flux) for i in config.streamlit.core_loss_freq],
                #y3=[loss(waveform='sine', algorithm='Ref', material=material, freq=i, flux=Flux) for i in config.streamlit.core_loss_freq],
                title=f'Core Loss with Frequency Sweeping and Fixed Flux Density {Flux*1e3} mT',
                x_title='Frequency [kHz]'
            )

        with col2:
            core_loss_multiple(
                st,
                x=[flux*1e3 for flux in config.streamlit.core_loss_flux],
                y1=[1e-3*loss(waveform='sine', algorithm='iGSE', material=material, freq=Freq, flux=i) for i in config.streamlit.core_loss_flux],
                y2=[1e-3*loss(waveform='sine', algorithm='ML', material=material, freq=Freq, flux=i) for i in config.streamlit.core_loss_flux],
                #y3=[loss(waveform='sine', algorithm='Ref', material=material, freq=Freq, flux=i) for i in config.streamlit.core_loss_flux],
                title=f'Core Loss with Flux Density Sweeping and Fixed Frequency {Freq/1e3} kHz',
                x_title='Flux Density [mT]'
            )
        
        st.header(f'{material}, {excitation}, f={Freq/1e3} kHz, \u0394B={Flux*1e3} mT, Bias={Bias} mT')
        core_loss_iGSE = loss(waveform='sine', algorithm='iGSE', material=material, freq=Freq, flux=Flux)/1e3
        core_loss_ML = loss(waveform='sine', algorithm='ML', material=material, freq=Freq, flux=Flux)/1e3
        st.header(f'Core Loss: {round(core_loss_iGSE,2)} kW/m^3 (by iGSE), {round(core_loss_ML,2)} kW/m^3 (by ML)')

    if excitation == "Triangular":
        header(material, excitation)
        col1, col2 = st.columns(2)
        with col1:
            st.header(f'Waveform Information for Material {m}')
            Freq = st.slider("Frequency (kHz)", 
                             config.streamlit.freq_min/1e3, 
                             config.streamlit.freq_max/1e3, 
                             config.streamlit.freq_max/2/1e3, 
                             step=config.streamlit.freq_step/1e3,key = f'Freq {m}')   *1e3
            Flux = st.slider("AC Flux Density Amplitude (mT)", 
                             config.streamlit.flux_min*1e3,  
                             config.streamlit.flux_max*1e3, 
                             config.streamlit.flux_max/2*1e3, 
                             step=config.streamlit.flux_step*1e3,key = f'Flux {m}')   /1e3
            Duty = st.slider("Duty Ratio", 
                             config.streamlit.duty_min,
                             config.streamlit.duty_max, 
                             (config.streamlit.duty_min+config.streamlit.duty_max)/2,
                             step=config.streamlit.duty_step,key = f'Duty {m}')
            Bias = st.slider("DC Bias (mT) (Coming Soon, Default as 0mT)", -300, 300, 0, step=int(1e7),key = f'Bias {m}') # Use step=Inf to disable this slider  #TBD
            duty_list = [0, Duty, 1]
            flux_read = [0, 2*Flux, 0]
            flux_mean = Flux
            flux_diff = Bias - flux_mean
            flux_list = np.multiply(np.add(flux_read, flux_diff),1e3)

        with col2:
            waveform_visualization(st, x=duty_list, y=flux_list)

        col1, col2, col3 = st.columns(3)
        with col1:
            core_loss_multiple(
                st,
                x=[freq/1e3 for freq in config.streamlit.core_loss_freq],
                y1=[1e-3*loss(waveform='triangle', algorithm='iGSE', material=material, freq=i, flux=Flux, duty_ratio=Duty) for i in config.streamlit.core_loss_freq],
                y2=[1e-3*loss(waveform='triangle', algorithm='ML', material=material, freq=i, flux=Flux, duty_ratio=Duty) for i in config.streamlit.core_loss_freq],
                title=f'Core Loss with Frequency Sweeping, Fixed Flux Density {Flux*1e3} mT at D={Duty}',
                x_title='Frequency [kHz]',
                x_log=True,
                y_log=True
            )

        with col2:
            core_loss_multiple(
                st,
                x=[flux*1e3 for flux in config.streamlit.core_loss_flux],
                y1=[1e-3*loss(waveform='triangle', algorithm='iGSE', material=material, freq=Freq, flux=i, duty_ratio=Duty) for i in config.streamlit.core_loss_flux],
                y2=[1e-3*loss(waveform='triangle', algorithm='ML', material=material, freq=Freq, flux=i, duty_ratio=Duty) for i in config.streamlit.core_loss_flux],
                title=f'Core Loss with Flux Density Sweeping, Fixed Frequency {Freq/1e3} kHz at D={Duty}',
                x_title='Flux Density [mT]',
                x_log=True,
                y_log=True
            )

        with col3:
            core_loss_multiple(
                st,
                x=config.streamlit.core_loss_duty,
                y1=[1e-3*loss(waveform='triangle', algorithm='iGSE', material=material, freq=Freq, flux=Flux, duty_ratio=i) for i in config.streamlit.core_loss_duty],
                y2=[1e-3*loss(waveform='triangle', algorithm='ML', material=material, freq=Freq, flux=Flux, duty_ratio=i) for i in config.streamlit.core_loss_duty],
                title=f'Core Loss with Duty Ratio Sweeping at {Freq/1e3} kHz and {Flux*1e3} mT',
                x_title='Duty Ratio',
                x_log=False,
                y_log=True
            )

        st.header(f'{material}, {excitation}, f={Freq/1e3} kHz \u0394B={Flux*1e3} mT, D={Duty}, Bias={Bias} mT')
        core_loss_iGSE = loss(waveform='triangle', algorithm='iGSE', material=material, freq=Freq, flux=Flux, duty_ratio=Duty)/1e3
        core_loss_ML = loss(waveform='triangle', algorithm='ML', material=material, freq=Freq, flux=Flux, duty_ratio=Duty)/1e3
        st.header(f'Core Loss: {round(core_loss_iGSE,2)} kW/m^3 (by iGSE), {round(core_loss_ML,2)} kW/m^3 (by ML)')

    if excitation == "Trapezoidal":
        header(material, excitation)
        col1, col2 = st.columns(2)
        with col1:
            st.header(f'Waveform information for Material {m}')
            Freq = st.slider("Frequency (kHz)", 
                             config.streamlit.freq_min/1e3, 
                             config.streamlit.freq_max/1e3, 
                             config.streamlit.freq_max/2/1e3, 
                             step=config.streamlit.freq_step/1e3,key = f'Freq {m}')   *1e3
            Flux = st.slider("AC Flux Density Amplitude (mT)",
                             config.streamlit.flux_min*1e3,  
                             config.streamlit.flux_max*1e3, 
                             config.streamlit.flux_max/2*1e3, 
                             step=config.streamlit.flux_step*1e3,key = f'Flux {m}')   /1e3
            DutyP = st.slider("Duty Ratio (Rising)", 
                             config.streamlit.flux_step,
                             1-config.streamlit.flux_step*3,
                             (config.streamlit.duty_min+config.streamlit.duty_max)/2,
                             step=config.streamlit.flux_step,key = f'DutyP {m}')
            DutyN = st.slider("Duty Ratio (Falling)", 
                             config.streamlit.flux_step, 
                             1-DutyP-config.streamlit.flux_step*2, 
                             round((1-DutyP)/3,2), 
                             step=config.streamlit.flux_step,key = f'DutyN {m}')
            Duty0 = st.slider("Duty Ratio (Flat) (Asymmetric Flat Duty Ratio Coming Soon)",  #TBD
                             config.streamlit.flux_step, 
                             1-config.streamlit.flux_step*3, 
                             (1-DutyP-DutyN)/2, 
                             step=1e7,key = f'Duty0 {m}')
            Bias = st.slider("DC Bias (mT) (Coming Soon, Default as 0mT)", -300, 300, 0, step=int(1e7),key = f'Bias {m}') # Use step=Inf to disable this slider  #TBD
            duty_list = [0, DutyP, DutyP+Duty0, 1-Duty0, 1]
            if DutyP>DutyN :
                BPplot=Flux # Since Bpk is proportional to the voltage, and the voltage is proportional to (1-dp+dN) times the dp
                BNplot=-BPplot*((-1-DutyP+DutyN)*DutyN)/((1-DutyP+DutyN)*DutyP) # proportional to (-1-dp+dN)*dn
            else :
                BNplot=Flux # proportional to (-1-dP+dN)*dN
                BPplot=-BNplot*((1-DutyP+DutyN)*DutyP)/((-1-DutyP+DutyN)*DutyN) # proportional to (1-dP+dN)*dP
            flux_read = [-BPplot,BPplot,BNplot,-BNplot,-BPplot]
            flux_list = np.multiply(np.add(flux_read, Bias),1e3)
            duty_ratios = [DutyP,DutyN,Duty0]

        with col2:
            waveform_visualization(st, x=duty_list, y=flux_list)

        col1, col2 = st.columns(2)
        with col1:
            core_loss_multiple(
                st,
                x=[freq/1e3 for freq in config.streamlit.core_loss_freq],
                y1=[1e-3*loss(waveform='trapezoid', algorithm='iGSE', material=material, freq=i, flux=Flux, duty_ratios=duty_ratios) for i in config.streamlit.core_loss_freq],
                y2=[1e-3*loss(waveform='trapezoid', algorithm='ML', material=material, freq=i, flux=Flux, duty_ratios=duty_ratios) for i in config.streamlit.core_loss_freq],
                title=f'Core Loss with Frequency Sweeping, Fixed Flux Density {Flux*1e3} mT and Given Duty Ratios',
                x_title='Frequency [kHz]'
            )

        with col2:
            core_loss_multiple(
                st,
                x=[flux*1e3 for flux in config.streamlit.core_loss_flux],
                y1=[1e-3*loss(waveform='trapezoid', algorithm='iGSE', material=material, freq=Freq, flux=i, duty_ratios=duty_ratios) for i in config.streamlit.core_loss_flux],
                y2=[1e-3*loss(waveform='trapezoid', algorithm='ML', material=material, freq=Freq, flux=i, duty_ratios=duty_ratios) for i in config.streamlit.core_loss_flux],
                title=f'Core Loss with Flux Density Sweeping, Fixed Frequency {Freq/1e3} kHz and Given Duty Ratios',
                x_title='Flux Density [mT]'
            )
            
        st.header(f'{material}, {excitation}, f={Freq/1e3} kHz, \u0394B={Flux*1e3} mT, DP={round(DutyP,2)}, DN={round(DutyN,2)}, D0={round(Duty0,2)}, Bias={Bias} mT')
        duty_ratios = [DutyP,DutyN,Duty0]
        core_loss_iGSE = loss(waveform='trapezoid', algorithm='iGSE', material=material, freq=Freq, flux=Flux, duty_ratios=duty_ratios)/1e3
        core_loss_ML = loss(waveform='trapezoid', algorithm='ML', material=material, freq=Freq, flux=Flux, duty_ratios=duty_ratios)/1e3
        st.header(f'Core Loss: {round(core_loss_iGSE,2)} kW/m^3 (by iGSE), {round(core_loss_ML,2)} kW/m^3 (by ML)')

    if excitation == "Arbitrary":
        header(material, excitation)
        st.header("Coming Soon! This Section is still under development.") #TBD
        col1, col2 = st.columns(2)
        with col1:
            Freq = st.slider("Cycle Frequency (kHz)",
                             config.streamlit.freq_min/1e3, 
                             config.streamlit.freq_max/1e3, 
                             config.streamlit.freq_max/2/1e3, 
                             step=config.streamlit.freq_step/1e3,key = f'Freq {m}')   *1e3
            duty_string = st.text_input("Waveform Pattern Duty in a Cycle (%)",
                                        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                        key = f'Duty {m}')
            flux_string = st.text_input("Waveform Pattern Relative Flux Density (mT)",
                                        [0, 10, 20, 10, 20, 30, -10, -30, 10, -10, 0],
                                        key = f'Flux {m}')
            Bias = st.slider("DC Bias (mT)", -300, 300, 0, step=10,key = f'Bias {m}')

            duty_list = [float(i) for i in re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", duty_string)]
            flux_read = [float(i) for i in re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", flux_string)]
            flux_mean = np.average(flux_read)
            flux_diff = Bias - flux_mean
            flux_list = np.add(flux_read, flux_diff)

        with col2:
            waveform_visualization(st, x=duty_list, y=flux_list)

        core_loss_iGSE = loss(waveform='arbitrary', algorithm='iGSE', material=material, freq=Freq, flux_list=flux_list, frac_time=duty_list)/1e3
        core_loss_ML = loss(waveform='arbitrary', algorithm='ML', material=material, freq=Freq, flux_list=flux_list, frac_time=duty_list)/1e3
        st.header("Coming Soon! This Section is still under development.") #TBD
        st.header(f'Core Loss: {round(core_loss_iGSE,2)} kW/m^3 (by iGSE), {round(core_loss_ML,2)} kW/m^3 (by ML)')

    if excitation == "Simulated":
        header(material, excitation)
        st.header("Coming Soon! This Section is still under development.") #TBD
        core_loss_iGSE = SimulationPLECS(material, algorithm='iGSE')/1e3
        core_loss_ML = SimulationPLECS(material, algorithm='ML')/1e3
        st.header(f'Core Loss: {round(core_loss_iGSE,2)} kW/m^3 (by iGSE), {round(core_loss_ML,2)} kW/m^3 (by ML)')
        
    st.markdown("""---""")