import pyvisa  # Control over USB (for the scope), GPIB, Ethernet, etc
import numpy as np
import csv
import time  # To wait specific time
import serial  # To control the power supplies and DSP
import scipy.integrate as integrate  # Just to get k from ki for Steinmetz Eqs
import math

rm = pyvisa.ResourceManager()
print(rm.list_resources())

# ============================================================================#
# Test to be performed

sine_or_trap = 1  # 0 to do sinusoidal, 1 to do triangular/trapezoidal

# Test under specific conditions if needed
DP_forced = 0  # disabled if set to 0, otherwise set to desired value
DN_forced = 0  # disabled if set to 0, otherwise set to desired value
Hdc_forced = -1  # disabled if set to -1, , otherwise, select desired value in A/m

if sine_or_trap == 0:
    print('Sinusoidal tests to be performed:\n')
elif sine_or_trap == 1:
    print('Triangular/Trapezoidal tests to be performed (use DSP code "main_trap_asym_controlled_v3"):\n')
else:
    input('Neither sinusoidal nor trapezoidal selected, error')
# ============================================================================#
# Details about the core to be tested (m^2 and H)
# Steinmetz parameters from 10 mT to 300 mT 50 kHz to 500 kHz 
# Maximum amplitude permeability at the tested temperature (mu_ra_max)

# ######### Core ########## 3C90
# N = 7; Ae = 48.9e-6; le = 60.18e-3  # TX25/15/10
# ki = 0.0422; alpha = 1.5424; beta = 2.6152
# mu_ra_max = 5250
# ######### Core ########## 3C94
# N = 7; Ae = 33.6e-6; le = 43.6e-3 # TX20/10/7
# ki = 0.0123; alpha = 1.6159; beta = 2.4982
# mu_ra_max = 4700
# ######### Core ########## 3E6
# N = 4; Ae = 24.8e-6; le=54.2e-3;  # TX-22-14-6.4
# ki = 0.0002; alpha = 1.9098; beta = 2.0903
# mu_ra_max = 18000
# ######### Core ########## 3F4
# N = 5; Ae = 130e-6; le = 35.1e-3 # E-32-6-20-R
# ki = 0.7580; alpha = 1.4146; beta =3.1455
# mu_ra_max = 1800 
# ######### Core ########## 77
# N = 7; Ae = 40e-6; le = 62e-3;
# ki = 0.0537; alpha = 1.5269; beta = 2.519
# mu_ra_max = 5800 
# ######### Core ########## 78
# N = 6; Ae = 0.52e-4; le = 54e-3;
# ki = 0.0169; alpha = 1.6090; beta = 2.5432
# mu_ra_max = 7800 
# ######### Core ########## N27
# N = 8; Ae = 33.63e-6; le = 43.55e-3; # R20.0X10.0X7.0
# ki = 0.0669; alpha = 1.5158; beta = 2.5254
# mu_ra_max = 3750 
# ######### Core ########## N30
# N = 6; Ae = 26.17e-6; le = 54.15e-3; #R22.1X13.7X6.35
# ki = 0.0001; alpha = 1.9629; beta = 2.3541
# mu_ra_max = 5450
# ######### Core ########## N49
# N = 10; Ae = 19.73e-6; le = 38.52e-3;  # R16.0x9.6x6.3
# ki = 0.1326; alpha = 1.4987; beta = 3.2337
# mu_ra_max = 2700
# ######### Core ########## N87
N = 5; Ae = 82.6e-6; le = 82.06e-3  # R34.0x20.5x12.5
ki = 0.1518; alpha = 1.4722; beta = 2.6147
mu_ra_max = 3800

N1 = 0; N2 = 0  # Set to N unless otherwise stated
N1 = N if N1 == 0 else N1
N2 = N if N2 == 0 else N2

# ============================================================================#
# Data required for the tests

# Limits
Vdc_max = 80.0  # Maximum input voltage of the voltage supply (limited by the 100V series capacitor)
Vdc_min = 5.0  # Minimum imposed by this specific voltage supply (cannot be set to 1 V as the power supply resets the limits in RMT mode)
Vac_max = 50.0  # The output of the power amplifier looks distorted if the amplitude is above 50 V
Vac_min = 1.0  # To ensure a good resolution of the waveform
Vsignal_min = 0.002  # The signal amplifier cannot give a pk-pk voltage below 2 mV
Vsignal_max = 1  # Something is off if the voltage is so high in the cases tested so far
Idc_max = 1  # Maximum DC current
I_max = 2  # Maximum total current
PV_min = 1000  # Minimum losses based on the iGSE equations, only an estimate
PV_max = 5000000  # Maximum losses, points are skipped if outside this range

T_wait = 1  # seconds waiting between each frequency run, to let the core cool down

Lm_min = 5e-6  # Warning if L is below this value
Lm_max = 1e-3  # Warning if L is above this value
gain_min = 10  # Warning if gain is below this value
gain_max = 1000  # Warning if gain is above this value

# Values that the vertical scale can take, maximum values are 5 times the scale, as there are 5 divisions in the scope 
scale_list_V = np.array([0.5, 1, 2, 5, 10, 20, 50])  # List of vertical scales for the voltage
scale_list_I = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])  # List for the current
# The last number of the scale sets a protection for the voltage and current (max at 250 V and 2.5 A)
vertical_scale_margin = 40/100  # Extra room for the scale in the oscilloscope

# ============================================================================#
# Maximum and minimum values for the test (update this info also in Code Composer)
# 1 means 100%
Dstep = 0.1  # Keep at 0.1, otherwise seriously review the other codes for compatibility
D0min = 0.0  # 0.0 to include triangular waveforms
D0max = 0.4  # Below 0.5

# in Hz
Fmin = 50e3
Fmax = 510e3
Fpointsperdecade = 10  # Number of points per decade

# in T, this is the PEAK flux, not the peak to peak
Bmin = 10e-3
Bmax = 300e-3
Bpointsperdecade = 10  # Number of points per decade

# in A/m
Hmin = 0
Hstep = 15
Hmax = (Bmax-Bmin)/(1.256637e-6*mu_ra_max)

# ============================================================================#
# Some constants
pi = 3.1415926
R_sense = 0.983
k_steinmetz = ki*(2*pi)**(alpha-1)*2**(beta-alpha)*max(integrate.quad((lambda x: (abs(math.cos(x))**alpha)), 0, 2*pi))

# Flux range from 1 mT to 1 T to ensure that specific points measured are not a function of Bmin
logB_range = np.arange(np.log10(0.001), np.log10(1), 1/Bpointsperdecade)
logB_range = logB_range[logB_range >= np.log10(Bmin)]  # Keeping the B points of interest
# Freq range from 10 kHz to 10 MHz to ensure that specific points measured are not a function of Fmin
logF_range = np.arange(np.log10(10000), np.log10(10000000), 1/Fpointsperdecade)
logF_range = logF_range[logF_range >= np.log10(Fmin)]  # Keeping the F points of interest
logF_range = logF_range[logF_range <= np.log10(Fmax)]

# ============================================================================#
# Modify the limits for each type of test

if sine_or_trap == 0:  # For Sinusoidal, disable the steps in duty cycle
    D0max = 0.0
    D0min = 0.0
    Dstep = 0.5  # Just to ensure Dpmin<Dpmax in the Dp loop, then we enter both D0 and Dp loops only once
if Bmax > 0.350:
    input('Warning: Bmax above 350 mT, ENTER to continue')
 
if sine_or_trap == 1:  # In case of error setting the duty cycle
    if D0max > 0.5-Dstep:  # D0 cannot be above 0.5, just in case there is a human error
        print('D0 max not valid, set to D0 min')
        D0max = D0min
    if Dstep < 0.01:  # Dstep cannot be below 0.01, just in case there is a human error
        print('Dstep max not valid, set to 0.01')
        Dstep = 0.01
        
# Variables for expected time left
N_Hdc_loops = int((Hmax-Hmin)/Hstep+1)
if sine_or_trap == 0:  # For sinusoidal, there is a single round
    N_runs = 1
else:  # For Triangular and Trapezoidal, just as in the iterations
    N_runs = 0
    for D0 in np.arange(D0min, D0max+Dstep*0.5, Dstep):
        for Dp in np.arange(Dstep, 1-2*D0-Dstep+Dstep*0.5, Dstep):
            N_runs = N_runs+1
N_f_loops = logF_range.size  # Number of frequency loops
        
# ============================================================================#
# Setup the scope
   
scope = rm.open_resource('USB::0x0699::0x0401::C012812::INSTR')  # DPO4054

scope.timeout = 10000  # in ms
scope.encoding = 'latin_1'
scope.write_termination = None
scope.write('*cls')  # clear ESR
scope.write('header OFF')  # disable attribute echo in replies
# print(scope.query('*idn?'))

# Initialize the scope measurement
scope.write('*rst')  # default setup
# autoset
scope.write('autoset EXECUTE')  # 50 kHz 1 Vpp Sine Wave as reference
scope.write('acquire:state RUN')  # stop
scope.write('acquire:stopafter RUNSTOP;state RUN')  # single

# Turn-on the channels in use
scope.write('SELECT:CH2 ON')
scope.write('SELECT:CH3 ON')
scope.write('SELECT:CH1 OFF')
scope.write('SELECT:CH4 OFF')

scope.write('TRIGger:A:EDGE:SOUrce CH2')
scope.write('TRIGger:A:Level:CH2 0')

scope.write('CH2:DESKew 0')
scope.write('CH3:DESKew 0')

scope.write('CH2:COUPling DC')
scope.write('CH3:COUPling DC')

scope.write('CH2:BANdwidth 20e6')
scope.write('CH3:BANdwidth 20e6')

scope.write('CH2:Termination 1e6')
scope.write('CH3:Termination 50')

scope.write('CH2:position 0')
scope.write('CH3:position 0')

scope.write('CH2:scale 10')  
scope.write('CH3:scale 0.1') 

# Regarding the sampling time, Ts, the maximum is 400ps as this scope has 2.5Gs/s
scope.write('HORIZONTAL:RECORDLENGTH 100000')  # Can be increased to 100k or decreased to 1k
scope.write('HORIZONTAL:scale 8e-6')  # The scope displays 10 "horizontal scales" --> Ts=10*10e-6/10000=10e-9
scope.write('Acquire:mode Sample')
r = scope.query('*opc?')

# ============================================================================#
# Setup the DC Voltage Source

if sine_or_trap == 1:  # For trapezoidal
    vdc = serial.Serial()
    vdc.baudrate = 57600
    vdc.port = 'COM3'
    vdc.timeout = 2
    # print(vdc)
    vdc.open()
    # vdc.read(1)  # It will return something if it is ok, but if not, it will give an error and stop the program
    print(f"Power supply communication ok? {vdc.is_open}")
    
    # Check the SeriesNumber of the Equipments
    vdc.write("CADR 1\n".encode())
    vdc.write("*IDN?\n".encode())
    vdc.write("CADR 2\n".encode())
    vdc.write("*IDN?\n".encode())
    
    # Recall the configuration of the power supply not keeping configuration setting (1V min 1V/ms slew rate)
    # Reset the Voltage Supply
    vdc.write("GRST\n".encode())
    
    # Set the OverVoltage Protection
    vdc.write("GOV 100\n".encode())  # Overvoltage at 100 V
    vdc.write("GOVP ON\n".encode())
    
    # Set the OverCurrent Protection
    vdc.write("GOC 1.0\n".encode())  # Overcurrent at 1 A
    vdc.write("GOCP ON\n".encode())
    
    # Preset the Output Voltage
    vdc.write(f"GPV {format(Vdc_min, '.2f')}\n".encode())

    # Disable the Output
    vdc.write("GOUT OFF\n".encode())

# ============================================================================#
# Setup the UART communication with DSP
    
if sine_or_trap == 1:  # For trapezoidal
    dsp = serial.Serial()
    dsp.baudrate = 4800
    dsp.port = 'COM7'
    dsp.timeout = 2
    # print(dsp)
    dsp.open()
    print(f"DSP communication ok? {dsp.is_open}")
    
# ============================================================================#
# Setup the signal generator
    
if sine_or_trap == 0:  # Sinusoidal
    # dg = rm.open_resource('USB::0x1AB1::0x0641::DG4E194002798::INSTR')
    # https://www.batronix.com/pdf/Rigol/ProgrammingGuide/DG1000Z_ProgrammingGuide_EN.pdf

    dg = rm.open_resource('USB::0x1AB1::0x0641::DG4E241100546::INSTR')
    dg.timeout = 10000  # in ms
    dg.encoding = 'latin_1'
    dg.write_termination = None
    dg.write('*cls')  # clear ESR
    dg.write('header OFF')  # disable attribute echo in replies
    # print(dg.query('*idn?'))

    dg.write(':SOUR1:APPL:SIN 50000,0.01,0,0')  # define the property of signal
    # [Type Freq(Hz),Amp(Vpp),Offset(Vdc),Phase(deg)]
    r = dg.query('*opc?')

    dg.write('OUTP1 ON')

# ============================================================================#
# Setup the DC bias supply
    
bias = rm.open_resource('USB::0xF4EC::0x1430::SPD3XIED5R8101::INSTR')
# https://www.siglenteu.com//wp-content/uploads/dlm_uploads/2017/10/SPD3303X_QuickStart_QS0503X-E01B.pdf

bias.write('OUTPut CH1,OFF')
bias.write('CH1:VOLTage 5')
bias.write('CH1:CURRent 0')  # Sets current to 0 A for safety
bias.write('OUTPut CH1,ON')

# ============================================================================#
# Main iteration

if sine_or_trap == 0:
    input("Ready to measure sinusoidal data, connect the probes, turn on the DC bias circuit and turn on the power amplifier, press ENTER to continue")
else:
    input("Ready to measure trapezoidal data, connect the probes, turn on the DC bias circuit, turn on the power supplies, the DSP code should be debug and run before python (if not, redo, check the D and F params) press ENTER to continue")

time.sleep(1)

flag_clipping_V = 0
flag_clipping_I = 0
it_test = 1  # Number of test

# ============================================================================#
# DC bias loop
it_H = 1 # Number for the H bias test
for Hdc in np.arange(Hmin, Hmax+Hstep*0.5, Hstep): 

    if Hdc_forced != -1.0:
        if int(Hdc+0.5) != int(Hdc_forced+0.5):
            print(f"\nHdc={format(Hdc, '.2f')} A/m skipped")
            continue
    
    t_H_loop = time.perf_counter()  # Store the current time each time a new d0 starts
    t_H_so_far = 0  # Updated later to store the total time in seconds for this Hdc loop
    
    Idc = Hdc*le/N1
    
    # Set the DC bias current
    bias.write(f"CH1:CURRent {Idc}")  # Sets current in the DC supply for the DC bias
    
    print(f"Hdc={Hdc} A/m; DC current set to {format(Idc*1000, '.0f')} mA")
    time.sleep(0.5)

    if Idc > Idc_max:  # Skip if the DC current is too high
        print(f"Measurement is aborted (Idc above {Idc_max} A).")
        continue

    # Calculation of the flux loop range
    # Estimated Bdc based on permeability, just to set the maximum B for the tests
    Bdc = 1.256637e-6*mu_ra_max*Hdc  # for that reason the max mu is used, so B is farther away form saturation
    Bmax_applied = max(Bmax-Bdc, Bmin)
    logB_range = logB_range[logB_range < np.log10(Bmax_applied)]
    if Bdc > 0:
        print(f"The DC flux density is expected to be around {format(Bdc*1000, '.0f')} mT so the max AC flux density is set to {format(Bmax_applied*1000, '.0f')} mT")

    # ============================================================================#
    # "Flat" duty cycle iteration (d2=d4)
    it_d = 1  # Number of file, of duty cycle iteration, or simply, run
    for D0 in np.arange(D0min, D0max+Dstep*0.5, Dstep):  # +Dstep*0.5 part only to always include Dmax
        
        time.sleep(0.5)
        
        if sine_or_trap == 1 and (DP_forced != 0.0 or DN_forced != 0.0):
            DO_forced = (1-DP_forced-DN_forced)/2
            if int(D0*100.0+0.5) != int(DO_forced*100.0+0.5):
                print(f"\nD0={format(D0, '.2f')} skipped")
                continue
        
        if sine_or_trap == 1:  # For trapezoidal
            dsp.write(b'1')
            time.sleep(0.5)
            print(f"\nD0={format(D0*100, '.0f')}% starts:", end=" ")
            temp = dsp.read(dsp.inWaiting())
            d0_dsp = int(temp.hex(), 16)
            d0_command = int(D0*100.0+0.5)
            print(f"D0={d0_dsp} % is loaded!")
            time.sleep(0.5)
            if d0_dsp != d0_command:
                print(f"Error in the DSP duty's command (d0: dsp={d0_dsp}, command={d0_command})")
                continue
        
        # ============================================================================#
        # "Rising" duty cycle iteration
        for Dp in np.arange(Dstep, 1-2*D0-Dstep+Dstep*0.5, Dstep):  # +Dstep*0.5 only to always include Dmax
            # Regarding the for loops, the loop is skipped if min=max, also, remember that max is not included
            
            t_run_loop = time.perf_counter()  # Store the current time each time a new dP starts
            t_run_so_far = 0  # Updated later to store the total time in seconds for this run
            time.sleep(0.5)
            
            if sine_or_trap == 1 and (DP_forced != 0.0 or DN_forced != 0.0):
                if int(Dp*100.0+0.5) != int(DP_forced*100.0+0.5):
                    print(f"\nDp={format(Dp, '.2f')} skipped")
                    continue

            Dn = 1-2*D0-Dp  # Fraction of the cycle where the negative power supply is connected.
        
            if sine_or_trap == 1:  # For trapezoidal
                dsp.write(b'1')
                time.sleep(0.5)
                print(f"\nD0={format(D0, '.2f')}; Dp={format(Dp, '.2f')}; Dn={format(Dn, '.2f')} starts:")
                temp = dsp.read(dsp.inWaiting())
                dp_dsp = int(temp.hex(), 16)
                dp_command = int(Dp*100.0+0.5)
                print(f"Dp={dp_dsp}% is loaded!")
                time.sleep(0.5)
                if dp_dsp != dp_command:
                    print(f"Error in the DSP duty's command (dP: dsp={dp_dsp}, command={dp_command})")
                    continue
    
            # ============================================================================#
            # Setup the output to .csv files
            csvV = open(f"{sine_or_trap}_{it_H}_{it_d}_Data_Volt.csv", "w", newline='')
            writerV = csv.writer(csvV)
            csvI = open(f"{sine_or_trap}_{it_H}_{it_d}_Data_Curr.csv", "w", newline='')
            writerI = csv.writer(csvI)
            csvT = open(f"{sine_or_trap}_{it_H}_{it_d}_Data_Time.csv", "w", newline='')
            writerT = csv.writer(csvT)
            csvP = open(f"{sine_or_trap}_{it_H}_{it_d}_Parameters.csv", "w", newline='')
            writerP = csv.writer(csvP)
            print(f"File {it_d} opened \n")
                
            # ============================================================================#
            # Frequency iteration
            it_f = 1  # Number of frequency iteration
            for freq in 10**logF_range:
                
                t_f_loop = time.perf_counter()
    
                print(f"Freq={format(freq/1000, '.1f')} kHz starts:", end=" ")

                if sine_or_trap == 1:  # For trapezoidal
                    dsp.write(b'1')
                    time.sleep(0.5)
                    temp = dsp.read(dsp.inWaiting())
                    f_norm_dsp = int(temp.hex(), 16)  # We send log10(f)/nfperdec
                    f_norm_command = int(math.log10(freq)*Fpointsperdecade+0.5)
                    # +0.5 part only to make sure it is not flooring the .9999 cases
                    print(f"Fnorm={f_norm_dsp} is loaded!")
                    time.sleep(0.5)
                    if f_norm_dsp != f_norm_command:
                        print(f"Error in the normalized DSP frequency's command (dsp={f_norm_dsp}, command={f_norm_command})")
                        continue
                
                # Initialization of protections
                soft_start = 1  
                flag_PV_min = 1
                flag_PV_max = 1
                flag_V_min = 1
                flag_V_max = 1
                flag_I_max = 1
                flag_Vsignal_min = 1
                flag_Vsignal_max = 1
                flag_Vscale = 1
                flag_Iscale = 1
    
                # ============================================================================#
                # Initial calibration (identification of gain and current scale)

                scope.write('acquire:state RUN')  # run the scope when changing the voltage
                scope.query('*opc?')
                scope.write('acquire:stopafter RUNSTOP;state RUN')  # single
                scope.query('*opc?')
                
                if sine_or_trap == 0:  # Sinusoidal
                    dg.write(f":SOUR1:APPL:SIN {freq},{Vsignal_min},0,0")

                if sine_or_trap == 1:  # Trapezoidal
                    vdc.write(f"GPV {format(Vdc_min, '.2f')}\n".encode())
                    vdc.write("GOUT ON\n".encode())
                time.sleep(0.5)
                
                if sine_or_trap == 0:
                    scale_init_V = 0.2
                if sine_or_trap == 1:
                    scale_init_V = 2
                max_I_init = (Idc+0.01)*R_sense
                I_division_init = max_I_init/5*(1+vertical_scale_margin)
                index_I_scale_init = sum(I_division_init > scale_list_I)
                scale_init_I=scale_list_I[index_I_scale_init]            
                scope.write(f"CH2:scale {format(scale_init_V, '.2f')}")
                scope.write(f"CH3:scale {format(scale_init_I, '.2f')}")
    
                if sine_or_trap == 0:
                    scope.write(f"TRIGger:A:Level:CH2 0")
                if sine_or_trap == 1:
                    scope.write(f"CH2:scale 2")
                    scope.write(f"CH3:scale 0.2")
                    trigger = Vdc_min/2*(1-2*Dp+2*Dn)*N2/N1
                    scope.write(f"TRIGger:A:Level:CH2 {format(trigger, '.1f')}")
                r = scope.query('*opc?')
                time.sleep(1)   
    
                scope.write('acquire:state STOP')  # stop
                scope.query('*opc?')
                scope.write('acquire:stopafter RUNSTOP;state STOP')  # single
                scope.query('*opc?')
                
                # curve configuration for CH2
                scope.write('data:encdg SRIBINARY')  # signed integer
                scope.write('data:source CH2')
                scope.write('data:start 1')
                scope.write('data:stop 100000')
                scope.write('wfmoutpre:byt_nr 1')  # 16 bits per sample
                scope.write('WFMINPRE:XINCR 8e-9')  # 10ns resolution
                scope.query('*opc?')
                # data query for CH2
                bin_wave_v = scope.query_binary_values('curve?', datatype='b', container=np.array)
                wave_v = np.array(bin_wave_v, dtype='double')
                wave_v_avg = (wave_v[::10]+wave_v[1::10]+wave_v[2::10]+wave_v[3::10]+wave_v[4::10]+bin_wave_v[5::10]+wave_v[6::10]+bin_wave_v[7::10]+wave_v[8::10]+wave_v[9::10])/10.0
                scope.query('*opc?')
                # retrieve scaling factors
                v_scale = float(scope.query('wfmoutpre:ymult?'))  # volts / level
                scaled_wave_ch2 = wave_v_avg*v_scale
                
                # curve configuration for CH3
                scope.write('data:encdg SRIBINARY')  # signed integer
                scope.write('data:source CH3')
                scope.write('data:start 1')
                scope.write('data:stop 100000')
                scope.write('wfmoutpre:byt_nr 1')  # 2 byte per sample
                scope.write('WFMINPRE:XINCR 8e-9')  # 10ns resolution
                # data query for CH3
                bin_wave_c = scope.query_binary_values('curve?', datatype='b', container=np.array)
                wave_c = np.array(bin_wave_c, dtype='double')
                wave_c_avg = (wave_c[::10]+wave_c[1::10]+wave_c[2::10]+wave_c[3::10]+wave_c[4::10]+wave_c[5::10]+wave_c[6::10]+wave_c[7::10]+wave_c[8::10]+wave_c[9::10])/10.0
                scope.query('*opc?')
                # retrieve scaling factors (vertical only this time)
                v_scale = float(scope.query('wfmoutpre:ymult?'))  # volts / level
                scaled_wave_ch3 = wave_c_avg*v_scale/R_sense    
                    
                print("Calibration:", end =" ")
                Vmeas = max(abs(scaled_wave_ch2))
                Iac_meas = (max(scaled_wave_ch3)-min(scaled_wave_ch3))/2
                if sine_or_trap == 0:  # For sinusoidal
                    amp_meas = Vmeas*(N1/N2)
                    Lm = amp_meas/(2*pi*freq)*R_sense/Iac_meas
                if sine_or_trap == 1:  # For trapezoidal
                    amp_meas = Vmeas*(N1/N2)/(1+max(Dp-Dn, Dn-Dp))
                    Lm = amp_meas*(Dp*Dn+max(Dp*(1-Dp), Dn*(1-Dn)))/(2*freq)*R_sense/Iac_meas
                print(f"Lm={format(Lm*1e6, '.1f')} uH (Vpk={format(Vmeas, '.2f')} V, Iac={format(Iac_meas*1000, '.0f')} mA)", end=" ")
                if Lm < Lm_min:
                    input(f"Warning: Lm below {Lm_min*1e6} uH, ENTER to continue")
                if Lm > Lm_max:
                    input(f"Warning: Lm above {Lm_max*1e6} uH, ENTER to continue")
                    
                if sine_or_trap == 0:  # For sinusoidal
                    gain = amp_meas*2/Vsignal_min
                    print(f"gain={format(gain, '.1f')};", end=" ")
                    if gain < gain_min:
                        input(f"Warning: gain below {gain_min}, check if the power amplifier is on and the gain is at its maximum, ENTER to continue")
                    if gain > gain_max:
                        input(f"Warning: gain above {gain_max}, ENTER to continue")
                    
                time.sleep(1)
                                
                # ============================================================================#
                # Peak flux density iteration
                
                print(f"File {sine_or_trap}-{it_H}-{it_d} Round {it_f}:")
                it_B = 1  # Number of flux iteration
                amp = 0  # Initialization in case the flux vector is empty, to avoid a large voltage during the turn off
                for flux in 10**logB_range:
                    
                    t_B_loop = time.perf_counter()
                    
                    scope.write('acquire:state RUN')  # run the scope when changing the voltage
                    scope.query('*opc?')
                    scope.write('acquire:stopafter RUNSTOP;state RUN')  # single
                    scope.query('*opc?')
                        
                    if sine_or_trap == 0:  # Sinusoidal
                        amp = flux*N1*Ae*2*pi*freq  # Actual amplitude
                    else:  # Trapezoidal
                        if Dp >= Dn:
                            amp = flux*N1*Ae*freq*1/Dp*2/(1-Dp+Dn)  # v=Ae N dB/dt
                        elif Dn > Dp:
                            amp = flux*N1*Ae*freq*1/Dn*2/(1-Dn+Dp)

                    if sine_or_trap == 0:  # Sinusoidal
                        PV_expected = k_steinmetz*freq**alpha*flux**beta
                    else:  # Trapezoidal
                        if D0 == 0.0:
                            PV_expected = ki*freq**alpha*(2*flux)**beta*(Dp**(1-alpha)+Dn**(1-alpha))
                        else:
                            if Dp >= Dn:                                
                                flux_pk_1 = flux
                                flux_pk_2 = flux*(-(-1-Dp+Dn)*Dn)/((1-Dp+Dn)*Dp)
                            elif Dn > Dp:                                
                                flux_pk_1 = flux*(-(1-Dp+Dn)*Dp)/((-1-Dp+Dn)*Dn)
                                flux_pk_2 = flux    
                            PV_expected = ki*freq**alpha*(2*flux)**(beta-alpha)*((2*flux_pk_1)**alpha*Dp**(1-alpha)+2*abs(flux_pk_1-flux_pk_2)**alpha*D0**(1-alpha)+(2*flux_pk_2)**alpha*Dn**(1-alpha))

                    if sine_or_trap == 0:  # Sinusoidal
                        if amp < Vac_min:  # Skip if the AC voltage is too low
                            if flag_V_min == 1:
                                print(f"Measurement aborted (Vac below {Vac_min} V).")
                                flag_V_min = 0
                            continue
                        if amp > Vac_max:  # Continue if the AC voltage is too high
                            if flag_V_max == 1:
                                print(f"Measurement aborted (Vac above {Vac_max} V)")
                                flag_V_max = 0
                            continue
                    else:  # Trapezoidal
                        if amp < Vdc_min:  # Skip if the supply voltage is too low
                            if flag_V_min == 1:
                                print(f"Measurement aborted (Vdc below {Vdc_min} V)")
                                flag_V_min = 0
                            continue
                        if amp > Vdc_max:  # Continue if the supply voltage is too high
                            if flag_V_max == 1:
                                print(f"Measurement aborted (Vdc above {Vdc_max} V)")
                                flag_V_max = 0
                            continue
 
                    if sine_or_trap == 0:  # Sinusoidal
                        Vsignal = amp/gain*2  # Times two because the signal generator input is peak to peak
                        if Vsignal < Vsignal_min:  # Skip if the voltage in the signal generator is too low
                            if flag_Vsignal_min == 1:
                                print(f"Measurement aborted (AC signal below {Vsignal_min} V)")
                                flag_Vsignal_min = 0
                            continue
                        if Vsignal > Vsignal_max:  # Skip if the voltage in the signal generator is too low
                            if flag_Vsignal_max == 1:
                                print(f"Measurement aborted (AC signal above {Vsignal_max} V)")
                                flag_Vsignal_max = 0
                            continue
    
                    if PV_expected < PV_min:  # Skip if the losses will be too low
                        if flag_PV_min == 1:
                            print(f"Measurement aborted (PV below {PV_min/1000} kW/m3)")
                            flag_PV_min = 0
                        continue
                    if PV_expected > PV_max:  # Skip if the losses will be too high
                        if flag_PV_max == 1:
                            print(f"Measurement aborted (PV above {PV_max/1000} kW/m3)")
                            flag_PV_max = 0
                        continue
            
                    # ============================================================================#
                    # Define the scope single acquisition parameters

                    # Voltage vertical scale
                    if sine_or_trap == 0:  # For sinusoidal
                        max_V = amp*N2/N1
                    else:  # For trapezoidal
                        max_V = amp*max(1+Dp-Dn, 1+Dn-Dp)*N2/N1

                    # Current vertical scale
                    if sine_or_trap == 0:  # For sinusoidal
                        Iac = amp/(2*pi*freq*Lm)
                    else:  # For trapezoidal
                        Iac = amp*(max(Dp*(1-Dp+Dn), Dn*(1-Dn+Dp)))/(2*freq*Lm)
                    max_I = (Iac+Idc)*R_sense
    
                    if max_I > I_max:  # Skip if the losses will be too high
                        if flag_I_max == 1:
                            print(f"Measurement aborted (peak I above {I_max} A)")
                            flag_I_max = 0
                        continue
    
                    # Index for the scale in the list
                    V_division = max_V/5*(1+vertical_scale_margin)
                    I_division = max_I/5*(1+vertical_scale_margin)
                    index_V_scale = sum(V_division > scale_list_V)
                    index_I_scale = sum(I_division > scale_list_I)
                    # Protections
                    if index_V_scale > scale_list_V.size-1:
                        if flag_Vscale == 1:  # Display it only once
                            print('Measurement aborted (V scale out of range)')
                            flag_Vscale = 0
                        continue
                    if index_I_scale > scale_list_I.size-1:
                        if flag_Iscale == 1:
                            print('Measurement aborted (I scale out of range)')
                            flag_Iscale = 0
                        continue
                    scale_V = scale_list_V[index_V_scale]
                    scale_I = scale_list_I[index_I_scale]
                    # Trigger
                    if sine_or_trap == 0:  # For sinusoidal
                        trigger = 0.0
                    else:  # For trapezoidal
                        trigger = amp/2*(1-2*Dp+2*Dn)*N2/N1
        
                    # ============================================================================#
                    # Configuration of the scale values
    
                    scope.write(f"TRIGger:A:Level:CH2 {format(trigger, '.1f')}")  # Trigger set between the positive and zero voltages
                    scope.write(f"CH2:scale {scale_V}")
                    scope.write(f"CH3:scale {scale_I}")
                    r = scope.query('*opc?')
                    
                    # ============================================================================#
                    # Set the desired voltage
                    
                    # Set the power supply
                    if sine_or_trap == 1:  # For trapezoidal
                        if (soft_start == 1) and (amp > Vdc_min/0.3+0.1):  # Soft-start.
                            vdc.write(f"GPV {format(amp*0.3, '.2f')}\n".encode())
                            time.sleep(0.5)
                            vdc.write(f"GPV {format(amp*0.6, '.2f')}\n".encode())
                            time.sleep(0.5)
                        soft_start = 0
                        
                        vdc.write(f"GPV {format(amp, '.2f')}\n".encode())
                    
                    # Set the output of signal generator
                    if sine_or_trap == 0:  # For sinusoidal
                        dg.write(f":SOUR1:APPL:SIN {freq},{Vsignal},0,0")  # set the output of the signal generator
                    time.sleep(0.5)
                    # ============================================================================#
                    # Trigger and hold

                    scope.write('acquire:state STOP')  # stop
                    scope.query('*opc?')
                    scope.write('acquire:stopafter RUNSTOP;state STOP')  # single
                    scope.query('*opc?')
                
                    # ============================================================================#
                    # Measure CH2 and CH3 (volt and curr respectively)
            
                    # curve configuration for CH2
                    scope.write('data:encdg SRIBINARY')  # signed integer
                    scope.write('data:source CH2')
                    scope.write('data:start 1')
                    scope.write('data:stop 100000')
                    scope.write('wfmoutpre:byt_nr 1')  # 16 bits per sample
                    scope.write('WFMINPRE:XINCR 8e-9')  # 10ns resolution
                    scope.query('*opc?')
                    # data query for CH2
                    bin_wave_v = scope.query_binary_values('curve?', datatype='b', container=np.array)
                    wave_v = np.array(bin_wave_v, dtype='double')
                    wave_v_avg = (wave_v[::10]+wave_v[1::10]+wave_v[2::10]+wave_v[3::10]+wave_v[4::10]+bin_wave_v[5::10]+wave_v[6::10]+bin_wave_v[7::10]+wave_v[8::10]+wave_v[9::10]) / 10.0
                    scope.query('*opc?')
                    # retrieve scaling factors
                    v_scale = float(scope.query('wfmoutpre:ymult?'))  # volts / level
                    scaled_wave_ch2 = wave_v_avg*v_scale
                    
                    # curve configuration for CH3
                    scope.write('data:encdg SRIBINARY')  # signed integer
                    scope.write('data:source CH3')
                    scope.write('data:start 1')
                    scope.write('data:stop 100000')
                    scope.write('wfmoutpre:byt_nr 1')  # 2 byte per sample
                    scope.write('WFMINPRE:XINCR 8e-9')  # 10ns resolution
                    # data query for CH3
                    bin_wave_c = scope.query_binary_values('curve?', datatype='b', container=np.array)
                    wave_c = np.array(bin_wave_c, dtype='double')
                    wave_c_avg = (wave_c[::10]+wave_c[1::10]+wave_c[2::10]+wave_c[3::10]+wave_c[4::10]+wave_c[5::10]+wave_c[6::10]+wave_c[7::10]+wave_c[8::10]+wave_c[9::10]) / 10.0
                    scope.query('*opc?')
                    # retrieve scaling factors (vertical only this time)
                    v_scale = float(scope.query('wfmoutpre:ymult?'))  # volts / level
                    scaled_wave_ch3 = wave_c_avg*v_scale/R_sense
    
                    # retrieve time factors     
                    wfm_record = int(scope.query('wfmoutpre:nr_pt?'))
                    t_scale = float(scope.query('wfmoutpre:xincr?'))
                    total_time = t_scale * wfm_record
    
                    # Error checking
                    int(scope.query('*esr?'))
                    scope.query('allev?').strip()
                   
                    scaled_time = np.linspace(0.0, total_time, num=10000, endpoint=False)
                    
                    # ============================================================================#
                    # Clipping warning
                    if max(scaled_wave_ch2)*0.98>5*scale_V:
                        print(f"Warning: V above scale: {format(max(scaled_wave_ch2), '.2f')} V, clipping")
                        flag_clipping_V = 1
                    if min(scaled_wave_ch2)*0.98<-5*scale_V:
                        print(f"Warning: V below scale: {format(min(scaled_wave_ch2), '.2f')} V, clipping")
                        flag_clipping_V = 1
                    if max(scaled_wave_ch3)*0.98>5*scale_I:
                        print(f"Warning: I above scale: {format(max(scaled_wave_ch3)*1000, '.0f')} mA, clipping")
                        flag_clipping_I = 1
                    if min(scaled_wave_ch3)*0.98<-5*scale_I:
                        print(f"Warning: I below scale: {format(min(scaled_wave_ch3)*1000, '.0f')} mA, clipping")
                        flag_clipping_I = 1
                    
                    # ============================================================================#
                    # Write the data
                    writerV.writerow([float(k) for k in np.array(scaled_wave_ch2)])
                    writerI.writerow([float(k) for k in np.array(scaled_wave_ch3)])
                    writerT.writerow([float(k) for k in np.array(scaled_time)])
                    if sine_or_trap == 0:  # For sinusoidal
                        writerP.writerow([Hdc, float("NaN"), float("NaN"), freq, flux])
                    else:
                        writerP.writerow([Hdc, D0, Dp, freq, flux])
                                                    
                    # ============================================================================#
                    # Status display
                    t_end = time.perf_counter()  # Storing the time after this B sweep
                    
                    print(f"Measurement {it_test}:", end=" ")
                    print(f"B={format(flux*1000, '.1f')} mT, PVexp={format(PV_expected/1000, '.1f')} kW/m3,", end=" ")
                    if sine_or_trap == 0:
                        print(f"gain={format(gain, '.1f')},", end=" ")
                    else:
                        print(f"Vdc={format(amp, '.1f')} V,", end=" ")
                    print(f"Vosci={format(max_V, '.1f')}({format(scale_V, '.1f')}->{format(scale_V*5, '.1f')}) V, Iosci={format(max_I*1000, '.0f')}({format(scale_I*1000, '.0f')}->{format(scale_I*1000*5, '.0f')}) mA, Lm={format(Lm*1e6, '.1f')} uH,", end=" ")
                    print(f"Vpk={format(Vmeas, '.1f')} V, Iac={format(Iac*1000, '.0f')} mA,", end=" ")
                    print(f"Elapsed time={format(t_end-t_B_loop, '.1f')} sec")
                    it_test = it_test+1

                    # ============================================================================#
                    # Updating the calibration for the next test (identification of gain and inductance)
                    Vmeas = max(abs(scaled_wave_ch2))
                    Iac_meas = (max(scaled_wave_ch3)-min(scaled_wave_ch3))/2
                    Vreal = Vmeas*N1/N2
                    if flag_clipping_I == 0 and flag_clipping_V == 0: 
                        if sine_or_trap == 0:  # For sinusoidal
                            Lm = Vreal/(2*pi*freq)*R_sense/Iac_meas
                        if sine_or_trap == 1:  # For trapezoidal
                            Lm = Vreal/(2*freq)*(max(Dp*(1-Dp+Dn), Dn*(1-Dn+Dp)))/max(1+Dp-Dn, 1+Dn-Dp)*R_sense/Iac_meas
                        
                    if sine_or_trap == 0 and flag_clipping_V == 0:  # For sinusoidal
                        gain = Vreal*2/Vsignal
                        if gain<gain_min:
                            print(f"Warning: gain below {gain_min}")
                        if gain>gain_max:
                            print(f"Warning: gain above {gain_max}")
                    
                    flag_clipping_V = 0
                    flag_clipping_I = 0
                    
                    it_B = it_B+1
                    # end of the last flux iteration
                # ============================================================================#   
                # Soft turn-off to avoid premagnetization in the next test
                
                # Soft-turn-off of the power supply
                if sine_or_trap == 1:  # For trapezoidal
                    vdc.write(f"GPV {format(max(Vdc_min, 0.8*min(Vdc_max, amp)), '.2f')}\n".encode())
                    time.sleep(0.2)
                    vdc.write(f"GPV {format(max(Vdc_min, 0.6*min(Vdc_max, amp)), '.2f')}\n".encode())
                    time.sleep(0.2)
                    vdc.write(f"GPV {format(max(Vdc_min, 0.4*min(Vdc_max, amp)), '.2f')}\n".encode())
                    time.sleep(0.2)
                    vdc.write(f"GPV {format(max(Vdc_min, 0.2*min(Vdc_max, amp)), '.2f')}\n".encode())
                    time.sleep(0.2)
                    vdc.write(f"GPV {format(Vdc_min, '.2f')}\n".encode())
                    time.sleep(0.2)
                    vdc.write("GOUT OFF\n".encode())
                
                # Soft turn-off of the power amplifier
                if sine_or_trap == 0:  # For sinusoidal
                    dg.write(f":SOUR1:APPL:SIN {freq},{Vsignal_min+Vsignal*0.8},0,0")
                    time.sleep(0.5) 
                    dg.write(f":SOUR1:APPL:SIN {freq},{Vsignal_min+Vsignal*0.6},0,0")
                    time.sleep(0.5) 
                    dg.write(f":SOUR1:APPL:SIN {freq},{Vsignal_min+Vsignal*0.4},0,0")
                    time.sleep(0.5) 
                    dg.write(f":SOUR1:APPL:SIN {freq},{Vsignal_min+Vsignal*0.2},0,0")
                    time.sleep(0.5) 
                    dg.write(f":SOUR1:APPL:SIN {freq},{Vsignal_min},0,0")
    
                time.sleep(T_wait)  # 1 min delay between freq iteration to let the core cool down
                              
                # Time so far report
                t_end = time.perf_counter()
                t_run_so_far = t_run_so_far+(t_end-t_f_loop)
                t_left_run = t_run_so_far/it_f*(N_f_loops-it_f)
                print(f"Round {it_f} of {N_f_loops}; Elapsed time (f loop): {format(t_end-t_f_loop, '.0f')} sec;", end=" ")
                if it_f < N_f_loops:
                    print(f"Expected time left for this run: {format(t_left_run/60.0, '.1f')} min \n")

                it_f = it_f+1  # Number of frequency iterations
                # end of the last frequency iteration
            # ============================================================================#
            
            # Closing the csv files
            csvV.close()
            csvI.close()
            csvT.close()
            csvP.close()
            print(f"\nFile {sine_or_trap}-{it_H}-{it_d} saved;", end=" ")
            
            # Time so far report (only for Trapezoidal)
            t_end = time.perf_counter()
            if sine_or_trap == 1: # Triangular and Trapezoidal
                t_H_so_far = t_H_so_far+(t_end-t_run_loop)
                t_left_H = t_H_so_far/it_d*(N_runs-it_d)
                print(f"Run {it_d} of {N_runs}; Elapsed time (run): {format((t_end-t_run_loop)/60.0, '.1f')} min;")
                if it_d < N_runs:
                    print(f"Expected time left for this Hdc: {format(t_left_H/60.0, '.0f')} min ({format(t_left_H/3600.0, '.2f')} hrs) \n")
    
            it_d = it_d+1  # The number of the file to be saved, updated for each duty cycle
            # end of the last dutyP iteration
        # ============================================================================#
        # end of the last duty0 iteration
    # ============================================================================#
    # Status display and elapsed time for each HDC cycle
    t_end = time.perf_counter()
    print(f"DC bias test {it_H} of {N_Hdc_loops}; Elapsed time in this Hdc: {format((t_end-t_H_loop)/60.0, '.1f')} min ({format((t_end-t_H_loop)/3600.0, '.2f')} hrs) \n")
    
    it_H = it_H+1
    # end of the last Hdc iteration
# ============================================================================#
# Disconnect

bias.write('CH1:CURRent 0')  # sets current to 0 A for safety
bias.write('OUTPut CH1,OFF')

if sine_or_trap == 0:  # For sinusoidal
    dg.write(f":SOUR1:APPL:SIN {Fmin},{Vsignal_min},0,0")
    time.sleep(0.05) 
    dg.write('OUTP1 OFF')
    dg.close()
    print("Communication with the signal generator closed")    
if sine_or_trap == 1:  # For trapezoidal
    # vdc.write("GPV 5\n".encode())
    vdc.write(f"GPV {format(Vdc_min, '.2f')}\n".encode())
    vdc.write("GOUT OFF\n".encode())
    vdc.close()
    print("Communication with the voltage supply closed")
    dsp.close()
    print("Communication with the DSP closed")

scope.close()
print("Communication with the oscilloscope closed")
rm.close()
print("Resource manager closed")
bias.close()
print("Communication with the DC bias supply closed")

print(f"\nTotal number of test recorded: {it_test}; Last file recorded {it_H-1}-{it_d-1}; Tests finished!")

# Run this commands in case of errors during the tests
# Trapezoidal or Triangular
# vdc.close()
# dsp.close()
# Sinusoidal
# dg.close()
