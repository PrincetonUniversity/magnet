def search(DATA,Fmin,Fmax,Bmin,Bmax,Dmin,Dmax):
    NEWDATA = dict()
    NEWDATA['Material'] = DATA['Material']
    NEWDATA['Core_Shape'] = []
    NEWDATA['Effective_Area']= []
    NEWDATA['Effective_Volume'] = []
    NEWDATA['Effective_Length'] = []
    NEWDATA['Primary_Turns'] = []
    NEWDATA['Secondary_Turns'] = []
    NEWDATA['Excitation_Type'] = []
    NEWDATA['Sampling_Time'] = []
    NEWDATA['Duty_Ratio'] = []
    NEWDATA['Voltage'] = []
    NEWDATA['Current'] = []
    NEWDATA['Time'] = []
    NEWDATA['Frequency'] = []
    NEWDATA['Flux_Density'] = []
    NEWDATA['Power_Loss'] = []
    
    j=0
    for i in range(len(DATA['Frequency'])):
        if ((DATA['Frequency'][i]>=Fmin) and
            (DATA['Frequency'][i]<=Fmax) and
            (DATA['Flux_Density'][i]>=Bmin) and
            (DATA['Flux_Density'][i]<=Bmax) and
            (DATA['Duty_Ratio'][i]>=Dmin) and
            (DATA['Duty_Ratio'][i]<=Dmax)):
            
            # NEWDATA['Voltage'].append(DATA['Voltage'][i])
            # NEWDATA['Current'].append(DATA['Current'][i])
            # NEWDATA['Time'].append(DATA['Time'][i])
            NEWDATA['Duty_Ratio'].append(DATA['Duty_Ratio'][i])
            NEWDATA['Frequency'].append(DATA['Frequency'][i])
            NEWDATA['Flux_Density'].append(DATA['Flux_Density'][i])
            NEWDATA['Power_Loss'].append(DATA['Power_Loss'][i])
            j=j+1
    return NEWDATA