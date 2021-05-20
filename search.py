def sinesearch(DATA,Fmin,Fmax,Bmin,Bmax):
    NEWDATA = dict()
    NEWDATA['Duty_Ratio'] = []
    NEWDATA['Frequency'] = []
    NEWDATA['Flux_Density'] = []
    NEWDATA['Power_Loss'] = []
    
    for i in range(len(DATA['Frequency'])):
        if ((DATA['Frequency'][i]>=Fmin) and
            (DATA['Frequency'][i]<=Fmax) and
            (DATA['Flux_Density'][i]>=Bmin) and
            (DATA['Flux_Density'][i]<=Bmax)):
            
            NEWDATA['Duty_Ratio'].append(DATA['Duty_Ratio'][i])
            NEWDATA['Frequency'].append(DATA['Frequency'][i])
            NEWDATA['Flux_Density'].append(DATA['Flux_Density'][i])
            NEWDATA['Power_Loss'].append(DATA['Power_Loss'][i])
    
    return NEWDATA


def taglsearch(DATA,Fmin,Fmax,Bmin,Bmax,Duty_Array,Duty_Margin):
    NEWDATA = dict()
    NEWDATA['Duty_Ratio'] = []
    NEWDATA['Frequency'] = []
    NEWDATA['Flux_Density'] = []
    NEWDATA['Power_Loss'] = []
    
    for duty in Duty_Array:
        Dmin=duty-Duty_Margin
        Dmax=duty+Duty_Margin
 
        for i in range(len(DATA['Frequency'])):
            if ((DATA['Frequency'][i]>=Fmin) and
                (DATA['Frequency'][i]<=Fmax) and
                (DATA['Flux_Density'][i]>=Bmin) and
                (DATA['Flux_Density'][i]<=Bmax) and
                (DATA['Duty_Ratio'][i]>=Dmin) and
                (DATA['Duty_Ratio'][i]<=Dmax)):
                
    
                NEWDATA['Duty_Ratio'].append(DATA['Duty_Ratio'][i])
                NEWDATA['Frequency'].append(DATA['Frequency'][i])
                NEWDATA['Flux_Density'].append(DATA['Flux_Density'][i])
                NEWDATA['Power_Loss'].append(DATA['Power_Loss'][i])
    
    return NEWDATA
    
    