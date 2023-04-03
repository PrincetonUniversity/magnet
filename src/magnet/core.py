import numpy as np
import torch
from magnet.net import model, model_lstm, model_transformer
from magnet.io import load_hull
from magnet import config as c
import time


def default_units(prop):  # Probably we are not going to need the default units
    prop = prop.lower().strip()
    return {
        'frequency': 'Hz',
        'flux_density': 'T',
        'power_loss': '[W/m^3]'
    }[prop]


def plot_label(prop):
    prop = prop.lower().strip()
    return {
        'frequency_khz': 'Frequency [kHz]',
        'flux_density_mt': 'AC Flux Density [mT]',
        'power_loss_kw/m3': 'Power Loss [kW/m^3]',
        'frequency': 'Frequency [Hz]',
        'flux_density': 'AC Flux Density [T]',
        'power_loss': 'Power Loss [W/m^3]',
        'duty_ratio': 'Duty Ratio'
    }[prop]


def plot_title(prop):
    prop = prop.lower().strip()
    return {
        'frequency_khz': 'Frequency',
        'flux_density_mt': 'Flux Density',
        'power_loss_kw/m3': 'Power Loss',
        'frequency': 'Frequency',
        'flux_density': 'Flux Density',
        'power_loss': 'Power Loss'
    }[prop]


def core_loss_default(material, freq, flux, temp, bias, duty=None, batched = False):
    Eq = load_hull(material)
    
    if not batched:
        if duty is None:  # Sinusoidal
            phase = 0
            dd = 0.5
        elif type(duty) is list:  # Trapezoidal
            if duty[0] <= duty[1]:
                phase = 0.5 - duty[0]/2
            else:
                phase = -duty[0]/2
            dd = duty[0]
        else:  # type(duty) == 'float' -> Triangular
            if duty <= 0.5:
                phase = 0.5 - duty/2
            else:
                phase = -duty/2
            dd = duty
        
        bdata = bdata_generation(flux, duty)
        bdata = np.roll(bdata, np.int_(phase * c.streamlit.n_nn))
        hdata = BH_Transformer(material, freq, temp, bias, bdata)
        core_loss = loss_BH(bdata, hdata, freq)     
        
        point = np.array([freq, flux, bias, temp, dd])
        not_extrapolated = point_in_hull(point,Eq)
        
        return core_loss, not_extrapolated
    
    else:
        bdata = np.zeros(shape=(len(freq), c.streamlit.n_nn))
        dd = np.zeros(shape=(len(freq), 1))
        not_extrapolated = [False for i in range(len(freq))]
        
        for k in range(len(freq)):
            if duty[k] is None:  # Sinusoidal
                phase = 0
                dd[k] = 0.5
            elif type(duty[k]) is list:  # Trapezoidal
                if duty[k][0] <= duty[k][1]:
                    phase = 0.5 - duty[k][0]/2
                else:
                    phase = -duty[k][0]/2
                dd[k] = duty[k][0]
            else:  # type(duty) == 'float' -> Triangular
                if duty[k] <= 0.5:
                    phase = 0.5 - duty[k]/2
                else:
                    phase = -duty[k]/2
                dd[k] = duty[k]
                
            bdata[k,:] = bdata_generation(flux[k], duty[k])
            bdata[k,:] = np.roll(bdata[k,:], np.int_(phase * c.streamlit.n_nn))
            
        hdata = BH_Transformer(material, freq, temp, bias, bdata)
        core_loss = np.zeros(shape=len(freq))
        
        for k in range(len(freq)):
            core_loss[k] = loss_BH(bdata[k,:], hdata[k,:], freq[k])
            
        for k in range(len(freq)):
            point = np.array([freq[k], flux[k], bias[k], temp[k], dd[k]])
            not_extrapolated[k] = point_in_hull(point,Eq)
            
        return core_loss, not_extrapolated


def core_loss_arbitrary(material, freq, flux, temp, bias, duty):
    bdata_pre = np.interp(np.linspace(0, 1, c.streamlit.n_nn), np.array(duty), np.array(flux))
    bdata = bdata_pre - np.average(bdata_pre)
    hdata = BH_Transformer(material, freq, temp, bias, bdata)
    core_loss = loss_BH(bdata, hdata, freq)
    not_extrapolated = [False for i in range(len(bdata))]

    return core_loss, not_extrapolated


def BH_Transformer(material, freq, temp, bias, bdata):
    
    net_encoder, net_decoder, norm = model_transformer(material)
        
    bdata = torch.from_numpy(np.array(bdata)).float()
    bdata = (bdata-norm[0])/norm[1]
    
    freq = np.log10(freq)
    freq = torch.from_numpy(np.array(freq)).float()
    freq = (freq-norm[2])/norm[3]
    
    temp = torch.from_numpy(np.array(temp)).float()
    temp = (temp-norm[4])/norm[5]
    
    bias = torch.from_numpy(np.array(bias)).float()
    bias = (bias-norm[6])/norm[7]
        
    if bdata.dim()==1:
        BATCHED = False
    else:
        BATCHED = True
    
    if not BATCHED:
        bdata = bdata.unsqueeze(0).unsqueeze(2)
        freq = freq.unsqueeze(0).unsqueeze(1)
        temp = temp.unsqueeze(0).unsqueeze(1)
        bias = bias.unsqueeze(0).unsqueeze(1)
    else:
        bdata = bdata.unsqueeze(2)
        freq = freq.unsqueeze(1)
        temp = temp.unsqueeze(1)
        bias = bias.unsqueeze(1)
        
    outputs = torch.zeros(bdata.size()[0], bdata.size()[1]+1, 1)
    tgt = (torch.rand(bdata.size()[0], bdata.size()[1]+1, 1)*2-1)
    tgt[:, 0, :] = 0.1*torch.ones(tgt[:, 0, :].size())
        
    src = net_encoder(src=bdata, tgt=tgt, var=torch.cat((freq, temp, bias), dim=1))
    
    for t in range(1, bdata.size()[1]+1):   
        outputs = net_decoder(src=src, tgt=tgt, var=torch.cat((freq, temp, bias), dim=1))
        tgt[:, t, :] = outputs[:, t-1, :].detach()
        
    outputs = net_decoder(src, tgt, torch.cat((freq, temp, bias), dim=1))
    
    hdata = outputs[:, :-1, :]*norm[9]+norm[8]
            
    if not BATCHED:
        hdata = hdata.squeeze(2).squeeze(0).detach().numpy()
    else:
        hdata = hdata.squeeze(2).detach().numpy()
    
    return hdata


def loss_BH(bdata, hdata, freq):
    loss = freq * np.trapz(hdata, bdata)
    return loss


def bdata_generation(flux, duty=None, n_points=c.streamlit.n_nn):
    # Here duty is not needed, but it is convenient to call the function recursively

    if duty is None:  # Sinusoidal
        bdata = flux * np.sin(np.linspace(0.0, 2 * np.pi, n_points+1))
        bdata = bdata[:-1]
    elif type(duty) is list:  # Trapezoidal
        assert len(duty) == 3, 'Please specify 3 values as the Duty Ratios'
        assert np.all((0 <= np.array(duty)) & (np.array(duty) <= 1)), 'Duty ratios should be between 0 and 1'

        if duty[0] > duty[1]:
            # Since Bpk is proportional to the voltage, and the voltage is proportional to (1-dp+dN) times the dp
            bp = flux
            bn = -bp * ((-1 - duty[0] + duty[1]) * duty[1]) / (
                    (1 - duty[0] + duty[1]) * duty[0])  # proportional to (-1-dp+dN)*dn
        else:
            bn = flux  # proportional to (-1-dP+dN)*dN
            bp = -bn * ((1 - duty[0] + duty[1]) * duty[0]) / (
                    (-1 - duty[0] + duty[1]) * duty[1])  # proportional to (1-dP+dN)*dP
        bdata = np.interp(np.linspace(0, 1, n_points+1),
                          np.array([0, duty[0], duty[0] + duty[2], 1 - duty[2], 1]),
                          np.array([-bp, bp, bn, -bn, -bp]))
        bdata = bdata[:-1]
    else:  # type(duty) == 'float' -> Triangular
        assert 0 <= duty <= 1.0, 'Duty ratio should be between 0 and 1'
        bdata = np.interp(np.linspace(0, 1, n_points+1), np.array([0, duty, 1]), np.array([-flux, flux, -flux]))
        bdata = bdata[:-1]

    return bdata

def point_in_hull(point, Eq, tolerance=1e-10):
    # Determine whether a given point lies in the convex hull or not
    return all(
        (np.dot(coeff[:-1], point) + coeff[-1] <= tolerance)
        for coeff in Eq)
