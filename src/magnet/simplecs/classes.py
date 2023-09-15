import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xmlrpc.client as xmlrpclib
import os
import os.path
import numpy as np
from magnet import config as c

import random

class CircuitModel(object):
    cntInst = 0
    op  = ""
    def __init__(self, name_init):
        self.Name = name_init
        self.opt = {'ModelVars': {}}
        CircuitModel.cntInst += 1

    def __del__(self):
        if CircuitModel.cntInst > 0:
            CircuitModel.cntInst -= 1

    # Set parameters
    def setParam(self, param_init):
        self.opt['ModelVars']['Vi'] = param_init["Vi"]
        self.opt['ModelVars']['Vo'] = param_init["Vo"]
        self.opt['ModelVars']['Ro'] = param_init["Ro"]
        self.opt['ModelVars']['Lk'] = param_init["Lk"]
        self.opt['ModelVars']['fsw'] = param_init["fsw"]
        self.opt['ModelVars']['duty'] = param_init["duty"]
        self.opt['ModelVars']['ph'] = param_init["ph"]
        self.opt['ModelVars']['C'] = 100e-6
    def __str__(self):
        # print("name", self.Name)
        # print("opt" , self.opt)
        # print("cntInst", self.cntInst)
        return self.Name+","+str(self.opt)
   
    # Display schematic
    def displaySch(self, path):
        st.image(Image.open(os.path.join(path, 'graphics', f'{self.Name}_sch.png')), width=400)

    def simulation(self, name, opt):
        """
            Python Based MagNet compatible simple case simulator
                + Buck convert CCM mode
                + Boost Converter CCM mode
                + DCM return wrong answer? It will be added. 
        """
        modeled = ["Buck", "Boost"]

        values = opt["ModelVars"]
        V = []
        R = values['Ro']

        T = 1/values['fsw']
        time = np.linspace(0,6 ,1000)*T
        time = time[:-1]
        if name == "Buck":
            vi = values['Vi']
            vo = values['duty']*values['Vi']

            V=np.where(time%T < values['duty']*T, vi - vo, -vo)
            Po = vo**2/R

            Imean = Po/vo
        if name == "Boost":
            vi = values['Vi']
            vo = values['Vi']/(1-values['duty'])
            V=np.where(time%T <= values['duty']*T, vi, vi-vo)
            Po = vo**2/R
            Imean = Po/vi
        V = np.array(V)
        dt = np.diff(time)
        dt = np.append(dt, dt[-1])

        mu_0 = 4*np.pi*10**-7
        lc = values['lc']
        lg = values['lg']
        N = values['Np']
        Ac = values["Ac"]
        mu =values['mu_r']*mu_0
        Rg = lg/(Ac*mu_0)
        Rc = lc/(Ac*mu)

        Req = Rg+Rc
        Inductance = values['Np']**2/Req
     
        mu_eff = Inductance*(lc+lg)/(Ac*N**2)
        I = np.nancumsum(V*dt)/Inductance # Ripple 
        Icenter = (np.max(I)+np.min(I))/2
        I_ = I + Imean - Icenter  ## add to average value

        I = I_
        H = I*values['Np']*(Rc/Req)/values['lc'] 

        B = np.nancumsum(V*dt)/(values['Np']*values['Ac'])

        Bmean = mu*np.mean(H)
        Bcenter= (np.max(B)-np.min(B))/2

        B = B + Bmean - Bcenter
        B = np.round(B,  decimals=5)

        output = {}
        output["B"] = B
        output["H"] = H
        output["V"] = V
        output["I"] = I
        output["Time"]= time
        output["Ploss"]=np.sum(B*H)*0.0 # Return 0 (To compute IGSE in final implementation)
        return output

    # Configure magnetic model
    def setMagModel(self, mag_init, material_init):
        self.mag = mag_init
        self.mag.configMaterialModel(material_init)
        self.opt['ModelVars']['lc'] = self.mag.lc
        self.opt['ModelVars']['Ac'] = self.mag.Ac
        self.opt['ModelVars']['lg'] = self.mag.lg
        self.opt['ModelVars']['Np'] = self.mag.Np
        self.opt['ModelVars']['Ns'] = self.mag.Ns
        self.opt['ModelVars']['mu_r'] = self.mag.material.mu_r
        self.opt['ModelVars']['iGSE_ki'] = self.mag.material.iGSE_ki
        self.opt['ModelVars']['iGSE_alpha'] = self.mag.material.iGSE_alpha
        self.opt['ModelVars']['iGSE_beta'] = self.mag.material.iGSE_beta

    # Run PLECS steady state analysis
    def steadyRun(self, path):
        server = xmlrpclib.Server("http://localhost:1080/RPC2")
        path_model = path + "/models"
        server.plecs.load(path_model + '/' + self.Name)
        Data_raw = server.plecs.analyze(self.Name, "Steady-State Analysis", self.opt)
        server.plecs.close(self.Name)
        self.op = "PLECS backend"
        self.Time = Data_raw['Time']
        self.V = Data_raw['Values'][0]
        self.I = Data_raw['Values'][1]
        self.H = Data_raw['Values'][2]
        self.B = Data_raw['Values'][3]
        self.Ploss = Data_raw['Values'][4][-1]/self.mag.Ac/self.mag.lc/1000
        return self.B,self.H,self.Time

     # Run python steady state analysis
    def steadyRun_py(self, path):
        """
        Implements SteadyRun function. Using a pure python backend, to avoid call Plecs
            + Under development
            + Given fixed sets of typologies, it should be possible to generate B and H, time
            outputs almost similar to what plecs out. 
            + Currently Boost and Buck converter under CCM operation are under test
        """
        self.op = "Python Backend"
        if self.Name in ["Buck", "Boost"]:
            Data_raw = self.simulation(self.Name, self.opt)
            self.Time = Data_raw["Time"]
            self.V = Data_raw["V"]
            self.I = Data_raw["I"]
            self.H = Data_raw["H"]
            self.B = Data_raw["B"]
            self.Ploss = Data_raw['Ploss']/self.mag.Ac/self.mag.lc/1000
            self.op = "Pure Python3"
        else:
            print("To be implemented") 
        # print("N points , ", len(self.B))
        # print("Python3 based loss : ", self.Ploss)
        return self.B,self.H,self.Time

    # Display waveform
    def displayWfm(self):
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=self.Time, y=self.V, name="Voltage", line=dict(color='firebrick', width=3)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=self.Time, y=self.I, name="Current", line=dict(color='mediumslateblue', width=3)),
            secondary_y=True,
        )
        
        fig.update_layout(
            title_text="<b>Simulated Operation Waveforms {}</b>".format(self.op),
            title_x=0.2,
            legend=dict(yanchor="bottom", y=0, xanchor="right", x=0.935)
        )
        
        fig.update_xaxes(title_text="Time [s]")

        fig.update_yaxes(title_text="Primary Winding Voltage [V]", nticks=5, secondary_y=False)
        fig.update_yaxes(title_text="Primary Winding Current [A]", nticks=5, secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
    def displayBH(self):
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        # print("self.Binterp", np.mean(self.Binterp),np.mean(self.bias))
        # print("self.Hinterp", np.mean(self.Hinterp),np.mean(c.streamlit.n_nn))
        fig.add_trace(
            go.Scatter(
                x=np.tile(self.Hinterp + self.bias * np.ones(c.streamlit.n_nn), 2),
                y=np.tile((self.Binterp + self.bias * self.mag.material.mu_r * c.streamlit.mu_0 * np.ones(c.streamlit.n_nn)) * 1e3, 2),
                line=dict(color='mediumslateblue', width=4),
                name="Predicted B-H Loop"),
            secondary_y=False,
        )
        
        fig.update_layout(
            title_text="<b>Predicted B-H Loop</b>",
            title_x=0.5,
            legend=dict(yanchor="bottom", y=0, xanchor="right", x=0.935)
        )

        fig.update_yaxes(title_text="B - Flux Density [mT]", zeroline=True, zerolinewidth=1.5, zerolinecolor='gray')
        fig.update_xaxes(title_text="H - Field Strength [A/m]", zeroline=True, zerolinewidth=1.5, zerolinecolor='gray')
        st.plotly_chart(fig, use_container_width=True)


class MagModel(object):
    cntInst = 0

    # Constructor
    def __init__(self, name_init):
        self.Name = name_init
        MagModel.cntInst += 1

    # Destructor
    def __del__(self):
        if MagModel.cntInst > 0:
            MagModel.cntInst -= 1

    # Set Parameters
    def setParam(self, param_init):
        self.lc = param_init["lc"]
        self.Ac = param_init["Ac"]
        self.lg = param_init["lg"]
        self.Np = param_init["Np"]
        self.Ns = param_init["Ns"]

    # Display schematic
    def displaySch(self, path):
        st.image(Image.open(os.path.join(path, 'graphics', f'{self.Name}.png')), width=200)

    # Configure material model
    def configMaterialModel(self, material_init):
        self.material = material_init

    # Calculate core loss
    def calCoreLoss(self):
        self.core_loss = 0


class CoreMaterial(object):
    cntInst = 0

    # Constructor
    def __init__(self, name_init):
        self.Name = name_init
        MagModel.cntInst += 1

    # Destructor
    def __del__(self):
        if MagModel.cntInst > 0:
            MagModel.cntInst -= 1

    # Set Parameters
    def setParam(self, param_init):
        self.mu_r = param_init["mu_r"]
        self.iGSE_ki = param_init["iGSE_ki"]
        self.iGSE_alpha = param_init["iGSE_alpha"]
        self.iGSE_beta = param_init["iGSE_beta"]
