import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xmlrpc.client as xmlrpclib
import os
import os.path

class CircuitModel(object):
    cntInst = 0

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

    # Display schematic
    def displaySch(self, path):
        st.image(Image.open(os.path.join(path, 'graphics', f'{self.Name}_sch.png')), width=500)

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
        self.Time = Data_raw['Time']
        self.V = Data_raw['Values'][0]
        self.I = Data_raw['Values'][1]
        self.H = Data_raw['Values'][2]
        self.B = Data_raw['Values'][3]
        self.Ploss = Data_raw['Values'][4][-1]/self.mag.Ac/self.mag.lc/1000
        return self.B,self.Time

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
            title_text="<b>Simulated Operation Waveforms</b>",
            title_x=0.5,
            legend=dict(yanchor="bottom", y=0, xanchor="right", x=0.935)
        )
        
        fig.update_xaxes(title_text="Time [s]")

        fig.update_yaxes(title_text="Primary Winding Voltage [V]", nticks=5, secondary_y=False)
        fig.update_yaxes(title_text="Primary Winding Current [A]", nticks=5, secondary_y=True)
        
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
        st.image(Image.open(os.path.join(path, 'graphics', f'{self.Name}.png')), width=300)

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
