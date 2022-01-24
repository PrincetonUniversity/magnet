from importlib.resources import path
import functools
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, NN_ARCHITECTURE):
        super(Net, self).__init__()
        # Define a fully connected layers model with three inputs (frequency, flux density, duty ratio)
        # and one output (power loss).
        self.layers = nn.Sequential(
            nn.Linear(NN_ARCHITECTURE[0], NN_ARCHITECTURE[1]),
            nn.ReLU(),
            nn.Linear(NN_ARCHITECTURE[1], NN_ARCHITECTURE[2]),
            nn.ReLU(),
            nn.Linear(NN_ARCHITECTURE[2], NN_ARCHITECTURE[3]),
            nn.ReLU(),
            nn.Linear(NN_ARCHITECTURE[3], NN_ARCHITECTURE[4])
        )

    def forward(self, x):
        return self.layers(x)

    # Returns number of trainable parameters in a network
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@functools.lru_cache(maxsize=8)
def model(material, waveform, device='cpu'):
    with path('magnet.models', f'Model_{material}_{waveform}.sd') as sd_file:
        state_dict = torch.load(sd_file)

    if waveform == 'Sinusoidal':
        NN_ARCHITECTURE = [2, 24, 24, 24, 1]
        neural_network = Net(NN_ARCHITECTURE).double().to(device)
    elif waveform == 'Trapezoidal':
        NN_ARCHITECTURE = [6, 24, 24, 24, 1]
        neural_network = Net(NN_ARCHITECTURE).double().to(device)
        
    neural_network.load_state_dict(state_dict, strict=True)
    neural_network.eval() 
    
    return neural_network


class Net_LSTM(nn.Module):
    def __init__(self):
        super(Net_LSTM, self).__init__()
        self.lstm = nn.LSTM(1, 32, num_layers=1, batch_first=True, bidirectional=False)
        self.fc1 = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),

            nn.Linear(16, 16),
            nn.LeakyReLU(0.2),

            nn.Linear(16, 15)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(16, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1)
            )
    
    def forward(self, x, freq):
        x, _ = self.lstm(x)
        x = x[:, -1, :] # Get last output only (many-to-one)
        x = self.fc1(x)
        y = self.fc2(torch.cat((x,freq),1))
        return y

    def count_parameters(self):
        # Returns number of trainable parameters in a network
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@functools.lru_cache(maxsize=8)
def model_lstm(material, device='cpu'):
    with path('magnet.models', f'Model_{material}_LSTM.sd') as sd_file:
        device = torch.device('cpu')
        state_dict = torch.load(sd_file, map_location=device)

    neural_network = Net_LSTM().double().to(device)
        
    neural_network.load_state_dict(state_dict, strict=True)
    neural_network.eval() 
    
    return neural_network
