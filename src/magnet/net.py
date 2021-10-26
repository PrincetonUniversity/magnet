from importlib.resources import path
import functools
import torch
import torch.nn as nn

NN_ARCHITECTURE = [3, 15, 15, 9, 1]  # Number of neurons in each layer


class Net(nn.Module):
    def __init__(self):
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
def model(material, device='cpu'):
    with path('magnet.models', f'{material}.sd') as sd_file:
        state_dict = torch.load(sd_file)

    neural_network = Net().double().to(device)
    neural_network.load_state_dict(state_dict, strict=True)
    neural_network.eval()  # TODO: ??

    return neural_network