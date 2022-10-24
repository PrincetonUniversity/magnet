from importlib.resources import path
import functools
import torch
import torch.nn as nn
from torch import Tensor
import math

#--------------------------------FNN------------------------------------------%
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

#--------------------------------LSTM-----------------------------------------%
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
        x = x[:, -1, :]
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

#--------------------------------Transformer----------------------------------%

class Transformer_encoder(nn.Module):

    def __init__(self, 
        input_size :int,
        dec_seq_len :int,
        max_seq_len :int,
        out_seq_len :int,
        dim_val :int,  
        n_encoder_layers :int,
        n_decoder_layers :int,
        n_heads :int,
        dropout_encoder,
        dropout_decoder,
        dropout_pos_enc,
        dim_feedforward_encoder :int,
        dim_feedforward_decoder :int,
        dim_feedforward_projecter :int,
        num_var: int=3
        ): 

        super().__init__() 

        self.dec_seq_len = dec_seq_len
        self.n_heads = n_heads
        self.out_seq_len = out_seq_len
        self.dim_val = dim_val

        self.encoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val))

        self.decoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val))

        self.linear_mapping = nn.Sequential(
            nn.Linear(dim_val, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, input_size))

        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc, max_len=max_seq_len)
        
        self.projector = nn.Sequential(
            nn.Linear(dim_val + num_var, dim_feedforward_projecter),
            nn.Tanh(),
            nn.Linear(dim_feedforward_projecter, dim_feedforward_projecter),
            nn.Tanh(),
            nn.Linear(dim_feedforward_projecter, dim_val))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            activation="relu",
            batch_first=True
            )

        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=n_encoder_layers, norm=None)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            activation="relu",
            batch_first=True
            )

        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=n_decoder_layers, norm=None)

    def forward(self, src: Tensor, tgt: Tensor, var: Tensor) -> Tensor:

        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src)
        enc_seq_len = 128
        var = var.unsqueeze(1).repeat(1,enc_seq_len,1)
        src = self.projector(torch.cat([src,var],dim=2))

        return src
        
    
class Transformer_decoder(nn.Module):

    def __init__(self, 
        input_size :int,
        dec_seq_len :int,
        max_seq_len :int,
        out_seq_len :int,
        dim_val :int,  
        n_encoder_layers :int,
        n_decoder_layers :int,
        n_heads :int,
        dropout_encoder,
        dropout_decoder,
        dropout_pos_enc,
        dim_feedforward_encoder :int,
        dim_feedforward_decoder :int,
        dim_feedforward_projecter :int,
        num_var: int=3
        ): 

        super().__init__() 

        self.dec_seq_len = dec_seq_len
        self.n_heads = n_heads
        self.out_seq_len = out_seq_len
        self.dim_val = dim_val

        self.encoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val))

        self.decoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val))

        self.linear_mapping = nn.Sequential(
            nn.Linear(dim_val, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, input_size))

        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc, max_len=max_seq_len)
        
        self.projector = nn.Sequential(
            nn.Linear(dim_val + num_var, dim_feedforward_projecter),
            nn.Tanh(),
            nn.Linear(dim_feedforward_projecter, dim_feedforward_projecter),
            nn.Tanh(),
            nn.Linear(dim_feedforward_projecter, dim_val))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            activation="relu",
            batch_first=True
            )

        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=n_encoder_layers, norm=None)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            activation="relu",
            batch_first=True
            )

        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=n_decoder_layers, norm=None)

    def forward(self, src: Tensor, tgt: Tensor, var: Tensor) -> Tensor:

        tgt = self.decoder_input_layer(tgt)
        tgt = self.positional_encoding_layer(tgt)
        batch_size = src.size()[0]
        tgt_mask = generate_square_subsequent_mask(sz1=self.out_seq_len, sz2=self.out_seq_len)
        output = self.decoder(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=None
            )
                
        output= self.linear_mapping(output)

        return output

class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz1: int, sz2: int) -> Tensor:
    return torch.triu(torch.ones(sz1, sz2) * float('-inf'), diagonal=1)


@functools.lru_cache(maxsize=8)
def model_transformer(material, device='cpu'):
    
    material = 'N87' #just for now
    
    with path('magnet.models', f'Model_{material}_Transformer.sd') as sd_file:
        device = torch.device('cpu')
        state_dict = torch.load(sd_file, map_location=device)
        
    net_decoder = Transformer_decoder(
          dim_val=24,
          input_size=1, 
          dec_seq_len=129,
          max_seq_len=129,
          out_seq_len=129, 
          n_decoder_layers=1,
          n_encoder_layers=1,
          n_heads=4,
          dropout_encoder=0.0, 
          dropout_decoder=0.0,
          dropout_pos_enc=0.0,
          dim_feedforward_encoder=40,
          dim_feedforward_decoder=40,
          dim_feedforward_projecter=40).float().to(device)
        
    net_encoder = Transformer_encoder(
          dim_val=24,
          input_size=1, 
          dec_seq_len=129,
          max_seq_len=129,
          out_seq_len=129, 
          n_decoder_layers=1,
          n_encoder_layers=1,
          n_heads=4,
          dropout_encoder=0.0, 
          dropout_decoder=0.0,
          dropout_pos_enc=0.0,
          dim_feedforward_encoder=40,
          dim_feedforward_decoder=40,
          dim_feedforward_projecter=40).float().to(device)
        
    net_decoder.load_state_dict(state_dict, strict=True)
    net_decoder.eval() 
    net_encoder.load_state_dict(state_dict, strict=True)
    net_encoder.eval() 
    
    with path('magnet.models', f'Norm_{material}.pt') as norm_file:
        norm = torch.load(norm_file)
    
    return net_encoder, net_decoder, norm

