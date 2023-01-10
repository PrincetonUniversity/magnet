import os.path
from PIL import Image
import streamlit as st

STREAMLIT_ROOT = os.path.dirname(__file__)


def ui_tutorial(m):

    st.title('MagNet-AI Tutorial')
    st.write("""
        Here we provide a few step-by-step tutorials to explain how MagNet AI work. The complete codes and data files can be downloaded from the [MagNet GitHub](https://github.com/PrincetonUniversity/Magnet) repository under "tutorial". 
    """)
    st.write("""    
        For PyTorch beginners, we recommend using [Google Golab](https://colab.research.google.com/) (free) to simplify the learning process.
    """)
    st.header('Tutorial #1: Train a Seq-to-Seq Neural Network to Predict B-H Loops')
    st.write("""
        Figure 1 illustrates the structure and the data flow of an encoder-projector-decoder neural network architecture. The general concept of the encoder-projector-decoder architecture is to map a time series into another time series while mixing other information about the operating conditions. As an example, we select $B(t)$ as the input sequence and $H(t)$ as the output sequence. Waveforms of $B(t)$ and $H(t)$ define the basic shape of hysteresis loops. The B-H loops are also significantly affected by other scalar inputs, including the frequency $f$, the temperature $T$, and the dc bias $H_{dc}$. Therefore, an additional projector is implemented between the encoder and the decoder to take these scalar inputs into consideration and predict the $B$-$H$ loop under different operating conditions.
    """)
    st.image(Image.open(os.path.join(STREAMLIT_ROOT, 'img', 'seq-to-seq.jpg')), width=600, caption='Fig. 1: Data flow of the encoder-projector-decoder structure')
    st.write("""
        The encoder takes the $B(t)$ sequence as input and maps the sequence into a vector of fixed dimension. It captures the sequential information and temporal correlation within the sequence, such as the shape, the sequence of the patterns, the amplitude, and the relative changing rate of the given excitation waveform. The output of the encoder is the hidden state vectors, which encapsulate all the necessary information extracted from the input sequence and map it to a domain with hidden states. The hidden state vectors are then fed into the projector and modified based on the additional scalar inputs (frequency $f$, temperature $T$, and dc bias $H_{dc}$). The rationale behind the projector is that the shape of $B$--$H$ loop is not only determined by the shape of $B(t)$ sequence itself, but also by many other factors. Finally, the modified hidden state vectors are processed by the decoder, where the decoder makes predictions of the output sequence $H(t)$. During the model inference, the expected response sequence is generated in an auto-regressive way, which means the prediction of each step is produced based on both the current hidden states vectors and all the predictions that are already produced, such that the temporal information of the sequence is retained and reconstructed sequentially while maintain time causality.
     """)   
    st.write("""
        The encoder and the decoder can be implemented in various ways, for instance, recurrent neural networks (RNN), attention-based networks (transformer), long-short-term-memory network (LSTM), or convolutional neural networks (CNN), all of which have been proved successful in processing sequential data with sophisticated temporal correlations, and impacted by many factors. Here we show an example pytorch code for training a transformer based neural network.
    """)
    
    with st.expander("Import Packages"):    
        st.write("""
                 In this demo, the neural network is synthesized using the PyTorch framework. Please install PyTorch according to the [official guidance](https://pytorch.org/get-started/locally/) , then import PyTorch and other dependent modules.
                 """)
        st.code("""
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import json
import math
""", language="python")
        
    with st.expander("Define Network Structure"):
        st.write("""
         In this part, we define the structure of the transformer-based encoder-projector-decoder neural network. Refer to the [PyTorch document](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) for more details.
         """)
        st.code("""
class Transformer(nn.Module):
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
        
        #   Args:
        #    input_size: int, number of input variables. 1 if univariate.
        #    dec_seq_len: int, the length of the input sequence fed to the decoder
        #    max_seq_len: int, length of the longest sequence the model will receive. Used in positional encoding. 
        #    out_seq_len: int, the length of the model's output (i.e. the target sequence length)
        #    dim_val: int, aka d_model. All sub-layers in the model produce outputs of dimension dim_val
        #    n_encoder_layers: int, number of stacked encoder layers in the encoder
        #    n_decoder_layers: int, number of stacked encoder layers in the decoder
        #    n_heads: int, the number of attention heads (aka parallel attention layers)
        #    dropout_encoder: float, the dropout rate of the encoder
        #    dropout_decoder: float, the dropout rate of the decoder
        #    dropout_pos_enc: float, the dropout rate of the positional encoder
        #    dim_feedforward_encoder: int, number of neurons in the linear layer of the encoder
        #    dim_feedforward_decoder: int, number of neurons in the linear layer of the decoder
        #    dim_feedforward_projecter :int, number of neurons in the linear layer of the projecter
        #    num_var: int, number of additional input variables of the projector

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

    def forward(self, src: Tensor, tgt: Tensor, var: Tensor, device) -> Tensor:

        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src)
        enc_seq_len = 128
        
        var = var.unsqueeze(1).repeat(1,enc_seq_len,1)
        src = self.projector(torch.cat([src,var],dim=2))
        
        tgt = self.decoder_input_layer(tgt)
        tgt = self.positional_encoding_layer(tgt)
        batch_size = src.size()[0]
        tgt_mask = generate_square_subsequent_mask(sz1=self.out_seq_len, sz2=self.out_seq_len).to(device)
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
    #Generates an upper-triangular matrix of -inf, with zeros on diag.
    return torch.triu(torch.ones(sz1, sz2) * float('-inf'), diagonal=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
""", language="python")
    
    with st.expander("Load the Dataset"):   
        st.write("""
         In this part, we load and pre-process the dataset for the network training and testing. In this demo, a small dataset containing sinusoidal waveforms measured with N87 ferrite material under different frequency, temperature, and dc bias conditions is used, which can be downloaded from the [MagNet GitHub](https://github.com/PrincetonUniversity/Magnet) repository under "tutorial". 
         """)
        st.code("""
def load_dataset(data_length=128):
    # Load .json Files
    with open('/content/Dataset_sine.json','r') as load_f:
        DATA = json.load(load_f)
    B = DATA['B_Field']
    B = np.array(B)
    Freq = DATA['Frequency']
    Freq = np.log10(Freq)  # logarithm, optional
    Temp = DATA['Temperature']
    Temp = np.array(Temp)      
    Hdc = DATA['Hdc']
    Hdc = np.array(Hdc)       
    H = DATA['H_Field']
    H = np.array(H)

    # Format data into tensors
    in_B = torch.from_numpy(B).float().view(-1, data_length, 1)
    in_F = torch.from_numpy(Freq).float().view(-1, 1)
    in_T = torch.from_numpy(Temp).float().view(-1, 1)
    in_D = torch.from_numpy(Hdc).float().view(-1, 1)
    out_H = torch.from_numpy(H).float().view(-1, data_length, 1)

    # Normalize
    in_B = (in_B-torch.mean(in_B))/torch.std(in_B)
    in_F = (in_F-torch.mean(in_F))/torch.std(in_F)
    in_T = (in_T-torch.mean(in_T))/torch.std(in_T)
    in_D = (in_D-torch.mean(in_D))/torch.std(in_D)
    out_H = (out_H-torch.mean(out_H))/torch.std(out_H)

    # Save the normalization coefficients for reproducing the output sequences
    # For model deployment, all the coefficients need to be saved.
    normH = [torch.mean(out_H),torch.std(out_H)]

    # Attach the starting token and add the noise
    head = 0.1 * torch.ones(out_H.size()[0],1,out_H.size()[2])
    out_H_head = torch.cat((head,out_H), dim=1)
    out_H = out_H_head
    out_H_head = out_H_head + (torch.rand(out_H_head.size())-0.5)*0.1

    print(in_B.size())
    print(in_F.size())
    print(in_T.size())
    print(in_D.size())
    print(out_H.size())
    print(out_H_head.size())

    return torch.utils.data.TensorDataset(in_B, in_F, in_T, in_D, out_H, out_H_head), normH""", language="python")
    
    with st.expander("Training and Testing the Model"):   
        st.write("""
         In this part, we program the training and testing procedure of the network model. The loaded dataset is randomly split into training set, validation set, and test set. The output of the training part is the state dictionary file (.sd) containing all the trained parameter values, and the output of the testing part is the (.csv) file containing the predicted sequences.
         """)
        st.code("""
def main():

    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyperparameters
    NUM_EPOCH = 2000
    BATCH_SIZE = 128
    DECAY_EPOCH = 150
    DECAY_RATIO = 0.9
    LR_INI = 0.004

    # Select GPU as default device
    device = torch.device("cuda")

    # Load dataset
    dataset, normH = load_dataset()

    # Split the dataset
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    # Setup network
    net = Transformer(
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
      dim_feedforward_projecter=40).to(device)

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 

    # Train the network
    for epoch_i in range(NUM_EPOCH):

        # Train for one epoch
        epoch_train_loss = 0
        net.train()
        optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))

        for in_B, in_F, in_T, in_D, out_H, out_H_head in train_loader:
            optimizer.zero_grad()
            output = net(src=in_B.to(device), tgt=out_H_head.to(device), var=torch.cat((in_F.to(device), in_T.to(device), in_D.to(device)), dim=1), device=device)
            loss = criterion(output[:,:-1,:], out_H.to(device)[:,1:,:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.25)
            optimizer.step()
            epoch_train_loss += loss.item()

        # Compute validation
        with torch.no_grad():
            net.eval()
            epoch_valid_loss = 0
            for in_B, in_F, in_T, in_D, out_H, out_H_head in valid_loader:
                output = net(src=in_B.to(device), tgt=out_H_head.to(device), var=torch.cat((in_F.to(device), in_T.to(device), in_D.to(device)), dim=1), device=device)
                loss = criterion(output[:,:-1,:], out_H.to(device)[:,1:,:])
                epoch_valid_loss += loss.item()
        
        print(f"Epoch {epoch_i:2d} "
            f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
            f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}")
        
    # Save the model parameters
    torch.save(net.state_dict(), "/content/Model_Transformer.sd")
    print("Training finished! Model is saved!")

    # Test the network
    with torch.no_grad():
        for in_B, in_F, in_T, in_D, out_H, out_H_head in test_loader:

            # Create dummy out_H_head                            
            outputs = torch.zeros(out_H.size()).to(device)
            tgt = (torch.rand(out_H.size())*2-1).to(device)
            tgt[:,0,:] = 0.1*torch.ones(tgt[:,0,:].size())                        

            # Compute inference
            for t in range(1, out_H.size()[1]):   
                outputs = net(src=in_B.to(device),tgt=tgt.to(device),var=torch.cat((in_F.to(device), in_T.to(device), in_D.to(device)), dim=1), device=device)                     
                tgt[:,t,:] = outputs[:,t-1,:]     
            outputs = net(in_B.to(device),tgt.to(device),torch.cat((in_F.to(device), in_T.to(device), in_D.to(device)), dim=1), device=device)

            # Save results
            with open("/content/pred.csv", "a") as f:
                np.savetxt(f, (outputs[:,:-1,:]*normH[1]+normH[0]).squeeze(2).cpu().numpy())
                f.close()
            with open("/content/meas.csv", "a") as f:
                np.savetxt(f, (out_H[:,1:,:]*normH[1]+normH[0]).squeeze(2).cpu().numpy())
                f.close()
            print("Testing finished! Results are saved!")""", language="python")
            
    with st.expander("Calculate the Core Loss"):   
        st.write("""
         In this part, we provide two methods of calculating the core loss. One is naturally based on the sequence and B-H loop predicted by the neural network, and the other one is the conventional iGSE method. 
         """)
        st.code("""
def loss_BH(bdata, hdata, freq):
    # bdata: the waveform sequence of flux density that provided by the user
    # hdata: the waveform sequence of field strength that predicted by the neural network
    loss = freq * np.trapz(hdata, bdata)
    return loss

def loss_iGSE(freq, flux_list, frac_time, k_i, alpha, beta, n_interval=10_000):
    # the flux density waveform is defined by the pairs of flux_list and frac_time
    # Steinmetz coefficients k_i, alpha, and beta are needed
    period = 1 / freq
    flux_delta = np.amax(flux_list) - np.amin(flux_list)
    time, dt = np.linspace(start=0, stop=period, num=n_interval, retstep=True)
    B = np.interp(time, np.multiply(frac_time, period), flux_list)
    dBdt = np.gradient(B, dt)
    loss = freq * np.trapz(k_i * (np.abs(dBdt) ** alpha) * (flux_delta ** (beta - alpha)), time)
    return loss""", language="python")
    
    st.markdown("""---""")
