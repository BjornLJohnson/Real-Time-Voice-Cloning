import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define all components of encoder network
class ContentEncoder(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_feats=80, stride=1, dropout=0.1):
        super(ContentEncoder, self).__init__()
        
        # Single convolutional layer for basic heirarchal feature extraction
        # Conv2D(in_channels, out_channels, kernel_size, stride, padding, dilation....)
        self.Convolutional_Feature_Extraction = nn.Conv2d(1, 32, 3, stride=1, padding=1)

        # n_cnn Residual Convolutional Layers for deeper feature extraction
        self.Residual_CNN_Blocks = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=80)
            for _ in range(n_cnn_layers)
        ])
                             
        # Single fully connected layer, 2560 inputs (80 features * 32 filters), rnn_dim outputs
        # Somewhat misleading, I believe this outputs a batch of matrices
        self.Feature_Downsampling = nn.Linear(80*32, rnn_dim)
        
        self.Recurrent_Block = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])

    def forward(self, x):
        # Input is a mel spectrogram
        # Input is of shape (batch, channels=1, mel_features=80, timesteps)
        x = self.Convolutional_Feature_Extraction(x)
        x = self.Residual_CNN_Blocks(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.Feature_Downsampling(x)
        x = self.Recurrent_Block(x)
        
        return x


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)   
        x = self.dropout(x)
        return x