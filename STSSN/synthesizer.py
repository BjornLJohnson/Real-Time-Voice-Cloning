import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define componenets of Synthesizer network
class Synthesizer(nn.Module):
    def __init__(self, embedding_size=768, prenet_dim=256, prenet_dropout=0.5, lstm_size=1024, mel_out=80, 
                 attention_size=128, location_feature_size=32):
        super(Synthesizer, self).__init__()
        
        self.prenet = PreNet(input_dim=mel_out, prenet_dim=prenet_dim, dropout=prenet_dropout)
        
        self.lstm = nn.LSTM(input_size=prenet_dim+embedding_size, hidden_size=lstm_size, num_layers=2, batch_first=True)
        
        self.attention = LocationSensitiveAttention(attention_size, lstm_size, embedding_size, location_feature_size)
        
        self.linear_proj = nn.Linear(in_features=lstm_size, out_features=mel_out)
        
        self.stop_token = nn.Linear(in_features=lstm_size, out_features=1)
        
        self.postnet = PostNet(in_channels=1, out_channels=256, kernel=(1,5),
                               stride=1, padding=(0,2), dropout=0.1)

    def forward(self, x):
        # Input is a batch of feature matrices from encoder network concatenated w/ style embeddings
        # Input is of shape (batch, "timesteps", features=768)
        
        # Initialize all tensors which depend on previous timestep to 0
        batch_size=x.size()[0]
        step_input = x.new_zeros(batch_size, 80)
        step_attention = x.new_zeros(batch_size, 768)
        lstm_state = (x.new_zeros(2*batch_size, 1, 1024), x.new_zeros(2*x.shape[0], 1, 1024))
        cumulative_attention_weight = x.new_zeros(batch_size, x.size()[1])
        
        # Forward
        outputs, stop_tokens, attention_weights = [], [], []
        self.attention.reset()
        
        for i in range(x.size()[1]):
            # Run previous frame through prenet
            step_input = self.prenet(step_input)
            
            print("step input size")
            print(step_input.shape)
            print("attention size")
            print(step_attention.shape)
            
            # Concatenate prenet output with attention, run through lstm
            lstm_out, lstm_state = self.lstm(
                torch.cat((step_input.unsqueeze(1), step_attention.unsqueeze(1)), dim=2), lstm_state)
                                               
            decoder_state = lstm_out.transpose(0,1).squeeze(1)

            print('decoder_state', decoder_state.shape)
            print('step_attention', step_attention.shape)
            
            # Concatenate lstm output with attention, project down to spectrogram output shape & scalar
            linear_input = torch.cat((decoder_state, step_attention), dim=1)
            output = self.linear_proj(linear_input)
            stop_token = self.stop_token(linear_input)
            
            # Calculate new attention based on total encoder output, single lstm step output, cumulative weight
            step_attention, attention_weight = self.attention(decoder_state, x, cumulative_attention_weight)
            
            # Update cumulative attention
            cumulative_attention_weight = cumulative_attention_weight + attention_weight
            
            # Save important values for later
            outputs+=output
            stop_tokens+=stop_token
            attention_weights+=attention_weight
            
            # Autoregressive, each step depends on previous output
            step_input = output
        
        
        outputs = torch.stack(feat_output, dim=1)
        stop_tokens = torch.stack(stop_tokens, dim=1).squeeze()
        attention_weights = torch.stack(attention_weights, dim=1)
        
        return self.postnet(outputs)

class PreNet(nn.Module):
    def __init__(self, input_dim=80, prenet_dim=256, dropout=0.5):
        super(PreNet, self).__init__()

        self.linear1 = nn.Linear(input_dim, prenet_dim)
        self.linear2 = nn.Linear(prenet_dim, prenet_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return x
    
class PostNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dropout):
        super(PostNet, self).__init__()

        # Consider changing to Conv1D layers
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding)
        self.cnn3 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding)
        self.cnn4 = nn.Conv2d(out_channels, 1, kernel, stride, padding)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.cnn1(x)
        x = self.dropout1(F.tanh(x))
        x = self.cnn2(x)
        x = self.dropout2(F.tanh(x))
        x = self.cnn3(x)
        x = self.dropout3(F.tanh(x))
        x = self.cnn4(x)

        x += residual
        return x # (batch, channel, feature, time)

class LocationSensitiveAttention(nn.Module):
    def __init__(self, attention_dim=128, lstm_size=1024,
                 embedding_size=768, location_feature_dim=32):
        super(LocationSensitiveAttention, self).__init__()
        
        self.W = nn.Linear(lstm_size, attention_dim, bias=True) # keep one bias
        self.V = nn.Linear(embedding_size, attention_dim, bias=False)
        self.U = nn.Linear(location_feature_dim, attention_dim, bias=False)
        self.F = nn.Conv1d(in_channels=1, out_channels=location_feature_dim,
                           kernel_size=31, stride=1, padding=(31-1)//2,
                           bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
        
        # Might not need
        self.reset()

    def reset(self):
        """Remember to reset at decoder step 0"""
        self.Vh = None # pre-compute V*h_j due to it is independent from the decoding step i

    def score(self, decoder_state, encoding, cumulative_attention_weights):
        """Calculate energy:
           e_ij = score(s_i, ca_i-1, h_j) = v tanh(W s_i + V h_j + U f_ij + b)
           where f_i = F * ca_i-1,
                 ca_i-1 = sum_{j=1}^{T-1} a_i-1
        Args:
            decoder_state: [batch, lstm_size]
            encoding: [batch, timesteps, embedding_size]
        Returns:
            energies: [batch, timesteps]
        """
        # print('decoder_state', decoder_state.size())
        # print('encoding', encoding.size())
        decoder_state = decoder_state.unsqueeze(1) #[batch, 1, lstm_size], insert time-axis for broadcasting
        Ws = self.W(decoder_state) #[N, 1, A]
        if self.Vh is None:
            self.Vh = self.V(encoding) #[N, Ti, A]
        location_feature = self.F(cumulative_attention_weights.unsqueeze(1)) #[N, 32, Ti]
        # print(location_feature.size())
        Uf = self.U(location_feature.transpose(1, 2)) #[N, Ti, A]
        energies = self.v(torch.tanh(Ws + self.Vh + Uf)).squeeze(-1) #[N, Ti]
        # print('W s_i', Ws.size())
        # print('V h_j', self.Vh.size())
        # print('U f_ij', Uf.size())
        # print('energies', energies)
        return energies
    
# called as:
# step_attention, attention_weight = self.attention(lstm_out, x, cumulative_attention_weight)
    def forward(self, decoder_state, encoding, cumulative_attention_weights):
        """
        Args:
            decoder_state: [batch, lstm_out]
            encoding: [batch, timesteps, embedding_size]
            cumulative_attention_weights: [?,?]
        Returns:
            attention_context: [batch, embedding_size]
            attention_weights: [batch, timesteps]
        """
        print('decoder_state', decoder_state.shape)
        print('encoding', encoding.shape)
        print('cumulative attention', cumulative_attention_weights.shape)
        
        energies = self.score(decoder_state, encoding, cumulative_attention_weights) #[batch, timesteps]
        attention_weights = F.softmax(energies, dim=1) #[batch, timesteps]
        
        print('weights', attention_weights.shape)
        print('encoding', encoding.shape)
        
        #[batch, 1, timesteps] bmm [batch, timesteps, embedding_size] -> [batch, 1, embedding_size]
        attention_context = torch.bmm(attention_weights.unsqueeze(1), encoding)
        
#         print("attention_context", attention_context.shape)
        attention_context = attention_context.squeeze(1) # [N, Ti]
    
        print('context', attention_context.shape)
        return attention_context, attention_weights