import math
import torch
import torch.nn as nn
from torch.nn import Linear,LayerNorm, \
    TransformerEncoder, TransformerEncoderLayer, \
    TransformerDecoder, TransformerDecoderLayer, \
    Transformer
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor
import numpy as np

from omniisaacgymenvs.model.attention import AttentionPooling

class TransformerEnc(nn.Module):
    def __init__(self,
                 input_dim: int = 128,
                 num_heads: int = 8,
                 attention_num_heads: int = 8,
                 encoder_hidden_dim: int = 64,
                 attention_hidden_dim: int = 64,
                 dim_feedforward: int = 512,  # encoder, decoder, etc 별로 세분화 하여 나뉘어 질 수도 있음
                 num_points: int = 200,
                 output_feature: int = 64,
                 num_layers: int = 6,
                 ):
        super(TransformerEnc, self).__init__()

        self.num_points = num_points
        self.encoder_hidden_dim = encoder_hidden_dim
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

        # self.encoder_input_embedding = Linear(input_dim, encoder_hidden_dim)
        # encoder_layers = TransformerEncoderLayer(d_model=input_dim, nhead=num_heads,
        #                                          dim_feedforward=dim_feedforward, batch_first=True)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers,
        #                                               norm=nn.LayerNorm(encoder_hidden_dim))
        # self.output_projection = Linear(encoder_hidden_dim, output_feature)
        # self.feature_projection = Linear(encoder_hidden_dim, 5)
        ''' feature를 전달받는 거라 feature projection은 필요 없음.
          그래서 d_model에 encoder_hidden_dim 대신 input_dim을 넣음'''
        encoder_layers = TransformerEncoderLayer(d_model=input_dim, nhead=num_heads,
                                                 dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers=num_layers,
                                                      norm=nn.LayerNorm(input_dim))
        
        # self.attention_pooling = AttentionPooling(embed_dim=input_dim,
        #                                           num_heads=attention_num_heads,
        #                                           latent_dim=attention_hidden_dim)

        # self.output_projection = Linear(input_dim, output_feature)
        
        # self.feature_projection = Linear(encoder_hidden_dim, 5)
        
        # projection from hidden dimension to 1 dimension for regression. output has a only one feature

        # self.linear_mapping = Linear(in_features=decoder_hidden_dim,
        #                              out_features=1)

    def forward(self, src: Tensor):
        """
        The length of trg must be equal to the length of the actual target sequence
        https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e#:~:text=The%20length%20of%20trg%20must%20be%20equal%20to%20the%20length%20of%20the%20actual%20target%20sequence

        Returns a tensor of shape:
        [target_sequence_length, batch_size, prediction_length]

        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input,
                 (S, N, E) if batch_first=False or (N, S, E) if
                 batch_first=True, where S is the source sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)
            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input,
                 (T, N, E) if batch_first=False or (N, T, E) if
                 batch_first=True, where T is the target sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)
            src_mask: the mask for the src sequence to prevent the models from
                      using data points from the target sequence
            tgt_mask: the mask for the tgt sequence to prevent the models from
                      using data points from the target sequence
        """
        # src = torch.flatten(src, start_dim=1) # src shape: [batch_size, src length, dim_val]
        # tgt = torch.flatten(tgt, start_dim=1) # tgt shape: [batch_size, target seq len, dim_val]
        # src_embedding = self.encoder_input_embedding(src)  # src shape: [batch_size, src length, dim_val] applied to each time step
        
        ''' feature를 전달받는 거라 feature projection은 필요 없음'''
        encoder_output = self.transformer_encoder(src)  # src shape: [batch_size, enc_seq_len, dim_val]

        encoder_output = self.dropout(encoder_output)
        # pooling_output = self.attention_pooling(encoder_output)
        # output = self.output_projection(pooling_output)

        # return output
        return encoder_output