import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import *

from unet import *
from diffusion import ConditionalDiffusionTrainingNetwork


class LatentDiffusion(nn.Module):
    def __init__(
      self,
      n_features,
      batch_size,
      window_size,
      out_dim,
      time_steps,
      noise_steps,
      denoise_steps,
      dim,
      init_dim,
      dim_mults,
      channels, 
      groups,
      gru_n_layers,
      n_layers,
      schedule, 
      gru_hid_dim,
      kernel_size,
      feat_gat_embed_dim,
      time_gat_embed_dim,
      use_gatv2,   
      alpha
      ):
        super().__init__()
        self.time_steps=time_steps
        self.encoder = EncoderBlock(gru_hid_dim, n_features, gru_n_layers, window_size,alpha)
        self.decoder = RNNDecoder(window_size,gru_hid_dim, n_layers,out_dim,dropout=0.2)
        self.conditionaldiffusion = ConditionalDiffusionTrainingNetwork(n_features,window_size,batch_size,time_steps,schedule,noise_steps,denoise_steps,dim,init_dim,dim_mults,channels,groups=groups) 
                                                                      
    def forward(self, x, anomaly_scores):
        t = torch.randint(0, self.time_steps, (x.shape[0],), device=x.device).long()
        x = self.encoder(x)
        loss,x=self.conditionaldiffusion(x,anomaly_scores=anomaly_scores)
        x=self.decoder(x)
        return loss,x
        
        
