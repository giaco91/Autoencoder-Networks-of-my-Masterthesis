import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import numpy as np
import math
import matplotlib.pyplot as plt

class DISCRIMINATOR(nn.Module):
    def __init__(self,data_dim=129,n=6,n_hidden=100,n_layers=1):
        super(DISCRIMINATOR, self).__init__()
        self.data_dim=data_dim
        self.n_hidden=n_hidden
        self.n=n
        self.n_layers=n_layers
        self.n_direction=2

        #---rnn recognition net
        self.rnn = nn.GRU(input_size=data_dim,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=True)

        #---discriminator
        #nn.utils.spectral_norm(nn.Linear(n_hidden*self.n_direction,500))
        self.discr1 = nn.Linear(n_hidden*self.n_direction,500)
        self.discr2 = nn.Linear(500,300)
        self.discr3 = nn.Linear(300,1)
        
    def discriminator(self,h):
        h1=F.relu(self.discr1(h))
        h2=F.relu(self.discr2(h1))
        h3=self.discr3(h2)
        return F.sigmoid(h3)

    def forward(self, x,seq_lengths):
        max_length=seq_lengths[0]
        
        #----rnn encoder
        _,h = self.rnn(x)#will be initialized with zero
        h=h.view(self.n_layers, self.n_direction, -1, self.n_hidden)
        h=torch.transpose(h,0,2)
        h=h[:,:,0,:].contiguous().view(-1,self.n_direction*self.n_hidden)#take only the first hidden state of the last multilayer rnn states

        d = self.discriminator(h)

        return d



