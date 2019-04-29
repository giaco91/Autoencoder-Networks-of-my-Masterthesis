import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import numpy as np
import math
import matplotlib.pyplot as plt

class CONCRETE(nn.Module):
    def __init__(self,data_dim=129,n=6,n_states=40,temp=0.5):
        super(CONCRETE, self).__init__()
        self.data_dim=data_dim
        self.n=n
        self.n_states=n_states
        self.temp=temp
        #--- map to the relaxed discrete latent space---
        self.fc_enc1 = nn.Linear(data_dim*n, 1000)
        self.fc_enc2 = nn.Linear(1000, 1000)
        self.fc_enc3 = nn.Linear(1000,300)
        self.fc_enc4 = nn.Linear(300, n_states)

        #---------------------------------

        #---map to the image--------
        self.fc_dec1 = nn.Linear(n_states,1000)
        self.fc_dec2 = nn.Linear(1000,1000)
        self.fc_dec3 = nn.Linear(1000,n*data_dim)
        #-----------------------
        
    def encode(self, x):
        h1 = F.relu(self.fc_enc1(x))
        h2 = F.relu(self.fc_enc2(h1))
        h3 = F.relu(self.fc_enc3(h2))
        return self.fc_enc4(h3)

    def reparameterize(self, log_alpha):
        eps = torch.rand_like(log_alpha)
        gumbel_noise=-torch.log(-torch.log(eps+1e-20)+1e-20)
        zeta=(gumbel_noise+log_alpha)/self.temp
        return F.softmax(zeta,dim=1)

    def decoder(self,y):
        h_1=F.relu(self.fc_dec1(y))
        h_2=F.relu(self.fc_dec2(h_1))
        return self.fc_dec3(h_2)

    def forward(self, x,training=True):
        log_alpha = self.encode(x.contiguous().view(-1, self.data_dim*self.n))
        alpha=torch.exp(log_alpha)
        y = self.reparameterize(log_alpha)
        x = self.decoder(y)
        #In evaluation mode we chose the maximum likelyhood one-hot encoding
        if not training:
            max_idx=torch.argmax(alpha,dim=1)
            # print(max_idx)
            y=torch.zeros_like(alpha)
            for i in range(len(max_idx)):
                y[i,max_idx[i]]=1
            x=torch.relu(self.decoder(y))
            return x,log_alpha, alpha

        return x,log_alpha, alpha

    def to_discrete(self,x):
    	#this function takes a sequence of image snippets x and makes a ML discrete encoding
    	#x should have size: (sequence_length,data_dim,n)
        alpha=torch.exp(self.encode(x.contiguous().view(-1, self.data_dim*self.n)))
        ml_idx=torch.argmax(alpha,dim=1)
        return ml_idx

    def to_image(self,z,representations=None):
    	#this function takes a sequence of integers and decodes it to image snippets
    	#z should be an array of integers, like the ouput of the function to_discrete()
    	with torch.no_grad():
            seq_length=len(z)
            if representations is None:
                y=torch.zeros(seq_length,self.n_states)
                for i in range(seq_length):
                    y[i,z[i]]=1
                x=self.decoder(y)
                x=x.view(seq_length,self.n,self.data_dim)
                x=torch.transpose(x,1,2)
            else:
                x=torch.zeros(seq_length,self.data_dim,self.n)
                for i in range(seq_length):
                    x[i,:,:]=torch.from_numpy(representations[z[i],:,:])             
            return x

