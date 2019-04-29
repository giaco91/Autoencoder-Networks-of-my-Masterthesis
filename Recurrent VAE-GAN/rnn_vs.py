import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import numpy as np
import math
import matplotlib.pyplot as plt


class RNN(nn.Module):
    def __init__(self,n_hidden=100,n_layers=1,latent_dim=15,bidirectional=False,
        det_latent_dim=5,non_linear_dynamics=False,data_dim=129):
        super(RNN, self).__init__()
        self.n_hidden=n_hidden#number of hidden states
        self.latent_dim=latent_dim
        self.n_layers=n_layers
        self.bidirectional=bidirectional
        self.n_direction=1
        self.det_latent_dim=det_latent_dim
        self.non_linear_dynamics=non_linear_dynamics
        self.data_dim=data_dim
        if bidirectional:
            self.n_direction=2

        #---rnn
        self.rnn = nn.GRU(input_size=data_dim,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        #---map to m-space
        self.m_enc1 = nn.Linear(n_hidden*self.n_direction, 1000)
        self.m_enc2 = nn.Linear(1000, 500) 
        self.m_enc31 = nn.Linear(500, latent_dim)
        self.m_enc32 = nn.Linear(500, latent_dim)

        if self.non_linear_dynamics:

            #---map to the z-space
            self.z_enc1 = nn.Linear(latent_dim,500)
            self.z_enc2 = nn.Linear(500,500)
            self.z_enc3 = nn.Linear(500,det_latent_dim)

            #---ODE blackbox
            self.ode1 = nn.Linear(det_latent_dim,500)
            self.ode2 = nn.Linear(500,500)
            self.ode3 = nn.Linear(500,det_latent_dim)

        else:

            #---map to the (A,dz,b)-space
            self.Azb_enc1 = nn.Linear(latent_dim,1000)
            self.Azb_enc2 = nn.Linear(1000,1000)
            self.Azb_enc3A = nn.Linear(1000,det_latent_dim**2)
            self.Azb_enc3z = nn.Linear(1000,det_latent_dim)
            self.Azb_enc3b = nn.Linear(1000,det_latent_dim)

        #---map to the image space
        self.dec1 = nn.Linear(det_latent_dim,500)
        self.dec2 = nn.Linear(500,1000)
        self.dec3 = nn.Linear(1000,1000)
        self.dec4 = nn.Linear(1000,data_dim)

    def encoder_m(self,h):
        h1=F.relu(self.m_enc1(h))
        h2=F.relu(self.m_enc2(h1))
        return self.m_enc31(h2), self.m_enc32(h2)

    def encoder_z(self,m):
        h1 = F.relu(self.z_enc1(m))
        h2 = F.relu(self.z_enc2(h1))
        return self.z_enc3(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        #std = torch.ones(logvar.size())*0.01
        #print(torch.mean(std))

        eps = torch.randn_like(std)#returns random numbers from the normal distribution, equal to: eps = torch.normal(torch.zeros_like(std),1)
        return eps.mul(std).add_(mu)

    def encoder_Azb(self,m):
        h_1=F.relu(self.Azb_enc1(m))
        h_2=F.relu(self.Azb_enc2(h_1))
        return self.Azb_enc3A(h_2),self.Azb_enc3z(h_2),self.Azb_enc3b(h_2) 

    def ld_step(self,A,z):
        z=torch.unsqueeze(z,1)
        z_next=torch.squeeze(torch.matmul(z,A))
        return z_next

    def linear_dynamic(self,A,dz,b,max_length):
        z_sequence=torch.zeros((dz.size()[0],self.det_latent_dim,max_length))
        z_sequence[:,:,0]=dz
        A=A.view(-1,self.det_latent_dim,self.det_latent_dim)
        for i in range(1,max_length):
            z_sequence[:,:,i]=self.ld_step(A,z_sequence[:,:,i-1].clone())
        z_sequence+=torch.unsqueeze(b,2)
        return z_sequence

    def ode_step(self,z):
        h1 = F.relu(self.ode1(z))
        h2 = F.relu(self.ode2(h1))
        return z+F.sigmoid(self.ode3(h2))

    def ode_solve(self,z_1,max_length):
        z_sequence=torch.zeros((z_1.size()[0],self.det_latent_dim,max_length))
        z_sequence[:,:,0]=z_1
        for i in range(1,max_length):          
            z_sequence[:,:,i]=self.ode_step(z_sequence[:,:,i-1].clone())
        return z_sequence

    def decoder(self,z_sequence,max_length):
        x_seq=torch.zeros((z_sequence.size()[0],self.data_dim,max_length))
        for i in range(0,max_length):
            h1 = F.relu(self.dec1(z_sequence[:,:,i]))
            h2 = F.relu(self.dec2(h1))
            h3 = F.relu(self.dec3(h2))
            x_seq[:,:,i]=self.dec4(h3)
        return x_seq



    def forward(self, x,seq_lengths,sample=False,return_z_seq=False):
        max_length=seq_lengths[0]
        #----rnn encoder
        _,h = self.rnn(x)#will be initialized with zero
        h=h.view(self.n_layers, self.n_direction, -1, self.n_hidden)
        h=torch.transpose(h,0,2)
        h=h[:,:,0,:].contiguous().view(-1,self.n_direction*self.n_hidden)#take only the first hidden state of the last multilayer rnn states
        
        #---map to latent space
        mu, logvar = self.encoder_m(h)

        if sample:
            print('*** sampling mode ***')
            m=mu
        else:
            m = self.reparameterize(mu, logvar)           

        #---dynamical part
        if self.non_linear_dynamics:
            z_1=self.encoder_z(m)
            z_seq = self.ode_solve(z_1,max_length)
            dz=z_1

        else:
            A,dz,b = self.encoder_Azb(m)
            #---restrict to guaranteed stable system:
            A=torch.tanh(A)/self.det_latent_dim
            z_seq = self.linear_dynamic(A,dz,b,max_length)
            #z_1=dz

        #---decoder
        x_seq = self.decoder(z_seq,max_length)
        x_hat = torch.transpose(x_seq,1,2)

        #---undo the masking
        x_padded, _= nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_hat.masked_fill_(x_padded.eq(0), 0)

        if return_z_seq:
            return x_hat, m, dz, x_padded, mu, logvar, z_seq

        return x_hat, m, dz, x_padded, mu, logvar

    def encode(self,x):
        #----rnn encoder
        _,h = self.rnn(x)#will be initialized with zero
        h=h.view(self.n_layers, self.n_direction, -1, self.n_hidden)
        h=torch.transpose(h,0,2)
        h=h[:,:,0,:].contiguous().view(-1,self.n_direction*self.n_hidden)#take only the first hidden state of the last multilayer rnn states
        
        #---map to latent space
        m = self.encoder_m(h)

        return m

    def decode(self,m,max_length):
        #---dynamical part
        if self.non_linear_dynamics:
            z_1=self.encoder_z(m)
            z_seq = self.ode_solve(z_1,max_length)
            dz=z_1

        else:
            A,dz,b = self.encoder_Azb(m)
            #---restrict to guaranteed stable system:
            A=torch.tanh(A)/self.det_latent_dim
            z_seq = self.linear_dynamic(A,dz,b,max_length)

        #---decoder
        x_seq = self.decoder(z_seq,max_length)
        x_hat = torch.transpose(x_seq,1,2)

        return x_hat


