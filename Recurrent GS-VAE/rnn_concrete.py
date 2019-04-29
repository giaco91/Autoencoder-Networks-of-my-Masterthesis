import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import numpy as np
import math
import matplotlib.pyplot as plt


class RNN_CONCRETE(nn.Module):
    def __init__(self,n_hidden=100,n_layers=1,n_states=100,temp=0.5,bidirectional=True,
        det_latent_dim=5,non_linear_dynamics=False,data_dim=129):
        super(RNN_CONCRETE, self).__init__()
        self.n_hidden=n_hidden#number of hidden states
        self.n_states=n_states
        self.n_layers=n_layers
        self.bidirectional=bidirectional
        self.n_direction=1
        self.det_latent_dim=det_latent_dim
        self.non_linear_dynamics=non_linear_dynamics
        self.data_dim=data_dim
        self.temp=temp
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
        self.m_enc3 = nn.Linear(500, n_states)

        if self.non_linear_dynamics:

            #---map to the z-space
            self.z_enc1 = nn.Linear(n_states,500)
            self.z_enc2 = nn.Linear(500,500)
            self.z_enc3 = nn.Linear(500,det_latent_dim)

            #---ODE blackbox
            self.ode1 = nn.Linear(det_latent_dim,500)
            self.ode2 = nn.Linear(500,500)
            self.ode3 = nn.Linear(500,det_latent_dim)

        else:

            #---map to the (A,dz,b)-space
            self.Azb_enc1 = nn.Linear(n_states,1000)
            self.Azb_enc2 = nn.Linear(1000,1000)
            self.Azb_enc3A = nn.Linear(1000,det_latent_dim**2)
            self.Azb_enc3z = nn.Linear(1000,det_latent_dim)
            self.Azb_enc3b = nn.Linear(1000,det_latent_dim)

        #---map to the image space
        self.dec1 = nn.Linear(det_latent_dim,500)
        self.dec2 = nn.Linear(500,1000)
        self.dec3 = nn.Linear(1000,data_dim)

    def encoder_m(self,h):
        h1=F.relu(self.m_enc1(h))
        h2=F.relu(self.m_enc2(h1))
        return self.m_enc3(h2)

    def reparameterize(self, log_alpha):
        eps = torch.rand_like(log_alpha)
        gumbel_noise=-torch.log(-torch.log(eps+1e-20)+1e-20)
        zeta=(gumbel_noise+log_alpha)/self.temp
        return F.softmax(zeta,dim=1)

    def encoder_z(self,m):
        h1 = F.relu(self.z_enc1(m))
        h2 = F.relu(self.z_enc2(h1))
        return self.z_enc3(h2)

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
            x_seq[:,:,i]=self.dec3(h2)
        return x_seq



    def forward(self, x,seq_lengths):
        max_length=seq_lengths[0]
        #----rnn encoder
        _,h = self.rnn(x)#will be initialized with zero
        h=h.view(self.n_layers, self.n_direction, -1, self.n_hidden)
        h=torch.transpose(h,0,2)
        h=h[:,:,0,:].contiguous().view(-1,self.n_direction*self.n_hidden)#take only the first hidden state of the last multilayer rnn states
        
        #---map to discrete latent space
        log_alpha = self.encoder_m(h)
        alpha=torch.exp(log_alpha)
        y = self.reparameterize(log_alpha)

        #---dynamical part
        if self.non_linear_dynamics:
            z_1=self.encoder_z(y)
            z_seq = self.ode_solve(z_1,max_length)
            dz=z_1

        else:
            A,dz,b = self.encoder_Azb(y)
            #---restrict to guaranteed stable system:
            A=torch.tanh(A)/self.det_latent_dim
            z_seq = self.linear_dynamic(A,dz,b,max_length)

        #---decoder
        x_seq = self.decoder(z_seq,max_length)
        x_hat = torch.transpose(x_seq,1,2)

        #---undo the masking
        x_padded, _= nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_hat.masked_fill_(x_padded.eq(0), 0)

        return x_hat, alpha, dz, x_padded

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