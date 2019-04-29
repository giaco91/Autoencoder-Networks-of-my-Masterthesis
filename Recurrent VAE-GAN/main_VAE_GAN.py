from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision.utils as tv
import matplotlib.pyplot as plt
import numpy as np
import math
import inspect
import time

from my_song_data_loader import *

from rnn_vs import *
from discriminator import *

parser = argparse.ArgumentParser(description='Recurrent VAE-GAN')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--latent_dim', type=int, default=15, metavar='N',
                    help='What is the dimension of the latent space?')
parser.add_argument('--det_latent_dim', type=int, default=5, metavar='N',
                    help='What is the dimension of the deterministic/dynamic latent space?')
parser.add_argument('--n_hidden_enc', type=int, default=100, metavar='N',
                    help='What is the dimension of the RNN hidden state of the encoder?')
parser.add_argument('--n_hidden_discr', type=int, default=100, metavar='N',
                    help='What is the dimension of the RNN hidden state of the discriminator?')
parser.add_argument('--n_rnn_layers', type=int, default=1, metavar='N',
                    help='How many RNNs to stack between two observations?')
parser.add_argument('--non_linear_dynamics', default=False, metavar='N',
                    help='do you want a gloabl non-linear dynamic system or a specific linear dynamic system?')
parser.add_argument('--burn_epochs', type=int, default=1, metavar='N',
                    help='how many epochs should we pretrain the encoder before we bring in the discriminator?')
parser.add_argument('--discr_reg', type=int, default=10000, metavar='N',
                    help='how much should we weight the discriminator regularization?')



#----some settings----
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#-----Hyper parameters and constants-----

load_model=False
make_samples=True
train=True
plot_every=10
save_every=1000
verbose=False
lr_enc=1e-4
lr_dec=1e-5
data_dim=129
name_id_list='id_list_2_2.pkl'

#----- load data ----------
path_to_id_list = '/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/song_data_loader/'+name_id_list
with open(path_to_id_list, 'rb') as f:
        id_list = pickle.load(f)
        print('number of songs loaded: '+str(len(id_list)))
external_file_path = '/Volumes/Sandro_driv'
batchSize = 10
limit_n_snippets = 10#maximum amount of snippets we want to train on (must be larger or equal than batchSize)
min_seq_length=10 #all non-silent sequences that are smaller than min_seq_length will be cut out as silent segments
max_seq_length=60 #all non-silent sequences that are larger than max_seq_length will be ignored from data
data_scaling_fac=100 #since the original data images in log(x+1)-scale are too small for least squares, scaling helps!
down_sampling=True #If true, the data gets down sampled by factor of 2 over the frequency axis -> dimensionality reduction by factor 2
num_workers = 6  
shuffle_on_epoch = False
if down_sampling:
    data_dim=65
train_dataset = songbird_segmented_dataset(path_to_id_list, external_file_path,
    limit_n_snippets=limit_n_snippets,max_seq_length=max_seq_length,
    min_seq_length=min_seq_length,data_scaling_fac=data_scaling_fac,down_sampling=down_sampling)

train_dataloader = data.DataLoader(train_dataset, batch_size = batchSize,
                                         shuffle=shuffle_on_epoch, num_workers=num_workers, \
                                  drop_last = True)
print('training on number of snippets: '+str(batchSize*len(train_dataloader)))


def pack_sequences(data):
    #the data from DataLoader must be preprocessed in order to use nn.utils.rnn.pack_padded_sequence
    data_batch=data[0]#(batchsize,max_seq_length,data_dim)
    T_1=torch.squeeze(data[1])
    sorted_T_1, indices=T_1.sort(descending=True)
    data_batch=data_batch[indices,:int(sorted_T_1[0]),:]#sort and cut at batch specific maximum seq_length
    #print(data_batch.size())
    seq_lengths=list(sorted_T_1.numpy().astype(int))#cast to list
    #print(seq_lengths)
    packed_data_batch = nn.utils.rnn.pack_padded_sequence(data_batch, seq_lengths, batch_first=True)
    return packed_data_batch, seq_lengths

def unpack_sequences(data_packed):
    #the inverse of pack sequences
    data, seq_lengths= nn.utils.rnn.pad_packed_sequence(data_packed, batch_first=True)
    return data,seq_lengths

# def segment_batch_to_snippet_batch(segment_batch):
#     return snippet_batch

#----initialize model------
def init_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.xavier_normal_(model.weight_hh_l0)
        nn.init.xavier_normal_(model.weight_ih_l0)



model = RNN(n_hidden=args.n_hidden_enc,n_layers=args.n_rnn_layers,
    latent_dim=args.latent_dim,det_latent_dim=args.det_latent_dim,
    bidirectional=True,non_linear_dynamics=args.non_linear_dynamics,data_dim=data_dim).to(device)
optimizer_enc = optim.Adam(model.parameters(), lr=lr_enc)
discriminator = DISCRIMINATOR(data_dim=data_dim, n_hidden=args.n_hidden_discr,n=min_seq_length).to(device)
optimizer_discr = optim.Adam(discriminator.parameters(), lr=lr_dec)
state_epoch=0
if load_model==True:
    print('reload model....')
    state_dict_enc=torch.load('drnn_vs_models/enc_n_hidden='+str(args.n_hidden_enc)+'.pkl')
    state_dict_discr=torch.load('drnn_vs_models/discr_n_hidden='+str(args.n_hidden_discr)+'.pkl')
    state_epoch=state_dict_enc['epoch']+0
    model.load_state_dict(state_dict_enc['model_state'])
    discriminator.load_state_dict(state_dict_discr['model_state'])
    optimizer_enc.load_state_dict(state_dict_enc['optimizer_state'])
    optimizer_discr.load_state_dict(state_dict_discr['optimizer_state'])
else:
    model.apply(init_weights)
    discriminator.apply(init_weights)


def encoder_loss(x_hat, x,z_1,d_hat,mu,logvar,epoch):
    #shape of x: (batchSize,max_seq_length_of_that_batch,data_dim)
    #---mean_squared_error
    dx=x_hat-x
    pixel_loss=torch.sum(torch.mul(dx,dx))/batchSize
    print('L2 loss='+str(pixel_loss/data_dim))

    #--regulalrization
    REG=0
    #---regularization for m-space towards zero
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())#negative, since we need a loss...
    REG+=KLD
    #---dz dowards zero
    lz=0.01
    REG+=torch.sum(torch.mul(z_1,z_1))/batchSize
    #print(torch.mean(torch.sum(torch.mul(z_1,z_1),dim=1)))
    #---discriminator
    if epoch>args.burn_epochs:
        print('*** adversarial regularization active ***')
        ld=args.discr_reg
        REG+=-ld*torch.sum(torch.log(d_hat+1e-10))/batchSize
    return pixel_loss+REG

def discriminator_loss(d,d_hat):
    discr_loss=-(torch.sum(torch.log(d+1e-10)+torch.log(1-d_hat+1e-10)))/batchSize
    #print('detect reals with prob.='+str(torch.mean(d)))
    print('detect fakes with prob.='+str(torch.mean(1-d_hat)))
    return discr_loss


def encoder_mode():
    for p in discriminator.parameters():
        p.requires_grad=False
    for p in model.parameters():
        p.requires_grad=True

def discriminator_mode():
    for p in discriminator.parameters():
        p.requires_grad=True
    for p in model.parameters():
        p.requires_grad=False

def train(epoch):
    model.train()
    train_loss_enc = 0
    train_loss_discr = 0
    #start_time = time.time()
    for (i,data) in enumerate(train_dataloader):
        data_batch,seq_lengths=pack_sequences(data)
        optimizer_enc.zero_grad()
        encoder_mode()
        x_hat,m,z_1,data_batch_padded,mu,logvar = model(data_batch,seq_lengths)
        x_hat_packed=nn.utils.rnn.pack_padded_sequence(x_hat, seq_lengths, batch_first=True)
        d_hat=discriminator(x_hat_packed,seq_lengths)
        loss_enc = encoder_loss(x_hat, data_batch_padded,z_1,d_hat,mu,logvar,epoch)
        loss_enc.backward()
        Loss_enc=loss_enc.item()/data_dim
        train_loss_enc += Loss_enc
        optimizer_enc.step()

        #---discriminator step
        optimizer_discr.zero_grad()
        discriminator_mode()
        d = discriminator(data_batch,seq_lengths)
        x_hat_packed=nn.utils.rnn.pack_padded_sequence(x_hat.detach(), seq_lengths, batch_first=True)
        d_hat = discriminator(x_hat_packed,seq_lengths)
        loss_discr =discriminator_loss(d,d_hat)
        loss_discr.backward()
        optimizer_discr.step()
        Loss_discr=loss_discr.item()

        train_loss_discr+=Loss_discr
        
        if verbose:
            print('encoder batch loss='+str(Loss_enc))
            print('discriminator batch loss='+str(Loss_discr))

    #print("--- %s seconds ---" % (time.time() - start_time))
    epoch_loss_enc=train_loss_enc / len(train_dataloader)
    epoch_loss_discr=train_loss_discr / len(train_dataloader)
    print('-------epoch: '+str(epoch+state_epoch)+' encoder loss: '+str(epoch_loss_enc)+'-------')
    print('-------epoch: '+str(epoch+state_epoch)+' discriminator loss: '+str(epoch_loss_discr)+'-------')

def batch_upsampling(batch):
    #batch must have shape (batchSize,data_dim,seq_length)
    size=batch.size()
    s1=up_sample(batch[0,:,:].numpy())
    up_sampled_batch=torch.zeros(size[0],s1.shape[0],size[2])
    up_sampled_batch[0,:,:]=torch.from_numpy(s1)
    for i in range(1,batch.size()[0]):
        up_sampled_batch[i,:,:]=torch.from_numpy(up_sample(batch[i,:,:].numpy()))
    return up_sampled_batch


def test():
    print('test...')
    model.eval()
    with torch.no_grad():
        for (i,data) in enumerate(train_dataloader):      
            data_batch,seq_lengths=pack_sequences(data)
            x_hat,m,_,data_batch,mu,logvar,z_seq= model(data_batch,seq_lengths,sample=True,return_z_seq=True)
            image_reconstructed=torch.relu(torch.transpose(x_hat,1,2))/data_scaling_fac
            image_original=torch.transpose(data_batch,1,2)/data_scaling_fac
            if down_sampling:
                image_reconstructed=batch_upsampling(image_reconstructed)
                image_original=batch_upsampling(image_original)
            return image_reconstructed,image_original,seq_lengths,m,mu,z_seq

if not os.path.exists('drnn_vs_samples/') and make_samples:
    os.mkdir('drnn_vs_samples/')
if not os.path.exists('drnn_vs_models/'):
    os.mkdir('drnn_vs_models/')



for epoch in range(1, args.epochs + 1):
    if train:
        train(epoch)
    if epoch%save_every==0:
        print('store model...')
        torch.save({'epoch': args.epochs+state_epoch, 'model_state': model.state_dict(),'optimizer_state': optimizer_enc.state_dict()}, 'drnn_vs_models/enc_n_hidden='+str(args.n_hidden_enc)+'.pkl')
        torch.save({'epoch': args.epochs+state_epoch, 'model_state': discriminator.state_dict(),'optimizer_state': optimizer_discr.state_dict()}, 'drnn_vs_models/discr_n_hidden='+str(args.n_hidden_discr)+'.pkl')
    if epoch%plot_every==0 and make_samples:
        print('store sample...')
        r,o,seq_lengths,_,_,_=test()
        b=np.random.randint(0, batchSize-6)
        #b=0
        fig = plt.figure()
        ax=plt.subplot(131)
        ax.set_title('original')
        ax.set_xlabel('time [4 ms]')
        ax.set_ylabel('frequency [64 Hz]')
        o=torch.cat((o[b,:,:seq_lengths[b]],o[b+1,:,:seq_lengths[b+1]],o[b+2,:,:seq_lengths[b+2]],o[b+3,:,:seq_lengths[b+3]],o[b+4,:,:seq_lengths[b+4]],o[b+5,:,:seq_lengths[b+5]],o[b+6,:,:seq_lengths[b+6]]),1).numpy()
        #o=o[7,:,:seq_lengths[7]].numpy()
        plt.imshow(np.log(o/np.max(o)+0.1),origin='lower',cmap='gray')
        ax=plt.subplot(133)
        ax.set_title('reconstructed')
        ax.set_xlabel('time [4 ms]')
        ax.set_ylabel('frequency [64 Hz]')
        r=torch.cat((r[b,:,:seq_lengths[b+0]],r[b+1,:,:seq_lengths[b+1]],r[b+2,:,:seq_lengths[b+2]],r[b+3,:,:seq_lengths[b+3]],r[b+4,:,:seq_lengths[b+4]],r[b+5,:,:seq_lengths[b+5]],r[b+6,:,:seq_lengths[b+6]]),1).numpy()
        #r=torch.cat((r[b,:,:],r[b+1,:,:],r[b+2,:,:],r[b+3,:,:],r[b+4,:,:]),1).numpy()
        #r=r[7,:,:].numpy()
        plt.imshow(np.log(r/np.max(r)+0.1),origin='lower',cmap='gray')
        plt.subplots_adjust(hspace=-1, wspace=0)
        plt.savefig('drnn_vs_samples/'+str(epoch+state_epoch) + '_'+ str(args.latent_dim)+ '.png', format='png', dpi=1000)
        #b+=7







