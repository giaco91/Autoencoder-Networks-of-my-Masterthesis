import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/song_data_loader')
from my_song_data_loader import *

from concrete import *


parser = argparse.ArgumentParser(description='concrete encoder')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--n_states', type=int, default=50, metavar='N',
                    help='how many (discrete) latent states should the model have?')
parser.add_argument('--temp', type=float, default=1, metavar='N',
                    help='how much temperature do you want to give your concrete distribution?')

#----some settings----
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


#-----additional hyper parameters and constants-----
load_model=False
make_samples=True
train=True
plot_every=1
save_every=1000
verbose=True
lr=1e-3
n=6#size of snippet
name_id_list='id_list_2_2.pkl'

data_dim=129

#-----read in data-----
path_to_id_list = '/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/song_data_loader/'+name_id_list
with open(path_to_id_list, 'rb') as f:
    id_list = pickle.load(f)
    print('number of songs loaded: '+str(len(id_list)))
external_file_path = '/Volumes/Sandro_driv'
batchSize = 20
limit_n_snippets=40#the limiting amount of snippets to load in to the training set. must be larger or equal than batchSize
num_workers = 0  
shuffle_on_epoch = False
data_scaling_fac=100
down_sampling=True
if down_sampling:
    data_dim=65
imageW = n # number of columns in spectrogram to extract from each wave file, should be = 1 for you
train_dataset = songbird_dataset(path_to_id_list, imageW, external_file_path,crop=True,
    limit_n_snippets=limit_n_snippets,randomize_snippets=False,
    data_scaling_fac=data_scaling_fac,down_sampling=down_sampling)
train_dataloader = data.DataLoader(train_dataset, batch_size = batchSize,
                                         shuffle=shuffle_on_epoch, num_workers=num_workers, \
                                  drop_last=True)

print('training on number of snippets: '+str(batchSize*len(train_dataloader)))

#----initialize model------
model = CONCRETE(data_dim=data_dim,n=n,n_states=args.n_states,temp=args.temp).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
state_epoch=0
if load_model==True:
    print('reload model....')
    state_dict=torch.load('concrete_models/concrete_h='+str(args.n_states)+'_down_sampling='+str(down_sampling)+'.pkl')
    state_epoch=state_dict['epoch']
    model.load_state_dict(state_dict['model_state'])
    optimizer.load_state_dict(state_dict['optimizer_state'])


def loss_function(x_hat, x,alpha):
    x_hat=x_hat.view(x_hat.size()[0],n,data_dim)

    #---squared distance
    dx=x_hat-x
    pixel_loss=torch.sum(torch.mul(dx,dx))

    #--regularize alpha, since they tend to become ridiculously large---
    da=torch.sum(torch.mul(alpha,alpha))
    l_a=0.01
    REG = l_a*da
    return pixel_loss + REG

def train(epoch):
    model.train()
    train_loss = 0
    for (i,data) in enumerate(train_dataloader):
        data_batch=data[0]
        data_batch=torch.transpose(data_batch,1,2)
        optimizer.zero_grad()
        x,log_alpha,alpha= model(data_batch,training=True)#for finetuning also set l_a=0!
        loss = loss_function(x, data_batch,alpha)
        loss.backward()
        # print(len(list(model.parameters())))
        # print(list(model.parameters())[0].grad)
        # clip_grad_norm: I think it could help for very unlikely samples from concrete in the low loss regime
        #to prevent crazy large losses --> large gradiends, which scrambles up the parameters quite a lot.
        nn.utils.clip_grad_norm_(model.parameters(), 1)       
        Loss=loss.item()/(batchSize*data_dim)
        if verbose:
            print('loss='+str(Loss))
        train_loss += Loss
        optimizer.step()
    epoch_loss=train_loss/len(train_dataloader)
    print('epoch: '+str(epoch+state_epoch)+' loss: '+str(epoch_loss))
    return epoch_loss


def test():
    model.eval()
    with torch.no_grad():
        for (i,data) in enumerate(train_dataloader):
            data_batch=data[0]
            data_batch=torch.transpose(data_batch,1,2)
            x_hat,log_alpha,alpha= model(data_batch,training=True)
            x_hathat,log_alpha,alpha= model(data_batch,training=False)
            if verbose:
                loss = loss_function(x_hathat, data_batch,alpha)
                print('test loss='+str(loss.item()/(batchSize*data_dim)))
            x_hat=x_hat.view(x_hat.size()[0],n,data_dim)
            x_hat=torch.relu(torch.transpose(x_hat,1,2)/data_scaling_fac)
            x_hathat=x_hathat.view(x_hat.size()[0],n,data_dim)
            x_hathat=torch.relu(torch.transpose(x_hathat,1,2)/data_scaling_fac)
            data_batch=torch.transpose(data_batch,1,2)/data_scaling_fac
            return x_hat,x_hathat,data_batch

if not os.path.exists('concrete_samples/') and make_samples:
    os.mkdir('concrete_samples/')
if not os.path.exists('concrete_models/'):
    os.mkdir('concrete_models/')


running_average_loss=0
for epoch in range(1, args.epochs + 1):
    if train:
        epoch_loss=train(epoch)
    running_average_loss+=epoch_loss
    if epoch%save_every==0:
        print('store model....')
        torch.save({'epoch': args.epochs+state_epoch, 'model_state': model.state_dict(),'optimizer_state': optimizer.state_dict()}, 'concrete_models/concrete_h='+str(args.n_states)+'_down_sampling='+str(down_sampling)+'.pkl')
    if epoch%plot_every==0 and make_samples:
        print('average_loss: '+str(running_average_loss/plot_every))
        print('store sampling...')
        running_average_loss=0
        k,r,s=test()
        b=np.random.randint(0, k.size()[0]-5)
        fig = plt.figure()
        fig.suptitle("Original vs. Reconstruction, Temperature="+str(args.temp), fontsize=16)
        ax=plt.subplot(131)
        ax.set_title('original')
        ax.set_xlabel('time')
        ax.set_ylabel('frequency')
        ax.axis('off')
        ax.imshow(torch.cat((s[b,:,:],s[b+1,:,:],s[b+2,:,:],s[b+3,:,:],s[b+4,:,:]),1),origin='lower',cmap='gray') 
        ax=plt.subplot(132)
        ax.set_title("relaxed arg_max")
        ax.axis('off')
        # ax.set_xlabel('time')
        # ax.set_ylabel('frequency')
        ax.imshow(torch.cat((k[b,:,:],k[b+1,:,:],k[b+2,:,:],k[b+3,:,:],k[b+4,:,:]),1),origin='lower',cmap='gray')
        ax=plt.subplot(133)
        ax.set_title('arg_max')
        ax.axis('off')
        # ax.set_xlabel('time')
        # ax.set_ylabel('frequency')
        ax.imshow(torch.cat((r[b,:,:],r[b+1,:,:],r[b+2,:,:],r[b+3,:,:],r[b+4,:,:]),1),origin='lower',cmap='gray') 
                   
        plt.subplots_adjust(hspace=-1, wspace=0.3)
        plt.savefig('concrete_samples/e='+str(epoch+state_epoch) + '_h='+ str(args.n_states)+'.png', format='png', dpi=1000)


print('store model....')
torch.save({'epoch': args.epochs+state_epoch, 'model_state': model.state_dict(),'optimizer_state': optimizer.state_dict()}, 'concrete_models/concrete_h='+str(args.n_states)+'_down_sampling='+str(down_sampling)+'.pkl')


#----this is how do use the trained encoder and the decoder-------
# state_sequence=[]
# for sequence in sequence_list:
#     states=model.to_discrete(sequence)
#     state_sequence.append(states.numpy())
# #print(np.histogram(state_sequence[0],bins=np.linspace(0,39,40)))
# sequence=model.to_image(torch.from_numpy(state_sequence[2]))
# plt.imshow(torch.cat((sequence[3,:,:],sequence[4,:,:],sequence[6,:,:],sequence[7,:,:]),1),origin='lower')
# plt.show()





