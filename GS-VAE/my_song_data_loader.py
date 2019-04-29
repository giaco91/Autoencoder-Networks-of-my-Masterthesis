import torch
import numpy as np
import os
import pickle
import h5py
from torch.utils import data#we use that for the DataLoader object: data.DataLoader()
import librosa.core as lc
import matplotlib.pyplot as plt
import time

def random_crop(im,width=8,idx=0,randomize=False):
    if randomize:
        ind = np.random.randint(width)
    else:
        ind=0
    return im[:,width*idx+ind:ind+width*(idx+1)]

def crop(X,snippet_length):
    L=X.shape[1]
    r=L%snippet_length
    if r != 0:
        X = np.append(X, np.zeros((X.shape[0], snippet_length-r)), axis=1)
    n_points=int(X.shape[1]/snippet_length)
    sequence=np.zeros((n_points,X.shape[0],snippet_length))
    for i in range(n_points):
        sequence[i,:,:]=X[:,i*snippet_length:(i+1)*snippet_length]
    return sequence



def down_sample(segment,rate=2):
    #downsampling along freq. axis
    shape=segment.shape
    floor=int(shape[0]/rate)
    down_sampled_segment=np.zeros((floor*(1-rate)+shape[0],shape[1]))
    for i in range(floor):
        down_sampled_segment[i,:]=np.mean(segment[rate*i:rate*(i+1),:],axis=0)
    return down_sampled_segment

def up_sample(segment,rate=2):
    #inverse of down_sample
    shape=segment.shape
    orig_data_dim=129
    floor=int(orig_data_dim/rate)
    up_sampled_segment=np.zeros((orig_data_dim,shape[1]))
    for i in range(floor):
        up_sampled_segment[rate*i:rate*(i+1),:]=segment[i,:]#broadcasting
    for j in range(orig_data_dim-floor*rate):
        up_sampled_segment[rate*floor+j,:]=segment[floor+j,:]
    return up_sampled_segment


def segment_image(im,threshold=0.025,silent_buffer=2, active_buffer=6):
    #threshold above a coloumn is calle non-silent
    #buffer, for how long a sequence must be silent to count as a silent segment
    #convention: the segment list begins with the index of the first non-silent segment
    segment_list=[]
    im_energy=np.sum(np.square(im),axis=0)
    # plt.plot(im_energy)
    # plt.show()
    count_buffer=0
    silent_state=True
    #just in case the sequence begins with non-silence we need to check that
    if im_energy[0]>threshold:
        silent_state=False
        segment_list.append(0)
    #loop over image and store the state switches
    l=im_energy.shape[0]
    for i in range(l):
        if im_energy[i]<threshold:
            if silent_state:
                count_buffer=0
            else:
                count_buffer+=1
                if count_buffer==silent_buffer:
                    segment_list.append(i-silent_buffer+2)#let the first silent cloumn be in the non-silent segment
                    count_buffer=0
                    silent_state= not silent_state
        else:
            if silent_state:
                count_buffer+=1
                if count_buffer==active_buffer:
                    segment_list.append(i-active_buffer)
                    count_buffer=0
                    silent_state= not silent_state
            else:
                count_buffer=0
    #for later convention we make sure that the last index always represents the ending of a non-silent segment        
    if not silent_state:
        segment_list.append(l)

    return segment_list


def from_polar(image):
    return image[:, :, 0]*np.cos(image[:, :, 1]) + 1j*image[:,:,0]*np.sin(image[:,:,1])


def transform(im):
        """
        This function should be used to transform data into the desired format for the network.
        inverse transoform should be an inverse of this function
        """
        im = from_polar(im)
        im, phase = lc.magphase(im)
        im = np.log1p(im)
        return im

class songbird_segmented_dataset(data.Dataset):
    def __init__(self, path2idlist, external_file_path=[],limit_n_snippets=None,
        max_seq_length=50,min_seq_length=6,data_scaling_fac=100,down_sampling=False):
        with open(path2idlist, 'rb') as f:
            # always save and load id_lists with pickle (not joblib.dump)
            self.id_list = pickle.load(f)
        self.external_file_path = external_file_path
        self.max_seq_length=max_seq_length
        self.min_seq_length=min_seq_length
        self.limit_n_snippets=limit_n_snippets
        self.data_scaling_fac=data_scaling_fac
        self.down_sampling=down_sampling
        self.n_segments,self.idx_to_id_list=self.get_n_segments()

        
    def __len__(self):
        # We need to calc the total amount of non-silent segments
        if self.limit_n_snippets is None:
            return self.n_segments
        else:
            return min(self.n_segments,self.limit_n_snippets)
    
    def __getitem__(self, index):
        # load one wav file and get a sample chunk from it
        ID = self.id_list[self.idx_to_id_list[index][0]]
        idx=self.idx_to_id_list[index][1]
        T_1=self.idx_to_id_list[index][2]
        T_2=self.idx_to_id_list[index][3]
        age_weight = ID['age']
        # this 'ID' is a dictionary containing several fields,
        # use field 'filepath' and 'within_file' to get data
        if self.external_file_path:
            birdname = os.path.basename(ID['filepath'])
            f = h5py.File(os.path.join(self.external_file_path, birdname),'r')
        else:
            f = h5py.File(ID['filepath'], 'r') 
        X = np.array(f.get(ID['within_file']))
        f.close()
        segment=transform(X[:,idx:idx+T_1])
        if self.down_sampling:
            segment=down_sample(segment)
        segment=self.data_scaling_fac*segment#need to scale the data: values too small --> squared error become too small in the low error regime!
        padded_segment=np.zeros((segment.shape[0],self.max_seq_length))#this is our padded segment
        padded_segment[:,:T_1]=segment
        padded_segment=torch.from_numpy(padded_segment).float()

        return torch.transpose(padded_segment,0,1), torch.Tensor([T_1]), torch.Tensor([T_2]), torch.Tensor([age_weight])

    def get_n_segments(self):
        #returns the amount of non-silent segments for each image summed over all images
        n_segments=0
        idx_to_id_list=[]#stores the information of which index belongs to which file and what snippet
        max_n_segments=1e10
        if self.limit_n_snippets is not None:
            max_n_segments=self.limit_n_snippets
        for i in range(len(self.id_list)):
            if n_segments<max_n_segments:
                ID = self.id_list[i]
                if self.external_file_path:
                    birdname = os.path.basename(ID['filepath'])
                    f = h5py.File(os.path.join(self.external_file_path, birdname),'r')
                else:
                    f = h5py.File(ID['filepath'], 'r') 
                X = np.array(f.get(ID['within_file']))
                f.close()
                segment_list=segment_image(X[:,:,0],active_buffer=self.min_seq_length-2) #X[:,:,0] are the amplitudes, see also from_ploar()
                n_segments_in_image=int(len(segment_list)/2)#note that len(segment_list) should always be even
                for j in range(n_segments_in_image-1):
                    k=2*j
                    T_1=segment_list[k+1]-segment_list[k]#duration of non-silent segment
                    #we only take segments that are smallter than our given max length
                    if T_1<=self.max_seq_length:
                        T_2=segment_list[k+2]-segment_list[k+1]#duration of following silent segment
                        idx_to_id_list.append((i,segment_list[k],T_1,T_2))
                        n_segments+=1
                        # plt.imshow(transform(X[:,segment_list[k]:segment_list[k]+T_1]))
                        # plt.show()

                #the last non-silent segemnt has no real following silent state, we mark it by setting to zero:
                if len(segment_list)>0:
                    k=2*(n_segments_in_image-1)
                    T_1=segment_list[k+1]-segment_list[k]
                    if T_1<=self.max_seq_length:
                        T_2=0
                        idx_to_id_list.append((i,segment_list[k],T_1,T_2))
        return n_segments,idx_to_id_list

      
class songbird_dataset(data.Dataset):
    def __init__(self, path2idlist, imageW, external_file_path=[],crop=True,
        limit_n_snippets=None,randomize_snippets=False,data_scaling_fac=100,down_sampling=False):
        with open(path2idlist, 'rb') as f:
            # always save and load id_lists with pickle (not joblib.dump)
            self.id_list = pickle.load(f)
        self.imageW = imageW
        self.crop=crop
        self.randomize_snippets=randomize_snippets
        self.external_file_path = external_file_path
        self.limit_n_snippets=limit_n_snippets
        self.data_scaling_fac=data_scaling_fac
        self.down_sampling=down_sampling
        if self.crop:
            self.n_snippets,self.idx_to_id_list=self.get_n_snippets()
        else:
            self.n_snippets=len(self.id_list)
            self.idx_to_id_list=None
        
    def __len__(self):
        # total number of samples
        if self.limit_n_snippets is None:
            return self.n_snippets
        else:
            return min(self.n_snippets,self.limit_n_snippets)
    
    def __getitem__(self, index):
        # if index%10==0:
        #     print('load idx='+str(index))
        # load one wav file and get a sample chunk from it
        #print(self.idx_to_id_list[index][0])
        ID = self.id_list[self.idx_to_id_list[index][0]]
        age_weight = ID['age']
        # this 'ID' is a dictionary containing several fields,
        # use field 'filepath' and 'within_file' to get data
        if self.external_file_path:
            birdname = os.path.basename(ID['filepath'])
            f = h5py.File(os.path.join(self.external_file_path, birdname),'r')
        else:
            f = h5py.File(ID['filepath'], 'r') 
        X = np.array(f.get(ID['within_file']))
        f.close()
        if self.crop:
            X = self.crop_and_transform(X,self.idx_to_id_list[index][1])
        else:
            X=transform(X)
        X=self.data_scaling_fac*X
        if self.down_sampling:
            X=down_sample(X)
        return torch.from_numpy(X).float(), torch.Tensor([age_weight])
    
    def crop_and_transform(self, X,idx):
        # random_crop takes out a small chunk of spectrogram of length = imageW
        # comment this line if you want to get the whole spectrogram
        if X.shape[1] < self.imageW:
            # zero pad X to required length
            X = np.append(X, np.zeros((X.shape[0], self.imageW-X.shape[1])), axis=1)
        else:
            X = random_crop(X, width=self.imageW,idx=idx,randomize=self.randomize_snippets)
        X = transform(X)
        return X

    def get_n_snippets(self):
        #returns the amount of snippets for each image summed over all images
        n_snippets=0
        idx_to_id_list=[]#stores the information of which index belongs to which file and what snippet
        max_n_snippets=1e10
        if self.limit_n_snippets is not None:
            max_n_snippets=self.limit_n_snippets
        for i in range(len(self.id_list)):
            if n_snippets<max_n_snippets:
                ID = self.id_list[i]
                if self.external_file_path:
                    birdname = os.path.basename(ID['filepath'])
                    f = h5py.File(os.path.join(self.external_file_path, birdname),'r')
                else:
                    f = h5py.File(ID['filepath'], 'r') 
                X = np.array(f.get(ID['within_file']))
                f.close()
                L=X.shape[1]
                # plt.imshow(transform(X))
                # plt.show()
                # raise ValueError('test')
                if L<self.imageW:
                    n_snippets+=1
                    idx_to_id_list.append((i,0))
                else:
                    n_snippets_in_image=int((L-self.imageW)/self.imageW)
                    n_snippets+=n_snippets_in_image
                    for j in range(n_snippets_in_image):
                        idx_to_id_list.append((i,j))
        return n_snippets,idx_to_id_list

class songbird_random_sample(object):
    def __init__(self, path2idlist, external_file_path=[]):
        with open(path2idlist, 'rb') as f:
            self.id_list = pickle.load(f)
            self.external_file_path = external_file_path
    def __len__(self):
        # total number of samples
        return len(self.id_list)
    
    def get(self, nsamps=1):
        # choose nsamp random files
        idx = np.random.randint(low = 0, high = self.__len__(), size = nsamps)
        X = [None for i in range(nsamps)]
        age_weights = [None for i in range(nsamps)]
        for (k,i) in enumerate(idx):
            ID = self.id_list[i]
            if self.external_file_path:
                birdname = os.path.basename(ID['filepath'])
                f = h5py.File(os.path.join(self.external_file_path, birdname),'r')
            else:
                f = h5py.File(ID['filepath'], 'r') 
            X[k] = np.array(f.get(ID['within_file']))
            f.close()
            
        return X, age_weights

    
class songbird_data_sample(object):
    def __init__(self, path2idlist, external_file_path):
        with open(path2idlist, 'rb') as f:
            self.id_list = pickle.load(f)
        self.external_file_path = external_file_path
    def __len__(self):
        # total number of samples
        return len(self.id_list)
    
    def __getitem__(self, index):
        # load one wav file and get a sample chunk from it
        ID = self.id_list[index]
        age_weight = ID['age_weight']
        # this 'ID' is a dictionary containing several fields,
        # use field 'filepath' and 'within_file' to get data
        if self.external_file_path:
            birdname = os.path.basename(ID['filepath'])
            f = h5py.File(os.path.join(self.external_file_path, birdname),'r')
        else:
            f = h5py.File(ID['filepath'], 'r') 
        x = np.array(f.get(ID['within_file']))
        f.close()
        return x, age_weight
    
    def get_contiguous_minibatch(self, start_idx, mbatchsize=64):
        ids = np.arange(start=start_idx, stop=start_idx+mbatchsize)
        X = [self.__getitem__(i)[0] for i in ids]
        return X

def get_sequences(path2idlist,external_file_path,snippet_length,down_sampling=True,data_scaling_fac=100,limit_n_sequences=None):
    #this function returns each image in the id_list as a sequence of snippets of legnth n
    with open(path2idlist, 'rb') as f:
        # always save and load id_lists with pickle (not joblib.dump)
        id_list = pickle.load(f)
    sequence_list=[]
    max_n_sequences=1e10
    if limit_n_sequences is not None:
        max_n_sequences=limit_n_sequences
    for i in range(len(id_list)):
        if i+1<=max_n_sequences:
            ID = id_list[i]
            birdname = os.path.basename(ID['filepath'])
            f = h5py.File(os.path.join(external_file_path, birdname),'r')
            X = np.array(f.get(ID['within_file']))
            f.close()
            X=down_sample(transform(X))*data_scaling_fac
            sequence_list.append(crop(X,snippet_length))
    return sequence_list


def get_segments(path2idlist, external_file_path=[],limit_n_sequences=None,
        max_seq_length=50,min_seq_length=6,data_scaling_fac=100,down_sampling=False):
    #this function returns each image in the id_list as a sequence of segments. It will leave out sequences that contain
    #longer semgents than the max_seq_length
    with open(path2idlist, 'rb') as f:
        # always save and load id_lists with pickle (not joblib.dump)
        id_list = pickle.load(f)
    sequence_list=[]
    T1_list=[]
    max_n_sequences=1e10
    if limit_n_sequences is not None:
        max_n_sequences=limit_n_sequences
    n_valid_seq=0
    for i in range(len(id_list)):
        #print(i)
        sequence_info=[]
        valid_sequence=True#check if valid sequence, i.e. if all segmetns are smaller or equal to max_seq_length
        #collect segmentation information
        if n_valid_seq+1<=max_n_sequences:
            ID = id_list[i]
            birdname = os.path.basename(ID['filepath'])
            f = h5py.File(os.path.join(external_file_path, birdname),'r')
            X = np.array(f.get(ID['within_file']))
            f.close()
            segment_list=segment_image(X[:,:,0],active_buffer=min_seq_length-2) #X[:,:,0] are the amplitudes, see also from_ploar()
            n_segments_in_image=int(len(segment_list)/2)#note that len(segment_list) should always be even
            for j in range(n_segments_in_image-1):
                k=2*j
                T_1=segment_list[k+1]-segment_list[k]#duration of non-silent segment
                #we only take segments that are smallter than our given max length
                if T_1<=max_seq_length:
                    T_2=segment_list[k+2]-segment_list[k+1]#duration of following silent segment
                    sequence_info.append((segment_list[k],T_1,T_2))
                else:
                    valid_sequence=False
                    #print('not valid: '+str(T_1))
            #the last non-silent segemnt has no real following silent state, we mark it by setting to zero:
            if len(segment_list)>0:
                k=2*(n_segments_in_image-1)
                T_1=segment_list[k+1]-segment_list[k]
                if T_1<=max_seq_length:
                    T_2=0
                    sequence_info.append((segment_list[k],T_1,T_2))
                else:
                    valid_sequence=False
                    #print('not valid: '+str(T_1))
            #print('n_points='+str(n_segments_in_image))
            #make the actual segmentation
            if valid_sequence:
                #print('this is a valid sequence')
                n_valid_seq+=1
                T1_in_sequence=[]
                n_points=len(sequence_info)
                if down_sampling:
                    image=down_sample(transform(X))
                # plt.imshow(image)
                # plt.show()
                sequence=np.zeros((n_points,image.shape[0],max_seq_length))
                for n in range(n_points):
                    sequence[n,:,:sequence_info[n][1]]=image[:,sequence_info[n][0]:sequence_info[n][0]+sequence_info[n][1]]*data_scaling_fac
                    T1_in_sequence.append(sequence_info[n][1])
                sequence_list.append(torch.from_numpy(sequence).float())
                T1_list.append(torch.from_numpy(np.asarray(T1_in_sequence)).int())
        else:
            return sequence_list,T1_list
    return sequence_list,T1_list

def make_IDlist_for_one_bird(birdhdfpath, birdname, age_range = [0,100]):
    birdfile = h5py.File(birdhdfpath, 'r')
    all_grps = list(birdfile.items())
    id_list = []
    age_list = []
    cnt = 0
    # cycle over days for this bird
    for g in all_grps:
        day = birdfile.get(g[0])
        day_wav_list = list(day.items())
        # cycle over files
        for f in day_wav_list:
            # for adults age fixed to 100, for juveniles it ranges from 0 to the number of recording days
            age = float(f[0].split('_')[-1])
            if age_range[0] <= age <= age_range[1]:
                age_list.append(age)
                id_list.append({'id':cnt, 'birdname': birdname, 'filepath': birdhdfpath, \
                            'within_file': '/'+g[0]+'/'+f[0], 'age': age})
            cnt += 1
    birdfile.close()
    return id_list, age_list, cnt

