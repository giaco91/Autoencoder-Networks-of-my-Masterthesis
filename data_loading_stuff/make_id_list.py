from song_data_loader import *
import pdb
import pickle

birdhdfpath = '/Volumes/Seagate Backup Plus Drive/mnt_data_backup/mdgan_training_input_with_age_HDF/mdgan_training_input_with_age_HDF/b13r16'
birdname = 'b13r16'
age_from=0
age_to=0
age_range = [age_from,age_to]

id_list, age_list, cnt = make_IDlist_for_one_bird(birdhdfpath, birdname, age_range)

print(len(id_list))

pickle.dump( id_list, open( "id_list_"+str(age_from)+'_'+str(age_to)+".pkl", "wb" ) )

# #define the path to the (training/testing) id list of a bird (or several birds)
# training_path = '/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/song_data_loader/id_list_test.pkl'

# # define the (outer path) to data hdf files!  it will be different on your system! 
# external_file_path = '/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/new_data_2019/'

# # parameters for pytorch data loader

# # this is the minibatch size for training
# batchSize = 100
# # number of cpu workers used for loading data
# num_workers = 6 
# # shuffle the dataset on every epoch? 
# shuffle_on_epoch = True 

# # parameters for songbird_dataset object
# imageW = 6 # number of columns in spectrogram to extract from each wave file, should be = 1 for you


# # initialize the dataset object
# train_dataset = songbird_dataset(training_path, imageW, external_file_path)

# # initilize dataloader object
# train_dataloader = data.DataLoader(train_dataset, batch_size = batchSize,
#                                          shuffle=shuffle_on_epoch, num_workers=num_workers, \
#                                   drop_last = True)
