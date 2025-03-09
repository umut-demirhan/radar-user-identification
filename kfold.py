# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 01:14:04 2022

@author: udemirha
"""

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from network_models import NetGraphGeneral
from torchsummary import summary
from network_functions import train_loop, eval_loop, test_accuracy, find_target_obj_indices_in_samples
from baseline import lookup_table, linear_regression, angle_offset
import torch
from torch.utils.data import TensorDataset, DataLoader

from pprint import pprint

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

dataset_folder = r'../feb26_final_dataset'
dataset_csv = 'session.csv'
radar_label_filepath = lambda idx: f'{dataset_folder}/radar_label/label_{idx}.txt'
beam_column = 'power_beam'

dataset_csv = pd.read_csv(os.path.join(dataset_folder, dataset_csv))
rng = 25
n_beams = 64
k_folds = 10
num_samples = len(dataset_csv)
x_beam = dataset_csv.loc[:num_samples, beam_column].to_numpy().reshape(-1, 1) # Last sample doesn't have the label

y_obj_all = []
y_label = []
y_label_all = []

#%% Read Radar Label Data - includes detected (candidate) objects and the target knowledge
for i in tqdm(range(num_samples), desc='Reading label files'):
    obj_file_dir = os.path.abspath(os.path.join(dataset_folder, radar_label_filepath(i)))
    with open(obj_file_dir, 'r') as file:
        y_obj_all.append(json.load(file))
        
print('Number of objects in sample 0:', len(y_obj_all[0]))
print('Information of first object:')
pprint(y_obj_all[0][0])
# Label keys:
# points: index of the peaks grouped together (as an object)
# avg: average of the peak indices
# bounds: bounding box min-max in range-doppler-angle map
# selected: if it is the candidate target
        
#%% Process the data to collect necessary information in arrays

# Range Doppler Angle bins of selected targets
y_label_all = np.zeros((num_samples, 3)) * np.nan # Avg point information of selected object
y_obj_avg = [None] * num_samples # Average point of each object in each sample - [sample][object_idx]
y_obj_sel = [None] * num_samples # If the detected object is the target - [sample][object_idx] = True/False
y_obj_ind = np.zeros(num_samples) * np.nan  # Indices of the target object in each sample
existing_labels = np.zeros(num_samples, dtype=bool) * False # Data points with the target labeled
num_candidates = np.zeros((num_samples), dtype=int)

# Check each sample one by one
for i in tqdm(range(num_samples), desc='Processing labels'):
    target_found = False
    sample_obj_avg_list = []
    sample_obj_sel_list = []
    for j in range(len(y_obj_all[i])):
        
        sample_obj_avg_list.append(y_obj_all[i][j]['avg'])
        sample_obj_sel_list.append(y_obj_all[i][j]['selected'])
        
        if y_obj_all[i][j]['selected']: # If it is the target object
            if not target_found:
                y_label_all[i] = y_obj_all[i][j]['avg']
                target_found = True
                y_obj_ind[i] = j
            else:
                print('Label problem, multiple objects are selected!')
                
    y_obj_avg[i] = np.array(sample_obj_avg_list)
    y_obj_sel[i] = np.array(sample_obj_sel_list)*1.
    
    num_candidates[i] = len(y_obj_sel[i])
    existing_labels[i] = target_found
        
print(f'\n{int(existing_labels.sum())}/{len(existing_labels)} samples are labeled.')

#%%
# Filter out samples without labeled target object
y_label_all = y_label_all[existing_labels, :] # Avg point information of selected object
y_obj_avg = np.array(y_obj_avg, dtype=object)[existing_labels]  # Average point of each object in each sample - [sample][object_idx]
y_obj_sel = np.array(y_obj_sel, dtype=object)[existing_labels] # If the detected object is the target - [sample][object_idx] = True/False
y_obj_ind = y_obj_ind[existing_labels]  # Indices of the target object in each sample
num_candidates = num_candidates[existing_labels]
max_num_of_objects = num_candidates.max() # Maximum number of objects in all samples
x_beam = x_beam[existing_labels]

#%% Extract sequence (unique passes) information
# TODO: DeepSense scenario csv file already provides this information
# -- to be removed and pulled from the csv

## Adding sequence number to each sample in the dataset 
seq_indices = dataset_csv['image_time_idx'].diff() != 1
seq_indices.iloc[-1] = True
indices = seq_indices[seq_indices].index
for i in range(len(indices)-1):
    dataset_csv.loc[indices[i]:indices[i+1], 'sequence'] = i
##
# Adding a column to the csv indicating the target is available
dataset_csv['radar_label'] = existing_labels

#%%
# Unique Sequences in Filtered Samples
seq_idx = dataset_csv['sequence'][existing_labels].to_numpy(dtype=int)
unique_seq_idx = np.unique(seq_idx)
all_seq_idx = dataset_csv['sequence'].to_numpy(dtype=int)
print(f"Filtered samples are from {len(unique_seq_idx)}/{len(np.unique(all_seq_idx))} sequences")

#%% Histogram of input sequence length
plt.hist(seq_idx, bins=43)
plt.title('Sequence Length Distribution')
plt.xlabel('Sequence Index')
plt.ylabel('Sequence Length')

#%% Plot histogram of Candidate Objects    
plt.hist(num_candidates, bins=34, density=True)
plt.title('Histogram of Candidate Objects')
plt.xlabel('# of Candidate Objects')
plt.ylabel('% of Samples')

#%%
# y_obj_sel and _avg are list of arrays. Flatten them and keep sample number.
# Example: Let's say first 3 samples have 1, 2, 3 objects. We will get [0, 1, 1, 2, 2, 2]
object_sample_idx = np.repeat(np.arange(len(num_candidates)), num_candidates)
# Now, we can flatten the objects in different samples
y_obj_sel = np.concatenate(y_obj_sel) # Every object in every sample as a single array
y_obj_avg = np.concatenate(y_obj_avg) # Every object in every sample as a single array

object_unique_seq_idx = np.repeat(seq_idx, num_candidates)
print('Total number of candidate objects:', len(y_obj_sel))

#%% Convert the average bin index to range, angle, Doppler 
rad2deg = 180/np.pi

carrier_freq = 77e9
speed_of_light = 3e8
BW = 512 * 10.042e12 / 16.670e6 # samples_per_chirp * chirp_slope / sampling_rate
chirp_duration = 5e-6 + 38e-6 # Idle time + ramp end time
n_chirps = 250
wavelength = speed_of_light / carrier_freq
velocity_resolution = wavelength / (2 * n_chirps * chirp_duration)
range_resolution = speed_of_light / (2 * BW)
angle_fft_bins = 128
bin_to_angle = lambda x: np.arcsin(x/(angle_fft_bins/2)-1)

y_obj_avg[:, 0] = (y_obj_avg[:, 0]-n_chirps/2)*velocity_resolution
y_obj_avg[:, 1] = y_obj_avg[:, 1]*range_resolution
y_obj_avg[:, 2] = bin_to_angle(y_obj_avg[:, 2])*rad2deg

import scipy.io as scio
codebook_angles = scio.loadmat('beam_codebook.mat')['codebook_angles'].squeeze()

x_beam_idx = x_beam
x_beam_angle = codebook_angles[x_beam]


#%% Repeat remaining to match the number of samples (noting that each input-output pair is separate)
x_obj_beam_idx = np.repeat(x_beam_idx, num_candidates)
x_obj_beam_angle = np.repeat(x_beam_angle, num_candidates)
x_obj_beam_idx = x_obj_beam_idx.reshape((-1, 1)) # num_samples x features (1)
y_obj_sel = y_obj_sel.reshape((-1, 1)) # num_samples x features (1)

#%% Train-Test Samples

n_available_seqs = len(unique_seq_idx)
n_train_seqs = int(0.8 * n_available_seqs)

np.random.seed(rng)
np.random.shuffle(unique_seq_idx)
train_seq_ind = unique_seq_idx[:n_train_seqs]
test_seq_ind = unique_seq_idx[n_train_seqs:]

kf = KFold(n_splits=k_folds, shuffle=True, random_state=rng)
kfold_acc = []
for fold, (train_indices, val_indices) in enumerate(kf.split(train_seq_ind)):
    print(f"Fold {fold+1}:")
    fold_acc = []
    
    train_seq_ind = unique_seq_idx[train_indices]
    test_seq_ind = unique_seq_idx[val_indices]

    train_sample_filter = np.isin(seq_idx, train_seq_ind)
    test_sample_filter =  np.isin(seq_idx, test_seq_ind)

    train_object_filter = np.isin(object_unique_seq_idx, train_seq_ind)
    test_object_filter = np.isin(object_unique_seq_idx, test_seq_ind)

    print(f'Number of samples: {existing_labels.sum()}, Train/Test: {train_sample_filter.sum()}/{test_sample_filter.sum()}')
    print(f'Number of objects: {num_candidates.sum()}, Train/Test: {train_object_filter.sum()}/{test_object_filter.sum()}')

    #%% Data Prep
    # For Deep Learning
    X_train_beam_idx = torch.from_numpy(x_obj_beam_idx[train_object_filter]).type(torch.float32)
    X_test_beam_idx = torch.from_numpy(x_obj_beam_idx[test_object_filter]).type(torch.float32)

    X_train_beam_angle = torch.from_numpy(x_obj_beam_angle[train_object_filter]).type(torch.float32)
    X_test_beam_angle = torch.from_numpy(x_obj_beam_angle[test_object_filter]).type(torch.float32)

    X_train_obj = torch.from_numpy(y_obj_avg[train_object_filter]).type(torch.float32)
    X_test_obj = torch.from_numpy(y_obj_avg[test_object_filter]).type(torch.float32)

    y_train = torch.from_numpy(y_obj_sel[train_object_filter]).type(torch.float32)
    y_test = torch.from_numpy(y_obj_sel[test_object_filter]).type(torch.float32)

    obj_sample_idx_train = object_sample_idx[train_object_filter]
    obj_sample_idx_test = object_sample_idx[test_object_filter]

    # For Baselines
    # Move it to numpy
    X_radar_angle_train = X_train_obj[:, 2].cpu().numpy().squeeze() # Radar angle input
    X_radar_train = X_train_obj.cpu().numpy().squeeze() # 3D input for linear regression
    X_beam_idx_train = X_train_beam_idx.cpu().numpy().squeeze()

    X_radar_angle_test = X_test_obj[:, 2].cpu().numpy().squeeze()
    X_radar_test = X_test_obj.cpu().numpy().squeeze()
    X_beam_idx_test = X_test_beam_idx.cpu().numpy().squeeze()

    X_beam_angle_train =  X_train_beam_angle.cpu().numpy().squeeze()
    X_beam_angle_test =  X_test_beam_angle.cpu().numpy().squeeze()

    # Pick target samples for training (to match comm-radar angle or any feature based on target's radar/comm information)
    t_filter = y_obj_sel[train_object_filter].astype(bool).squeeze()
    target_object_radar_angle_train = X_radar_angle_train[t_filter] # We only pick target object information
    target_object_beam_idx_train = X_beam_idx_train[t_filter] # We only pick target object information
    target_object_beam_angle_train = X_beam_angle_train[t_filter] # We only pick target object information
    target_object_radar_train = X_radar_train[t_filter]

    # Take all the samples for the test (Later apply the metric (e.g. distance) to select the closest candidate object in each sample)
    target_object_radar_angle_test = X_radar_angle_test # Test requires all of the candidate objects
    target_object_beam_idx_test = X_beam_idx_test # Test requires all of the candidate objects
    target_object_beam_angle_test = X_beam_angle_test
    target_object_radar_test = X_radar_test

    target_idx = find_target_obj_indices_in_samples(y_test, obj_sample_idx_test, fun='max') # Labels

    #%%
    # Training Settings
    num_epoch = 100
    learning_rate = 1e-3
    batch_size = 128
    rng_num = rng

    # Create a folder to save the model
    results_path = os.path.abspath('results')
    model_directory = lambda idx: os.path.join(results_path, f'run_{idx}_fold_{fold}')
    run_idx = 1
    while os.path.exists(model_directory(run_idx)):
        run_idx += 1
    model_directory = model_directory(run_idx)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # For Reproducibility
    torch.manual_seed(rng_num)
    np.random.seed(rng_num)
    torch.backends.cudnn.deterministic = True

    # PyTorch GPU/CPU selection
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Neural Network
    net = NetGraphGeneral()
    net.to(device)
        
    # # Summarize the Network Model
    # summary(net, [(3,), (1,)])

    # Training Settings
    criterion = torch.nn.MSELoss() # Training Criterion
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4) # Optimizer
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1) 

    #%% Instantiate Dataset/DataLoader for pytorch
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_obj, X_train_beam_idx, y_train)
    test_dataset = TensorDataset(X_test_obj, X_test_beam_idx, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    #%% Train
    train_loss = np.zeros((num_epoch))
    val_loss = np.zeros((num_epoch))
    test_acc = np.zeros((num_epoch))
    # Epochs
    tqdm_loop = tqdm(range(num_epoch), desc="Epochs", dynamic_ncols=True)
    for epoch in tqdm_loop:

        train_loss[epoch] = train_loop(train_loader, net, optimizer, criterion, device)
        val_loss[epoch] = eval_loop(test_loader, net, criterion, device)
        test_acc[epoch] = test_accuracy(test_loader, obj_sample_idx_test, net, device)

        scheduler.step()
        
        tqdm_loop.set_description(f"[Train Loss: {train_loss[epoch]:.4f}, Val Loss: {val_loss[epoch]:.4f}, Test Acc: {test_acc[epoch]:.2f}]")

    torch.save(net.state_dict(), os.path.join(model_directory, 'model_final.pth'))

    print('Finished Training')

    #%% Evaluation
    model_path = os.path.join(model_directory, 'model_final.pth')
    #model_path = 'C:\\Users\\demir\\OneDrive\\Desktop\\TX_identification\\results\\run_84\\model_final.pth' # Or try 87-89
    test_accuracy_score = test_accuracy(test_loader, obj_sample_idx_test, net, device, model_path)
    print('Accuracy:', test_accuracy_score)
    fold_acc.append(test_accuracy_score)
    
    #%% Lookup Table
    target_idx_hat, mean_radar_angles = lookup_table(target_object_radar_angle_train,
                                                    target_object_beam_idx_train, 
                                                    target_object_radar_angle_test,
                                                    target_object_beam_idx_test,
                                                    obj_sample_idx_test,
                                                    n_beams)
    fold_acc_score = (target_idx_hat == target_idx).mean()*100
    print(f'Lookup Table: {fold_acc_score:.2f}%')
    fold_acc.append(fold_acc_score)
    #%% Linear Regression - Angle
    target_idx_hat, model_angle = linear_regression(X_train=target_object_beam_angle_train.reshape((-1, 1)),
                                                    y_train=target_object_radar_angle_train,
                                                    X_test=target_object_beam_angle_test.reshape((-1, 1)), 
                                                    y_test=target_object_radar_angle_test, 
                                                    obj_sample_idx_test=obj_sample_idx_test)

    print(f'Linear Regression (Angle): {(target_idx_hat == target_idx).mean()*100:.2f}%')
    fold_acc_score = (target_idx_hat == target_idx).mean()*100
    fold_acc.append(fold_acc_score)
    #%% Linear Regression - Doppler-Range-Angle
    target_idx_hat, model_lr_three = linear_regression(X_train=target_object_radar_train,
                                                    y_train=target_object_beam_angle_train,
                                                    X_test=target_object_radar_test,
                                                    y_test=target_object_beam_angle_test,
                                                    obj_sample_idx_test=obj_sample_idx_test)

    print(f'Linear Regression (3D): {(target_idx_hat == target_idx).mean()*100:.2f}%')
    fold_acc_score = (target_idx_hat == target_idx).mean()*100
    fold_acc.append(fold_acc_score)
    
    #%% Angle Offset
    target_idx_hat, offset = angle_offset(X_train=target_object_radar_angle_train, 
                                        y_train=target_object_beam_angle_train, 
                                        X_test=target_object_radar_angle_test, 
                                        y_test=target_object_beam_angle_test, 
                                        obj_sample_idx_test=obj_sample_idx_test)

    print(f'Angle Offset: {(target_idx_hat == target_idx).mean()*100:.2f}%')
    fold_acc_score = (target_idx_hat == target_idx).mean()*100
    fold_acc.append(fold_acc_score)
    kfold_acc.append(fold_acc)
    
#%%
kfold_acc = np.array(kfold_acc)
print('Solutions - Deep Learning | Lookup Table | Linear Regression (Angle) | Linear Regression (3D) | Angle offset')
print('K-Fold Accuracy')
print(kfold_acc.mean(axis=0))
print('K-Fold Std')
print(kfold_acc.std(axis=0))
