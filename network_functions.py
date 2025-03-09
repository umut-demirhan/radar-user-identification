# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:24:30 2021

@author: Umt
"""

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

def train_loop(train_data, net, optimizer, criterion, device):
    net.train()
    
    running_loss = 0.0
    total_samples = 0
    for x_obj, x_beam, y_sel in train_data:
        batch_size = x_obj.shape[0]
        total_samples += batch_size
        
        x_obj, x_beam, y_sel = x_obj.to(device), x_beam.to(device), y_sel.to(device)
        
        optimizer.zero_grad() # Make the gradients zero
        y_sel_hat = net(x_obj, x_beam) # Prediction
        loss = criterion(y_sel_hat, y_sel) # Loss computation
        loss.backward() # Backward step
        optimizer.step() # Update coefficients
        
        running_loss += loss.item() * batch_size
            
    curr_loss = running_loss / total_samples

    return curr_loss
    
def eval_loop(val_data, net, criterion, device, batch_size=64):
    net.eval()
    
    running_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x_obj, x_beam, y_sel in val_data:
            batch_size = x_obj.shape[0]
            total_samples += batch_size
            
            x_obj, x_beam, y_sel = x_obj.to(device), x_beam.to(device), y_sel.to(device)
            
            y_sel_hat = net(x_obj, x_beam) # Prediction
            loss = criterion(y_sel_hat, y_sel) # Loss computation
            
            running_loss += loss.item() * batch_size
            
            
    curr_loss = running_loss / total_samples
    
    return curr_loss
    
def test_accuracy(test_data, obj_sample_idx, net, device, model_path=None):
    # Network Setup
    if model_path is not None:
        net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    
    y_sel_hat_all = []
    y_sel_all = []
    total_samples = 0
    with torch.no_grad():
        for x_obj, x_beam, y_sel in test_data:
            batch_size = x_obj.shape[0]
            total_samples += batch_size
            
            x_obj, x_beam, y_sel = x_obj.to(device), x_beam.to(device), y_sel.to(device)
            
            y_sel_hat = net(x_obj, x_beam) # Prediction
            
            y_sel_all.append(y_sel.cpu())
            y_sel_hat_all.append(y_sel_hat.cpu())
    
    y_sel_hat_all = np.concatenate(y_sel_hat_all)
    y_sel_all = np.concatenate(y_sel_all)
    
    tar_idx_hat = find_target_obj_indices_in_samples(y_sel_hat_all, obj_sample_idx, 'max')
    tar_idx = find_target_obj_indices_in_samples(y_sel_all, obj_sample_idx, 'max')
    
    accuracy = np.mean(tar_idx_hat == tar_idx)*100
    
    return accuracy
     
def find_target_obj_indices_in_samples(samples, obj_sample_idx, fun='min'):
    """
    Finds the index of the maximum value in each group, with indices
    counting from 0 within each group.

    Args:
        samples: A list or NumPy array of sample values.
        group_indices: A list or NumPy array of corresponding group indices.

    Returns:
        A dictionary where keys are group indices and values are the indices
        of the maximum samples within each group (starting from 0).
    """
    samples = np.array(samples).squeeze()
    obj_sample_idx = np.array(obj_sample_idx)

    df = pd.DataFrame({'samples': samples, 'obj_sample_idx': obj_sample_idx})
    df['object_index_in_sample'] = df.groupby('obj_sample_idx').cumcount()  # Create group-specific indices
    if fun =='max':
        max_indices_df = df.loc[df.groupby('obj_sample_idx')['samples'].idxmax()]
    elif fun == 'min':
        max_indices_df = df.loc[df.groupby('obj_sample_idx')['samples'].idxmin()]
    else:
        raise NotImplementedError
    return max_indices_df['object_index_in_sample'].to_numpy()

