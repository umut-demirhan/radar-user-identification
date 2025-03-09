import numpy as np
from network_functions import find_target_obj_indices_in_samples
from sklearn.linear_model import LinearRegression

def lookup_table(target_object_radar_angle_train,
                 target_object_beam_idx_train, 
                 target_object_radar_angle_test,
                 target_object_beam_idx_test,
                 obj_sample_idx_test,
                 n_beams):
    #%% Baseline Solution - Lookup Table - CLosest object to the average angle of each given beam
    # Find mean radar angle for each beam
    mean_radar_angles = np.zeros(n_beams) - 1000
    for i in range(n_beams):
        beam_radar_targets = target_object_beam_idx_train.astype(int)==i
        if beam_radar_targets.sum() > 0: 
        # If there is an object in the data with comm beam i
            mean_radar_angles[i] = target_object_radar_angle_train[beam_radar_targets].mean()

    obj_comm_to_beam_radar_angle_dist = np.abs(target_object_radar_angle_test.flatten() - mean_radar_angles[target_object_beam_idx_test.astype(int)].flatten())
    # Run argmin in each sample using the sample indices of the objects 
    target_idx_hat = find_target_obj_indices_in_samples(obj_comm_to_beam_radar_angle_dist, obj_sample_idx_test, fun='min')
    
    return target_idx_hat, mean_radar_angles

def linear_regression(X_train, y_train, X_test, y_test, obj_sample_idx_test):
    #%% Linear Regression
    model = LinearRegression()
    model.fit(X_train,
              y_train.reshape(-1, 1), 
              )

    y_test_hat = model.predict(X_test)
    dist_lr = np.abs(y_test_hat.flatten() - y_test.flatten())
    # Run argmin in each sample using the sample indices of the objects 
    target_idx_hat = find_target_obj_indices_in_samples(dist_lr, obj_sample_idx_test, fun='min')
    
    return target_idx_hat, model

def angle_offset(X_train, y_train, X_test, y_test, obj_sample_idx_test):
    #%% Angle Offset
    offset = (y_train - X_train).mean()
    y_test_hat = X_test + offset
    dist_lr = np.abs(y_test.flatten() - y_test_hat.flatten())
    # Run argmin in each sample using the sample indices of the objects 
    target_idx_hat = find_target_obj_indices_in_samples(dist_lr, obj_sample_idx_test, fun='min')
    return target_idx_hat, offset