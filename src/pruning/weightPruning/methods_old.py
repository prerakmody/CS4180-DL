import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pruning.weightPruning.utils import prune_rate, arg_nonzero_min


def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''    
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = p.data.abs() > threshold
            masks.append(pruned_inds.float())
    return masks

def quick_filter_prune(model, pruning_perc):
    '''
    Prune pruning_perc% filters globally
    '''
    # Step0 - Return Value
    masks = []
    values = [] # holds the L2 norms of all filters

    # Step1 - Loop over all modules
    for p in model.parameters():

        # [TODO] p.name == 'MaskedCOnv2D'
        # Step2 - Pick a MaskedCOnv2D
        if len(p.data.size()) == 4: # nasty way of selecting conv layer #[prev_filter,curr_filter,H,W]
            p_np = p.data.cpu().numpy()

            masks.append(np.ones(p_np.shape).astype('float32'))                           

            # Step3 - Find L2 norm 
            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.sum(np.square(p_np), (1,2,3)) /(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer/np.sqrt(np.square(value_this_layer).sum())
            
            # Step ?? - Not used
            # min_value, min_ind = arg_nonzero_min(list(value_this_layer))           

            # Step 4 - Appending L2 norms            
            max_value        = np.max(value_this_layer)
            value_this_layer /= max_value
            values           = np.concatenate((values, value_this_layer))

    # Step5 - Find the threshold
    threshold = np.percentile(values, pruning_perc)

    # Step6 - Prune on the basis of threshold
    ind = 0
    for idx_param, p in enumerate(model.parameters()):

        if len(p.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            max_value = np.max(value_this_layer)

            value_this_layer /= max_value

            # masks[ind].shape = [B,3,3,3]
            masks[ind][value_this_layer < threshold] = 0.
            ind += 1
            
    masks = [torch.from_numpy(mask) for mask in masks]
    return masks

# ------------------------------------------------------------- #

def prune_one_filter(model, masks):
    '''
    Pruning one least ``important'' feature map by the scaled l2norm of 
    kernel weights
    arXiv:1611.06440
    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for p in model.parameters():

        if len(p.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"

    values = np.array(values) #.shape = [num_layers, 2]

    # set mask corresponding to the filter to prune
    to_prune_layer_ind = np.argmin(values[:, 0])
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.

    print('Prune filter #{} in layer #{}'.format(
        to_prune_filter_ind, 
        to_prune_layer_ind))

    return masks

def filter_prune(model, pruning_perc):
    '''
    Prune filters one by one until reach pruning_perc
    (not iterative pruning)
    '''
    print (' -- THIS IS THE OLDER FUNCTION')
    masks = []
    current_pruning_perc = 0.

    while current_pruning_perc < pruning_perc:
        masks = prune_one_filter(model, masks)
        model.set_masks(masks)
        current_pruning_perc = prune_rate(model, verbose=False)
        print('{:.2f} pruned'.format(current_pruning_perc))

    return masks
