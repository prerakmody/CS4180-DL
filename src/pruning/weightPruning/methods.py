import pdb
import numpy as np
import matplotlib.pyplot as plt

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

def quick_filter_prune_v2(model, pruning_perc, min_conv_id=-1, max_conv_id=-1, verbose=0):
    '''
    Prune pruning_perc% filters globally
    '''
    # Step0 - Return Value
    masks  = []
    values_l2norm_tomas   = [] # holds the L2 norms of all filters
    values_l2norm         = []
    values_l2norm_notouch = []

    # Step1 - Loop over all modules
    for module_obj in model.named_parameters():
        
        # Step1 - Pick module name
        conv_id = -1
        if min_conv_id > 0 and max_conv_id > 0:
            # if len(module_name) >= 5:
            if ('conv' in module_obj[0]) and ('weight' in module_obj[0]):
                module_name = module_obj[0].split('.')[2]  
                conv_id = int(module_name[4:])
        
        # Step2 - Pick Up Module weights
        if len(module_obj[1].data.size()) == 4: # nasty way of selecting conv layer #[prev_filter,curr_filter,H,W]
            module_weights       = module_obj[1].data.cpu().numpy()
            module_total_weights = module_weights.shape[1]*module_weights.shape[2]*module_weights.shape[3]
            masks.append(np.ones(module_weights.shape).astype('float32'))                           

            # Step3 - Find scaled L2-norm for each filter of this layer
            value_this_layer = np.sum(np.square(module_weights), (1,2,3)) / module_total_weights
            value_this_layer = value_this_layer/np.sqrt(np.square(value_this_layer).sum()) # normalization (important)
            
            if verbose:
                print ('  -- [DEBUG] ', module_name)
                print ('  -- [DEBUG][scaled L2-norm] min: ', np.min(value_this_layer), ' || max : ', np.max(value_this_layer))
                plt.plot(value_this_layer)
                plt.title(module_name)
                plt.show()

            # Step 4 - Appending L2 norms
            if conv_id > -1 and min_conv_id > 0 and max_conv_id > 0:
                if conv_id < min_conv_id or conv_id > max_conv_id:
                    continue
                else:
                    values_l2norm           = np.concatenate((values_l2norm, value_this_layer))
                    values_l2norm_tomas     = np.concatenate((values_l2norm_tomas, value_this_layer / np.max(value_this_layer)))
            else:
                values_l2norm           = np.concatenate((values_l2norm, value_this_layer))
                values_l2norm_tomas     = np.concatenate((values_l2norm_tomas, value_this_layer / np.max(value_this_layer)))

    # Step5 - Find the threshold
    threshold_l2norm        = np.percentile(values_l2norm                         , pruning_perc)
    threshold_l2norm_tomas  = np.percentile(values_l2norm_tomas                   , pruning_perc)
    threshold_l2norm_prerak = np.percentile(values_l2norm / np.max(values_l2norm) , pruning_perc)

    if verbose:
        f,axarr = plt.subplots(1,3, figsize=(15,5), sharey=True);
        axarr[0].hist(values_l2norm);
        axarr[0].set_title('Scaled L2 norms - %.6f' % (threshold_l2norm));
        axarr[0].plot([threshold_l2norm,threshold_l2norm],[0,6000]);
        
        axarr[1].hist(values_l2norm_tomas);
        axarr[1].set_title('[Tomas] Scaled L2 norms - %.6f' % (threshold_l2norm_tomas));
        axarr[1].plot([threshold_l2norm_tomas,threshold_l2norm_tomas],[0,6000]);
        
        axarr[2].hist(values_l2norm / np.max(values_l2norm));
        axarr[2].set_title('[Prerak] Scaled L2 norms - %.6f' % (threshold_l2norm_prerak));
        axarr[2].plot([threshold_l2norm_prerak,threshold_l2norm_prerak],[0,6000]);
        
        plt.show();

    # Step6 - Prune on the basis of threshold
    ind = 0
    for module_obj in model.named_parameters():
        
        # Step2 - Pick Up Module weights
        if len(module_obj[1].data.size()) == 4: # nasty way of selecting conv layer #[prev_filter,curr_filter,H,W]

            # Step1 - Find the name
            if min_conv_id > 0 and max_conv_id > 0:
                # if len(module_name) >= 5:
                if ('conv' in module_obj[0]) and ('weight' in module_obj[0]):
                    module_name = module_obj[0].split('.')[2]  
                    conv_id = int(module_name[4:])
                    if conv_id < min_conv_id or conv_id > max_conv_id:
                        ind += 1
                        continue


            module_weights       = module_obj[1].data.cpu().numpy()
            module_total_weights = module_weights.shape[1]*module_weights.shape[2]*module_weights.shape[3]
            
            # Step3 - Find scaled L2-norm for each filter of this layer
            value_this_layer = np.sum(np.square(module_weights), (1,2,3)) / module_total_weights
            value_this_layer = value_this_layer/np.sqrt(np.square(value_this_layer).sum()) # normalization (important)

            # Step4 - Tomas's addition
            # value_this_layer /= np.max(value_this_layer)

            # Step5 - Apply mask
            # masks[ind].shape = [B,3,3,3]
            masks[ind][value_this_layer < threshold_l2norm] = 0.
            ind += 1
            
    masks = [torch.from_numpy(mask) for mask in masks]
    return masks

def quick_filter_prune_v1(model, pruning_perc, verbose=0):
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
