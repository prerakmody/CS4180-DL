import pdb
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import sampler


def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

    
def train(model, loss_fn, optimizer, param, loader_train, loader_val=None):

    model.train()
    for epoch in range(param['num_epochs']):
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

        for t, (x, y) in enumerate(loader_train):
            x_var, y_var = to_var(x), to_var(y.long())

            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            if (t + 1) % 100 == 0:
                print('t = %d, loss = %.8f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
         

def test(model, loader):

    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)
    for x, y in loader:
        x_var = to_var(x, volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples

    print('Test accuracy: {:.2f}% ({}/{})'.format(
        100.*acc,
        num_correct,
        num_samples,
        ))
    
    return acc
    

def prune_rate(model, method="weight", verbose=True):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0

    total_filters = 0
    total_zero_filters = 0

    layer_id = 0

    # Along with %, show ratio, also what method is being used
    #if verbose:
    #    print("Pruning with method: {}-pruning".format(method))

    for parameter in model.parameters():

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            total_filters += parameter.shape[0]
            layer_id += 1
            zero_param_this_layer = np.count_nonzero(parameter.cpu().data.numpy()==0)
            nb_zero_param += zero_param_this_layer

            zero_filters = parameter.shape[0] - np.count_nonzero(np.sum(parameter.cpu().data.numpy(), (1,2,3)))
            total_zero_filters += zero_filters

            if verbose and method == "weight":
                print("  -- [DEBUG][pruning] Layer {} | {} layer | {:.2f}% weights pruned".format(
                        layer_id,
                        'Conv' if len(parameter.data.size()) == 4 else 'Linear',
                        100.*zero_param_this_layer/param_this_layer
                        ))
            # elif verbose and method == "filter":
            #     print("  -- [DEBUG][pruning]Layer {} | {} layer | {:.2f}% filters pruned | {}/{}".format(
            #             layer_id,
            #             'Conv' if len(parameter.data.size()) == 4 else 'Linear',
            #             100.*(zero_filters)/parameter.shape[0],
            #             zero_filters,
            #             parameter.shape[0]
            #             ))
            elif verbose and method == "filter":
                print("  -- [DEBUG][pruning]Layer {} | {} layer | {:.2f}% filters pruned | {}/{} | {:.2f}% weights pruned".format(
                        layer_id,
                        'Conv' if len(parameter.data.size()) == 4 else 'Linear',
                        100.*(zero_filters)/parameter.shape[0],
                        zero_filters,
                        parameter.shape[0],
                        100.*zero_param_this_layer/param_this_layer
                        ))
    
    if method == "weight":
        pruning_perc = 100.*nb_zero_param/total_nb_param
    else:
        pruning_perc = 100.*total_zero_filters/total_filters
    if verbose:
        print("  -- [DEBUG][pruning] Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc


def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """

    if not a:
        return

    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    if not min_ix:
        print('Warning: all zero')
        return np.inf, np.inf

    # search for the smallest nonzero
    for i, e in enumerate(a):
         if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix
    
def are_masks_consistent(model, masks, debug=0):
    conv_params    = [p for p in model.parameters() if len(p.data.size()) == 4]
    inverted_masks = [abs(m - 1) for m in masks]

    if debug:
        pdb.set_trace()

    assert len(conv_params) == len(inverted_masks)

    sum_non_zero = 0
    for i in range(len(conv_params)):
        vals = conv_params[i] * inverted_masks[i].cuda()
        sum_non_zero += vals.sum((0,1,2,3)).item()

    return sum_non_zero == 0