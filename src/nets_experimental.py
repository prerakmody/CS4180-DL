import cv2
import os
import sys
import math
import traceback
import numpy as np

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.nn.functional as F

from torchsummary import summary

from pruning.weightPruning.layers import MaskedConv2d
from pruning.weightPruning.methods import weight_prune
from pruning.weightPruning.utils import prune_rate

def model_parse_cfg(cfgfile, verbose=0):
    if verbose:
        print ('- cfgfile : ', cfgfile)
    blocks = []
    fp     = open(cfgfile, 'r')
    block  =  None
    line   = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue

        elif line[0] == '[':
            if block:
                if verbose:
                    print ('')
                    print (' - block : ', block)
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks

def model_print(blocks):
    print('layer     filters    size              input                output');
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters =[]
    out_widths =[]
    out_heights =[]
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block['type'] == 'net':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size-1)/2 if is_pad else 0
            width = (prev_width + 2*pad - kernel_size)/stride + 1
            height = (prev_height + 2*pad - kernel_size)/stride + 1
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = prev_width/stride
            height = prev_height/stride
            print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'max', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1
            height = 1
            print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' % (ind, 'avg', prev_width, prev_height, prev_filters,  prev_filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s                                    ->  %3d' % (ind, 'softmax', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            print('%5d %-6s                                     ->  %3d' % (ind, 'cost', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            filters = stride * stride * prev_filters
            width = prev_width/stride
            height = prev_height/stride
            print('%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'reorg', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
            if len(layers) == 1:
                print('%5d %-6s %d' % (ind, 'route', layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert(prev_width == out_widths[layers[1]])
                assert(prev_height == out_heights[layers[1]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'region':
            print('%5d %-6s' % (ind, 'detection'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id+ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected':
            filters = int(block['output'])
            print('%5d %-6s                            %3d x %3d x%4d  ->  %3d' % (ind, 'connected', prev_width, prev_height, prev_filters,  filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)

        elif block['type'] == 'local':
            # print('%5d %-6s' % (ind, 'local'))
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size-1)/2 if is_pad else 0
            width = (prev_width + 2*pad - kernel_size)/stride + 1
            height = (prev_height + 2*pad - kernel_size)/stride + 1
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'local', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)

        elif block['type'] == 'dropout':
            print('%5d %-6s' % (ind, 'dropout'))
        elif block['type'] == 'detection':
            print('%5d %-6s' % (ind, 'detection'))
        else:
            print('unknown type %s' % (block['type']))

def model_create(blocks, name):
        models = nn.ModuleList()
    
        prev_filters = 3 # default
        out_filters  = []
        conv_id      = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue

            elif block['type'] == 'convolutional':
                conv_id         = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters         = int(block['filters'])
                kernel_size     = int(block['size'])
                stride          = int(block['stride'])
                is_pad          = int(block['pad'])
                pad             = int((kernel_size-1)/2) if is_pad else 0
                activation      = block['activation']
                model           = nn.Sequential()
                
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), MaskedConv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), MaskedConv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride    = int(block['stride'])
                if name == 'YOLOv1':
                    model = nn.MaxPool2d(pool_size, stride)
                elif name == 'YOLOv2':
                    if stride > 1:
                        model = nn.MaxPool2d(pool_size, stride)
                    else:
                        model = MaxPoolStride1()
                out_filters.append(prev_filters)
                models.append(model)
            
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters * 7 * 7, filters, bias=False)
                    # model = nn.Linear(prev_filters * 7, filters)

                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            
            elif block['type'] == 'local':
                conv_id         = conv_id + 1
                filters         = int(block['filters'])
                kernel_size     = int(block['size'])
                stride          = int(block['stride'])
                is_pad          = int(block['pad'])
                pad             = int((kernel_size-1)/2) if is_pad else 0
                activation      = block['activation']
                model           = nn.Sequential()

                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), MaskedConv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), MaskedConv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            
            elif block['type'] == 'dropout':
                models.append(nn.Dropout(float(block['probability'])))

            elif block['type'] == 'detection':
                print ('---------- : ', block['type'])

            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                models.append(model)

            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)

            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model)
            
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors)/loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                models.append(loss)
            
            else:
                print('unknown type %s' % (block['type']))

        return models

def load_param(file, param):
    # print ('  -- param : ', param, ' || count : ', param.numel())
    print (' -- count : ', param.numel())
    param.data.copy_(torch.from_numpy(
        np.fromfile(file, dtype=np.float32, count=param.numel()).reshape(param.shape)
    ))

def load_conv(file, conv_model):
    load_param(file, conv_model.bias)
    load_param(file, conv_model.weight)

def load_conv_bn(file, conv_model, bn_model):
    load_param(file, bn_model.bias)
    load_param(file, bn_model.weight)
    load_param(file, bn_model.running_mean)
    load_param(file, bn_model.running_var)
    load_param(file, conv_model.weight)
    print ('  -- Successfully loaded!')

def load_fc(file, fc_model):
    print ('')
    print ('')
    print ('=======================================================================================')
    load_param(file, fc_model.bias)
    load_param(file, fc_model.weight)

## ---------------------------- YOLOV1 ---------------------------- ##

class YOLOv1(nn.Module):

    def __init__ (self, cfgfile, print_model=0):
        super(YOLOv1, self).__init__()
        self.blocks_json = model_parse_cfg(cfgfile,0)
        if print_model:
            model_print(self.blocks_json)
        self.blocks_nnmodules  = model_create(self.blocks_json, name='YOLOv1')

    def forward(self,x):
        verbose = 0
        i = -2
        for block in self.blocks_json:
            i = i + 1 #[-1, 0, 1 ...]

            if verbose and block['type'] in ['convolutional', 'maxpool', 'local', 'dropout', 'connected']:
                print ('')
                print (' --------------------------------------------- ')
                print (' -> i:', i, ' :: ', self.blocks_nnmodules[i])
                print (' -> ip:', x.shape)
                    

            if block['type'] in ['net', 'detection']:
                continue
            # if block['type'] in ['convolutional', 'maxpool', 'connected', 'dropput']:
            elif block['type'] in ['convolutional', 'maxpool', 'local', 'dropout']:
                try:
                    x = self.blocks_nnmodules[i](x)
                except:
                    print ('')
                    print (' - Error : idx :', i, ' || block : ', block)
                    print (self.blocks_nnmodules[i])
                    traceback.print_exc()
                    sys.exit(1)

            elif block['type'] == 'connected':
                x = x.view(x.size(0), -1)
                x = self.blocks_nnmodules[i](x)
            
            else:
                print (' - Unknown Block :', block)

            if verbose:
                print (' ---> op:', x.shape)

        return x

    def load_weights(self, weightfile):
        with open(weightfile, mode='rb') as f:
            major = np.fromfile(f, dtype=np.int32, count=1)
            minor = np.fromfile(f, dtype=np.int32, count=1)
            np.fromfile(f, dtype=np.int32, count=1)  # revision
            if major * 10 + minor >= 2 and major < 1000 and minor < 1000:
                np.fromfile(f, dtype=np.int64, count=1)  # seen
            else:
                np.fromfile(f, dtype=np.int32, count=1)  # seen

            ind = -2
            for block in self.blocks_json:
                try:
                    if ind >= len(self.blocks_nnmodules):
                        break
                    ind = ind + 1
                    print (' - (ind:',ind,') || block : ', block)
                    if block['type'] in ['net', 'detection']:
                        continue
                    elif block['type'] == 'convolutional':
                        model           = self.blocks_nnmodules[ind]
                        batch_normalize = int(block['batch_normalize'])
                        if batch_normalize:
                            load_conv_bn(f, model[0], model[1])
                        else:
                            load_conv(f, model[0])
                    elif block['type'] == 'connected':
                        model = self.models[ind]
                        if block['activation'] != 'linear':
                            load_fc(f, model[0])
                        else:
                            load_fc(f, model)
                    elif block['type'] == 'maxpool':
                        pass
                    elif block['type'] == 'reorg':
                        pass
                    elif block['type'] == 'route':
                        pass
                    elif block['type'] == 'shortcut':
                        pass
                    elif block['type'] == 'region':
                        pass
                    elif block['type'] == 'avgpool':
                        pass
                    elif block['type'] == 'softmax':
                        pass
                    elif block['type'] == 'cost':
                        pass
                    else:
                        print('unknown type %s' % (block['type']))
                except:
                    traceback.print_exc()

## ---------------------------- YOLOV2 ---------------------------- ##

class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target):
        #output : BxAs*(4+1+num_classes)*H*W
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        output   = output.view(nB, nA, (5+nC), nH, nW)
        x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()

        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        pred_boxes[0] = x.data + grid_x
        pred_boxes[1] = y.data + grid_y
        pred_boxes[2] = torch.exp(w.data) * anchor_w
        pred_boxes[3] = torch.exp(h.data) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
                                                               nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum().data[0])

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        tcls  = Variable(tcls.view(-1)[cls_mask].long().cuda())

        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
        cls        = cls[cls_mask].view(-1, nC)  

        t3 = time.time()

        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0], loss.data[0]))
        return loss

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view((B, C, int(H/hs), hs, int(W/ws), ws)).transpose(3,4).contiguous()
        x = x.view((B, C, int(H/hs*W/ws), hs*ws)).transpose(2,3).contiguous()
        x = x.view((B, C, hs*ws, int(H/hs), int(W/ws))).transpose(1,2).contiguous()
        x = x.view((B, hs*ws*C, int(H/hs), int(W/ws)))
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

class YOLOv2(nn.Module):

    def __init__(self, cfgfile, print_model=False):
        super(YOLOv2, self).__init__()
        self.blocks_json = model_parse_cfg(cfgfile,0)
        if print_model:
            model_print(self.blocks_json)
        self.blocks_nnmodules  = model_create(self.blocks_json, name='YOLOv2')

    def forward(self, x):
        pass

    def set_masks(self, masks):
        count = 0
        for m in self.blocks_nnmodules:
            try:
                if m[0].name == 'MaskedConv2d':
                    m[0].set_mask(masks[count])
                    count += 1
            except:
                print(m)


if __name__ == "__main__":
    if 1:
        cfg_file = 'yolov2-voc.cfg'
        weights_file = 'yolov2-voc.weights'
    else:
        cfg_file = '/home/strider/Work/Netherlands/TUDelft/1_Courses/Sem2/DeepLearning/Project/repo1/data/cfg/github_pjreddie/yolov1.cfg'
        weights_file = 'data/weights/github_pjreddie/yolov1.weights'


    TORCH_DEVICE = "cpu" # ["cpu", "cuda"]
    model = YOLOv2(cfg_file, weights_file).to(TORCH_DEVICE)


    params = [p for p in model.parameters()]

    pruning_perc = 90.
    masks = weight_prune(model, pruning_perc)

    #set_trace()
    model.set_masks(masks)

    prune_rate(model)
    # model.load_weights(weights_file)
    # op    = model(torch.rand((1,3,448,448)).to(TORCH_DEVICE))
    # print (' - op : ', op.shape)
    # summary(model, input_size=(3, 448, 448))