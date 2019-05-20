import time
import math
import traceback
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from nets2_parsecfg import *
from nets2_utils import *

torch.cuda.empty_cache()
USE_GPU = torch.cuda.is_available()
print (' - USE_GPU : ', USE_GPU)

"""
URLs
    - YOLOv1 
        - https://pjreddie.com/darknet/yolov1/
            - wget http://pjreddie.com/media/files/yolov1/yolov1.weights (~800MB) [trained on 2007 train/val+ 2012 train/val]
    - YOLOv2
        - https://pjreddie.com/darknet/yolov2/
            - https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg [416 x 416]
            - wget https://pjreddie.com/media/files/yolov2-voc.weights (~MB) [trained on 2007 train/val+ 2012 train/val]
        - On other gthubs
            - wget http://pjreddie.com/media/files/yolo.weights
            - wget https://pjreddie.com/media/files/yolo-voc.weights
"""

# from region_loss import RegionLoss
#from layers.batchnorm.bn import BN2d

def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    anchor_step = len(anchors)/num_anchors
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW) 
    ty         = torch.zeros(nB, nA, nH, nW) 
    tw         = torch.zeros(nB, nA, nH, nW) 
    th         = torch.zeros(nB, nA, nH, nW) 
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW) 

    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    for b in xrange(nB):
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
        cur_ious = torch.zeros(nAnchors)
        for t in xrange(50):
            if target[b][t*5+1] == 0:
                break
            gx = target[b][t*5+1]*nW
            gy = target[b][t*5+2]*nH
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1).t()
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        conf_mask[b][cur_ious>sil_thresh] = 0
    if seen < 12800:
       if anchor_step == 4:
           tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
           ty = torch.FloatTensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
       else:
           tx.fill_(0.5)
           ty.fill_(0.5)
       tw.zero_()
       th.zero_()
       coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    for b in xrange(nB):
        for t in xrange(50):
            if target[b][t*5+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t*5+1] * nW
            gy = target[b][t*5+2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            gt_box = [0, 0, gw, gh]
            for n in xrange(nA):
                aw = anchors[anchor_step*n]
                ah = anchors[anchor_step*n+1]
                anchor_box = [0, 0, aw, ah]
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if anchor_step == 4:
                    ax = anchors[anchor_step*n+2]
                    ay = anchors[anchor_step*n+3]
                    dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                elif anchor_step==4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]

            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
            tw[b][best_n][gj][gi] = math.log(gw/anchors[anchor_step*best_n])
            th[b][best_n][gj][gi] = math.log(gh/anchors[anchor_step*best_n+1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t*5]
            if iou > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

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

### ------------------------------- <MODEL> ---------------------------- ###

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

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

# support route shortcut and reorg
class Darknet(nn.Module):

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks) # merge conv, bn,leaky
        self.loss = self.models[len(self.models)-1]

        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        if self.blocks[(len(self.blocks)-1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

    def forward(self, x):
        ind = -2
        self.loss = None
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1
            #if ind > 0:
            #    return x

            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'reorg' or block['type'] == 'avgpool' or block['type'] == 'softmax' or block['type'] == 'connected':
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1,x2),1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        return x

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()
    
        prev_filters = 3
        out_filters =[]
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad    = int((kernel_size-1)/2) if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
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
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
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
            for block in self.blocks:
                if ind >= len(self.models):
                    break
                ind = ind + 1
                if block['type'] == 'net':
                    continue
                elif block['type'] == 'convolutional':
                    model = self.models[ind]
                    batch_normalize = int(block['batch_normalize'])
                    if batch_normalize:
                        start = load_conv_bn(f, model[0], model[1])
                    else:
                        start = load_conv(f, model[0])
                elif block['type'] == 'connected':
                    model = self.models[ind]
                    if block['activation'] != 'linear':
                        start = load_fc(f, model[0])
                    else:
                        start = load_fc(f, model)
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

    def load_weights_old(self, weightfile):
        fp          = open(weightfile, 'rb')
        header      = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen   = self.header[3]
        buf         = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        ind = -2
        for block_i, block in enumerate(self.blocks):
            print ('')
            print (' ----------------- block : ', block_i, '(start:',start,')')
            print (' || --------------> block : ', block)
            if start >= buf.size:
                break
            ind = ind + 1
            print (' || ---------------> self.models[ind] : ', self.models[ind])
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                try:
                    model           = self.models[ind]
                    batch_normalize = int(block['batch_normalize'])
                    if batch_normalize:
                        start = load_conv_bn(buf, start, model[0], model[1])
                    else:
                        start = load_conv(buf, start, model[0])
                except:
                    print (' - [Err] Block :', block_i)
                    traceback.print_exc()
                    import sys; sys.exit(1)
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
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

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks)-1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    save_fc(fc, model)
                else:
                    save_fc(fc, model[0])
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
        fp.close()


def foo(x):
    import torch
    y = torch.rand(3, 4)
    y.copy_(x)
    return y

if __name__ == "__main__":
    if (1):
        if (0):
            cfgfile    = 'cfg/yolo.cfg'
            weightfile = '../demo/demo1/weights/yolo.weights'
        else:
            print (' - Loading YOLO')
            cfgfile    = 'cfg/yolo_voc.cfg'
            weightfile = '../../tmp/pytorch-yolo2/yolo-voc.weights'
            
        imgdir     = '../../tmp/pytorch-yolo2/data' # [ ]
        namesfile  = '../../tmp/pytorch-yolo2/data/voc.names'

        model = Darknet(cfgfile)
        print (' - Model is made. Loading weights ')
        model.load_weights(weightfile)

        use_cuda = USE_GPU
        if use_cuda:
            model.cuda()
        
        class_names = load_class_names(namesfile)
        for each in ['dog.jpg', 'eagle.jpg',  'giraffe.jpg',  'horses.jpg',  'person.jpg',  'scream.jpg']:
            try:
                print ('')
                imgfile = os.path.join(imgdir, each)        
                img     = Image.open(imgfile).convert('RGB')
                sized   = img.resize((model.width, model.height))

                start  = time.time()
                boxes  = do_detect(model, sized, 0.5, 0.4, use_cuda)
                finish = time.time()
                print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))
                plot_boxes(img, boxes, os.path.join('tmp', each), class_names)
            except:
                pass


    else:
        tmp = foo(torch.from_numpy(np.random.rand(3,4)))
        print (tmp)
        tmp = foo(torch.from_numpy(np.random.rand(12)))
        print (tmp)




    # print (model)