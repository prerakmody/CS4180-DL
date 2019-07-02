import os
import cv2
import random
import traceback
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import pickle
import xml.etree.ElementTree as ET

import torch
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import transforms

## --------------------------------------- PASCAL VOC - v2 --------------------------------------- ##

class VOCDatasetv2(data.Dataset):

    def __init__(self, IMAGELIST_TXT, shape=None, shuffle=True, transform=None, target_transform=None, train=False, num_workers=1, seen=0, verbose=0):
        with open(IMAGELIST_TXT, 'r') as file:
            self.lines = file.readlines()

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples         = len(self.lines)
        self.transform        = transform
        self.target_transform = target_transform
        self.train            = train
        self.shape            = shape

        self.seen             = seen
        self.num_workers      = num_workers
        self.verbose          = verbose

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), ' - [ERROR][dataloader.py] Index range error'
        imgpath = self.lines[index].rstrip()

        if self.verbose:
            print ('  -- [DEBUG][VOCDatasetv2] index : ', index)
            print ('  -- [DEBUG][VOCDatasetv2] imgpath : ', imgpath)
        
        if (0): # different images sizes
            if self.train and index % 64== 0:
                if self.seen < 4000*64:
                    width = 13*32
                    self.shape = (width, width)
                elif self.seen < 8000*64:
                    width = (random.randint(0,3) + 13)*32
                    self.shape = (width, width)
                elif self.seen < 12000*64:
                    width = (random.randint(0,5) + 12)*32
                    self.shape = (width, width)
                elif self.seen < 16000*64:
                    width = (random.randint(0,7) + 11)*32
                    self.shape = (width, width)
                else: # self.seen < 20000*64:
                    width = (random.randint(0,9) + 10)*32
                    self.shape = (width, width)

        if self.train:
            jitter     = 0.2
            hue        = 0.1
            saturation = 1.5 
            exposure   = 1.5

            img, label = getData(imgpath, self.shape, jitter, hue, saturation, exposure)
            label      = torch.from_numpy(label)

        else:
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
    
            labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            label   = torch.zeros(50*5)

            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))
            except Exception:
                traceback.print_exc()
                tmp = torch.zeros(1,5)

            tmp = tmp.view(-1)
            tsz = tmp.numel()
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        
        return (img, label.float())

# --------------- DATA AUGMENTATIONS --------------- #
def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)

    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img, flip, dx,dy,sx,sy 

# --------------- DATA AND LABELS --------------- #

def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])

    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(int(truths.size/5), 5) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)

# Unused
def getLabels(labpath):
    max_boxes = 50
    label     = np.zeros((max_boxes,5))
    cc        = 0

    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)  # the values here are [norm_centre_x, norm_centre_y, norm_width, norm_height] wrt image-origin (top-left)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))

        for i in range(bs.shape[0]):
            label[cc] = bs[i]
            cc        += 1
            if cc >= 50:
                break
        
        label = np.reshape(label, (-1))
    
    return label

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes,5))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)   # the values here are [norm_centre_x, norm_centre_y, norm_width, norm_height] wrt image-origin (top-left)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0

        for i in range(bs.shape[0]):
            if (1):
                # xmin, ymin
                x1 = bs[i][1] - bs[i][3]/2
                y1 = bs[i][2] - bs[i][4]/2
                # xmax, ymax
                x2 = bs[i][1] + bs[i][3]/2
                y2 = bs[i][2] + bs[i][4]/2
            
            if (1):
                x1 = min(0.999, max(0, x1 * sx - dx)) 
                y1 = min(0.999, max(0, y1 * sy - dy)) 
                x2 = min(0.999, max(0, x2 * sx - dx))
                y2 = min(0.999, max(0, y2 * sy - dy))
            
            if (1):
                bs[i][1] = (x1 + x2)/2 # x_centre
                bs[i][2] = (y1 + y2)/2 # y_centre
                bs[i][3] = (x2 - x1)   # width
                bs[i][4] = (y2 - y1)   # height

            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
            
            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue

            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label

def getData(imgpath, shape, jitter, hue, saturation, exposure):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

    ## data augmentation
    img                  = Image.open(imgpath).convert('RGB')
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    label                = fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)

    return img,label


## --------------------------------------- PASCAL VOC - v2 (voc_label.py) --------------------------------------- ##

# Step3 : converts box corners to centre and width,height
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    # centre of BBox
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    # Dimensions of BBox
    w = box[1] - box[0]
    h = box[3] - box[2]

    # Normalization
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return (x,y,w,h)

# Step2 : reads .xml () and converts to .txt
def convert_annotation(DATA_DIR, year, image_id):
    classes  = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    in_file  = open(os.path.join(DATA_DIR, 'VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id)))
    out_file = open(os.path.join(DATA_DIR,'VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id)), 'w')
    tree     = ET.parse(in_file)
    root     = tree.getroot()
    size     = root.find('size')
    w        = int(size.find('width').text)
    h        = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls       = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b      = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb     = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# Step1 : entry-point
def setup_VOC(DATA_DIR):

    sets    = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

    for year, image_set in sets:
        print (' - year : ', year, ' || image_set : ', image_set)
        if not os.path.exists(os.path.join(DATA_DIR, 'VOCdevkit/VOC%s/labels/'%(year))):
            os.makedirs(os.path.join(DATA_DIR, 'VOCdevkit/VOC%s/labels/'%(year)))

        image_ids = open(os.path.join(DATA_DIR, 'VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set))).read().strip().split()
        list_file = open(os.path.join(DATA_DIR, 'VOCdevkit/%s_%s.txt'%(year, image_set)), 'w')
        for image_id in image_ids:
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(DATA_DIR, year, image_id))
            convert_annotation(DATA_DIR, year, image_id)
        list_file.close()