import os
import cv2
import pdb
import tqdm
import traceback
import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import requests

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.nets import *
from src.dataloader import * 
from src.nets2_utils import *

USE_GPU = torch.cuda.is_available()

class PASCALVOCEval():

    def __init__(self, MODEL, MODEL_CFGFILE, MODEL_WEIGHTFILE, MODEL_LOSS
    , PASCAL_DIR, EVAL_IMAGELIST, EVAL_OUTPUTDIR, EVAL_PREFIX, EVAL_OUTPUTDIR_PKL
    , LOGGER='', LOGGER_EPOCH=-1, verbose=0):
        self.MODEL            = MODEL
        self.MODEL_CFGFILE    = MODEL_CFGFILE
        self.MODEL_WEIGHTFILE = MODEL_WEIGHTFILE
        self.MODEL_LOSS       = MODEL_LOSS
        self.PASCAL_DIR       = PASCAL_DIR
        self.EVAL_IMAGELIST   = EVAL_IMAGELIST
        self.EVAL_OUTPUTDIR   = EVAL_OUTPUTDIR
        self.EVAL_PREFIX      = EVAL_PREFIX
        self.EVAL_OUTPUTDIR_PKL = EVAL_OUTPUTDIR_PKL
        self.LOGGER             = LOGGER
        self.LOGGER_EPOCH       = LOGGER_EPOCH
        self.USE_GPU            = torch.cuda.is_available()

        self.verbose            = verbose

        self.VOC_CLASSES = (    # always index 0
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
        
        self.VOC_CLASSES_ = ('__background__', # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor') 
        self.VOC_YEAR = '2007'
    
    # Step1 - Entry Point - Make .txt files with predictions
    def predict(self, BATCH_SIZE=2, CONF_THRESH=0.005,NMS_THRESH=0.45):
        # above default params from here --> https://github.com/marvis/pytorch-yolo2/blob/6c7c1561b42804f4d50d34e0df220c913711f064/valid.py#L47
        # CONF_THRESH=0.25,NMS_THRESH=0.45, IOU_THRESH    = 0.5

        # Step1 - Get Model
        if (1):
            if self.MODEL == '' or self.MODEL == None:
                print (' - 1. Loading model : ', self.MODEL_WEIGHTFILE)
                self.MODEL = getYOLOv2(self.MODEL_CFGFILE, self.MODEL_WEIGHTFILE)
            self.MODEL.eval()

        # Step2 - Get Dataset
        if (1):
            with open(self.EVAL_IMAGELIST) as fp:
                tmp_files   = fp.readlines()
                valid_files = [item.rstrip() for item in tmp_files]

            eval_dataset = VOCDatasetv2(self.EVAL_IMAGELIST, shape=(self.MODEL.width, self.MODEL.height),
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]))
            kwargs = {'num_workers': 1, 'pin_memory': True}
            eval_loader = torch.utils.data.DataLoader(
                eval_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs) 

        # Step3 - Create File pointers for prediction storage (after removing the older files) 
        if (1): 
            fps = [0]*self.MODEL.num_classes
            if not os.path.exists(self.EVAL_OUTPUTDIR):
                os.mkdir(self.EVAL_OUTPUTDIR)
            else:
                for i in range(self.MODEL.num_classes):
                    buf = '%s/%s%s.txt' % (self.EVAL_OUTPUTDIR, self.EVAL_PREFIX, self.VOC_CLASSES[i])
                    if os.path.exists(buf):
                        os.remove(buf)
                # Should I delete folder and remake??
            for i in range(self.MODEL.num_classes):
                buf = '%s/%s%s.txt' % (self.EVAL_OUTPUTDIR, self.EVAL_PREFIX, self.VOC_CLASSES[i])
                fps[i] = open(buf, 'w')
    
        lineId = -1
        verbose = 0
        
        with torch.no_grad():
            val_loss_total       = 0.0
            with tqdm.tqdm_notebook(total = len(eval_loader)*BATCH_SIZE) as pbar:
                
                for batch_idx, (data, target) in enumerate(eval_loader):
                    pbar.update(BATCH_SIZE)

                    if self.USE_GPU:
                        data   = data.cuda()
                        # target = target.cuda()                        
                    data, target = Variable(data), Variable(target)
                    output       = self.MODEL(data).data
                    
                    if self.LOGGER != '':
                        if self.MODEL_LOSS != None:
                            if (len(target[target != 0.0])):
                                try:
                                    val_loss     = self.MODEL_LOSS(output, target)
                                    val_loss_total += val_loss.data
                                    if self.verbose:
                                        print (' - loss : ', val_loss)
                                except:
                                    traceback.print_exc()
                                    pdb.set_trace()
                            else:
                                print (' - No annotations : ', valid_files[lineId])

                    batch_boxes = get_region_boxes(output, CONF_THRESH, self.MODEL.num_classes, self.MODEL.anchors, self.MODEL.num_anchors, 0, 1)
                    
                    for i in range(output.size(0)): # output.size(0) = batch_size
                        lineId        = lineId + 1
                        fileId        = os.path.basename(valid_files[lineId]).split('.')[0]
                        width, height = get_image_size(valid_files[lineId])
                        # print(valid_files[lineId])
                        boxes = batch_boxes[i]
                        boxes = nms(boxes, NMS_THRESH)
                        for box in boxes: # box = [x,y,w,h, box_conf, class_conf, cls_id]
                            # Top-Left Corner (xmin, xmax)
                            x1 = (box[0] - box[2]/2.0) * width   # x - w/2 (x = centre of BBox)
                            y1 = (box[1] - box[3]/2.0) * height  # y - h/2
                            # Top-Right Corner (ymin, ymax)
                            x2 = (box[0] + box[2]/2.0) * width   # x + h/2
                            y2 = (box[1] + box[3]/2.0) * height  # y + h/2

                            box_conf = box[4]
                            for j in range(int((len(box)-5)/2)):
                                cls_conf = box[5+2*j]
                                cls_id   = box[6+2*j]
                                prob     = box_conf * cls_conf
                                fps[cls_id].write('%s %f %f %f %f %f\n' % (fileId, prob, x1, y1, x2, y2)) # for each class_id, write down [prob, x1,y1,x2,y2]

                        if (verbose):
                            print ('    -- Time : imread : ', round(t32 - t31,4), ' || boxes loop : ', round(t33 - t32, 4))
                                

        if self.LOGGER != '':
            if self.MODEL_LOSS != None:
                self.LOGGER.save_value('Total Loss', 'Val Loss', self.LOGGER_EPOCH+1, val_loss_total / len(eval_loader))

        for i in range(self.MODEL.num_classes):
            fps[i].close()
        
        mAP, finalMAP = self._do_python_eval()
        if self.LOGGER != '' :
            if self.MODEL_LOSS != None:
                self.LOGGER.save_value('mAP', 'Val mAP', self.LOGGER_EPOCH+1, mAP)
                for mAP_obj in finalMAP:
                    self.LOGGER.save_value('AP - Classes', mAP_obj[0], self.LOGGER_EPOCH+1, mAP_obj[1])

    # Step2.2 - Reading XML annotations
    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                int(bbox.find('ymin').text),
                                int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)]
            objects.append(obj_struct)

        return objects

    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    # Step2.1 - VOC for each file
    def voc_eval(self, file_predictions,
                file_xml_annotations,
                file_test_images,
                classname,
                dir_cache,
                ovthresh=0.5,
                use_07_metric=False):
        """rec, prec, ap = voc_eval(detpath,
                                    annopath,
                                    imagesetfile,
                                    classname,
                                    [ovthresh],
                                    [use_07_metric])

        Top level function that does the PASCAL VOC evaluation.

        detpath: Path to detections
            detpath.format(classname) should produce the detection results file.
        annopath: Path to annotations
            annopath.format(imagename) should be the xml annotations file.
        imagesetfile: Text file containing the list of images, one image per line.
        classname: Category name (duh)
        cachedir: Directory for caching the annotations
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
            (default False)
        """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)
        # assumes imagesetfile is a text file with each line an image name
        # cachedir caches the annotations in a pickle file

        # first load gt

        ## -------------------------- STEP 0 (load cached file) -------------------------- ##
        if (1):
            if not os.path.isdir(dir_cache):
                os.mkdir(dir_cache)
            file_cache = os.path.join(dir_cache, 'annots.pkl')
            
            with open(file_test_images, 'r') as f:
                lines = f.readlines()
            imagenames = [x.strip() for x in lines]

            if not os.path.isfile(file_cache):
                # load annots
                recs = {}
                for i, imagename in enumerate(imagenames):
                    recs[imagename] = self.parse_rec(file_xml_annotations.format(imagename))
                    # if i % 100 == 0:
                    #     # print ('Reading annotation for {0}/{1}'.format(i + 1, len(imagenames))
                    #     print ('Reading annotation for ', i+1, '/', len(imagenames))
                # save
                # print ('Saving cached annotations to {0}'.format(cachefile))

                with open(file_cache, 'wb') as f:
                    cPickle.dump(recs, f)

            else:
                # load
                with open(file_cache, 'rb') as f:
                    recs = cPickle.load(f)

        ## -------------------------- STEP 0 (init) -------------------------- ##
        if (1):
            # extract gt objects for this class
            class_recs = {}
            npos       = 0
            for imagename in imagenames:
                R = [obj for obj in recs[imagename] if obj['name'] == classname]
                bbox = np.array([x['bbox'] for x in R])
                difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
                det = [False] * len(R)
                npos = npos + sum(~difficult)
                class_recs[imagename] = {'bbox': bbox,
                                        'difficult': difficult,
                                        'det': det}

            # read dets
            detfile = file_predictions.format(classname)
            with open(detfile, 'r') as f:
                lines = f.readlines()

            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # sort by confidence
            try:
                sorted_ind = np.argsort(-confidence)
                sorted_scores = np.sort(-confidence)
                BB = BB[sorted_ind, :]
                image_ids = [image_ids[x] for x in sorted_ind]
            except:
                print ('  -- [ERROR][PASCALVOCEval.voc_eval()] classname : ', classname)
                # traceback.print_exc()

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)

                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

        ## -------------------------- STEP 3 (return results) -------------------------- ##
        if (1):
            # compute precision recall
            fp  = np.cumsum(fp)
            tp  = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap   = self.voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap

    # Step2 - Compare .txt files with predictions
    def _do_python_eval(self):
        ## -------------------------- STEP 0 (init) -------------------------- ##
        if (1):
            print ('  -- Reading predictions from : ', self.EVAL_OUTPUTDIR, '/',self.EVAL_PREFIX,'*')
            res_prefix           = os.path.join(self.EVAL_OUTPUTDIR, self.EVAL_PREFIX)
            file_predictions     = res_prefix + '{:s}.txt' # file_predictions
            file_xml_annotations = os.path.join(self.PASCAL_DIR, 'VOC' + self.VOC_YEAR, 'Annotations','{:s}.xml') # file_xml_annotations
            file_test_images     = os.path.join(self.PASCAL_DIR, 'VOC' + self.VOC_YEAR,'ImageSets','Main','test.txt') # file_test_images
            dir_cache            = os.path.join(self.PASCAL_DIR, 'annotations_cache') # dir_cache
            aps                  = []

            # The PASCAL VOC metric changed in 2010
            use_07_metric = True if int(self.VOC_YEAR) < 2010 else False
            # print (' - VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

        ## -------------------------- STEP 1 (per class AP) -------------------------- ##
        finalMAP = []
        # with tqdm.tqdm_notebook(total = len(self.VOC_CLASSES_)) as pbar:
        for i, cls in enumerate(self.VOC_CLASSES_):
            # pbar.update(1)
            if cls == '__background__':
                continue
            rec, prec, ap = self.voc_eval(
                file_predictions, file_xml_annotations, file_test_images, cls, dir_cache, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            finalMAP.append([cls, ap])

        ## -------------------------- STEP 2 (print results) -------------------------- ##
        df_MAP = pd.DataFrame(finalMAP, columns=['class', 'mAP'])        
        mAP = np.mean(aps)
        print('Mean AP = {:.4f}'.format(mAP))
        return mAP, finalMAP