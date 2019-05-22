import os
import cv2
import tqdm
import traceback
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.nets import *
from src.dataloader import YoloDataset 

from google.colab.patches import cv2_imshow

USE_GPU = torch.cuda.is_available()

class YOLOv2Test():

    def __init__(self):
        pass
        

class YOLOv1Test():
    
    def __init__(self, model, model_chkp='', IMAGE_SIZE=448, IMAGE_GRID=7):
        if model_chkp != '':
            self.model = self.loadModelChkp(model, model_chkp)
        else:
            self.model = model
        
        if USE_GPU:
            model.cuda()

        self.IMAGE_SIZE = IMAGE_SIZE
        self.IMAGE_GRID = IMAGE_GRID

        self.VOC_CLASSES = (    # always index 0
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
        
        self.VOC_CLASSES_COLOR = [
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], 
            [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], 
            [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128]]

    def loadModelChkp(self, model, model_chkp):
        if os.path.exists(model_chkp):
            print ('  -- [TEST] Loading Chkpoint : ', model_chkp)
            checkpoint  = torch.load(model_chkp)
            epoch_start = checkpoint['epoch']
            print ('  -- [TEST] Start Epoch : ', epoch_start)
            print ('  -- [TEST][Loss] Train : ', checkpoint['loss_train'])
            print ('  -- [TEST][Loss] Val   : ', checkpoint['loss_val'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print ('')

    def test_decoder_nms(self, bboxes,scores,threshold=0.5):
        '''
        bboxes(tensor) [N,4]
        scores(tensor) [N,]
        '''
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]
        areas = (x2-x1) * (y2-y1)

        _,order = scores.sort(0,descending=True)
        keep = []
        while order.numel() > 0:
            try:
                if order.numel() > 1:
                    i = order[0]
                elif order.numel() == 1:
                    i = order.item()

                keep.append(i)

                if order.numel() == 1:
                    break

                xx1 = x1[order[1:]].clamp(min=x1[i])
                yy1 = y1[order[1:]].clamp(min=y1[i])
                xx2 = x2[order[1:]].clamp(max=x2[i])    
                yy2 = y2[order[1:]].clamp(max=y2[i])

                w = (xx2-xx1).clamp(min=0)
                h = (yy2-yy1).clamp(min=0)
                inter = w*h

                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                ids = (ovr<=threshold).nonzero().squeeze()
                if ids.numel() == 0:
                    break
                order = order[ids+1]
            except:
                traceback.print_exc()
                print (' ------------ [ERR!] order : ', order)
                import sys; sys.exit(1)

        return torch.LongTensor(keep)
    
    def test_decoder(self, Y):
        '''
            Input  : Y (tensor) : [1 x 7 x 7 x 30]
            return : (tensor) box[[x1,y1,x2,y2]] label[...]
        '''
        res_boxes      = []
        res_cls_indexs = []
        res_probs      = []
        CELL_SIZE      = 1./self.IMAGE_GRID

        Y        = Y.data
        Y        = Y.squeeze(0) #7x7x30
        contain1 = Y[:,:,4].unsqueeze(2)
        contain2 = Y[:,:,9].unsqueeze(2)
        contain  = torch.cat((contain1,contain2),2)
        mask1    = contain > 0.1
        mask2    = (contain == contain.max()) # we always select the best contain_prob what ever it>0.9
        mask     = (mask1 + mask2).gt(0)
        # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
        for i in range(self.IMAGE_GRID):
            for j in range(self.IMAGE_GRID):
                for b in range(2):
                    # index = min_index[i,j]
                    # mask[i,j,index] = 0
                    if mask[i,j,b] == 1:
                        #print(i,j,b)
                        box          = Y[i,j,b*5:b*5+4]
                        contain_prob = torch.FloatTensor([Y[i,j,b*5+4]])
                        xy           = torch.FloatTensor([j,i]) * CELL_SIZE # cell左上角  up left of cell
                        box[:2]      = box[:2] * CELL_SIZE + xy               # return cxcy relative to image
                        box_xy       = torch.FloatTensor(box.size())        # 转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                        box_xy[:2]   = box[:2] - 0.5*box[2:]
                        box_xy[2:]   = box[:2] + 0.5*box[2:]
                        max_prob,cls_index = torch.max(Y[i,j,10:],0)
                        if float((contain_prob * max_prob)[0]) > 0.1:
                            res_boxes.append(box_xy.view(1,4))
                            res_cls_indexs.append(cls_index)
                            res_probs.append(contain_prob*max_prob)

        if len(res_boxes) ==0:
            res_boxes      = torch.zeros((1,4))
            res_probs      = torch.zeros(1)
            res_cls_indexs = torch.zeros(1)
        else:
            res_boxes      = torch.cat(res_boxes, dim=0)      #(n,4)
            res_probs      = torch.cat(res_probs, dim=0)      #(n,)
            res_cls_indexs = torch.stack(res_cls_indexs, dim=0) #(n,)
        
        
        keep = self.test_decoder_nms(res_boxes,res_probs)
        return res_boxes[keep], res_cls_indexs[keep], res_probs[keep]

    def test_plot(self, axarr, X, yHat):
        boxes, cls_indexs, probs = yHat
        
        if USE_GPU : img = X.cpu().data.numpy().transpose(1,2,0) 
        else       : img = X.data.numpy().transpose(1,2,0)
        img    = img*255; 
        img    = img.astype(np.uint8)
        h,w,_  = img.shape
        img    = img.copy()
        # print ('  -- img : ', img.dtype, ' || Contiguous : ', img.flags['C_CONTIGUOUS'])

        result = []
        for i,box in enumerate(boxes):
            x1        = int(box[0]*w)
            x2        = int(box[2]*w)
            y1        = int(box[1]*h)
            y2        = int(box[3]*h)
            cls_index = cls_indexs[i]
            cls_index = int(cls_index) # convert LongTensor to int
            prob      = probs[i]
            prob      = float(prob)
            result.append([(x1,y1),(x2,y2), self.VOC_CLASSES[cls_index], '', prob])

        for left_up, right_bottom, class_name, _, prob in result:
            color               = self.VOC_CLASSES_COLOR[self.VOC_CLASSES.index(class_name)]
            label               = class_name + str(round(prob,2))
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1                  = (left_up[0], left_up[1]- text_size[1])
            # import pdb; pdb.set_trace()
            cv2.rectangle(img, left_up, right_bottom, color, 2)                
            cv2.rectangle(img, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
            # cv2.putText(img, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
            cv2.putText(img, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255))
            
        
        # axarr.imshow(img)
        cv2_imshow(img)
        # cv2.imshow('image',img)

    def test(self, X, Y, plot=0):
        X = Variable(X)
        if USE_GPU:
            X = X.cuda()
        yHat = self.model(X) # [N x 7 x 7 x 30]
        yHat = yHat.cpu()

        boxes_pred       = []
        cls_indexs_pred  = []
        probs_pred       = []

        if plot : 
            f,axarr = plt.subplots(len(yHat),2, figsize=(10,10))
        
        for i_batch, yHat_ in enumerate(yHat):
            # print ('  -- yHat_ : ', yHat_.shape)
            boxes_pred_, cls_indexs_pred_, probs_pred_ = self.test_decoder(yHat_)
            boxes_pred.append(boxes_pred_)
            cls_indexs_pred.append(cls_indexs_pred_)
            probs_pred.append(probs_pred_)
            if (plot):
                self.test_plot(axarr[i_batch][0], X[i_batch], [boxes_pred_, cls_indexs_pred_, probs_pred_])
                print (' - Y : ', Y.shape, ' || ', Y[i_batch].shape)
                boxes_true_, cls_indexs_true_, probs_true_ = self.test_decoder(Y[i_batch])
                self.test_plot(axarr[i_batch][1], X[i_batch], [boxes_true_, cls_indexs_true_, probs_true_])

        return [boxes_pred, cls_indexs_pred, probs_pred]

    def test_metrics_vocap(self, rec,prec,use_07_metric=False):
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0.,1.1,0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec>=t])
                ap = ap + p/11.
        else:
            # correct ap caculation
            mrec = np.concatenate(([0.],rec,[1.]))
            mpre = np.concatenate(([0.],prec,[0.]))

            for i in range(mpre.size -1, 0, -1):
                mpre[i-1] = np.maximum(mpre[i-1],mpre[i])

            i = np.where(mrec[1:] != mrec[:-1])[0]

            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    def getmAP(self, file_annotations, YoloDatasetTest, BATCH_SIZE=16):
        target =  defaultdict(list)
        preds  = defaultdict(list)

        if (1):
            file_list = []
            for line in open(file_annotations).readlines():
                file_list.append(line.strip().split())

            for index,image_file in enumerate(file_list):
                # image_id = image_file[0]
                # image_list.append(image_id)
                # image_list.append(index)
                num_obj = (len(image_file) - 1) // 5
                for i in range(num_obj):
                    x1 = int(image_file[1+5*i])
                    y1 = int(image_file[2+5*i])
                    x2 = int(image_file[3+5*i])
                    y2 = int(image_file[4+5*i])
                    c = int(image_file[5+5*i])
                    class_name = self.VOC_CLASSES[c]
                    target[(index,class_name)].append([x1,y1,x2,y2])
                    if index == 0:
                        print (';')
                        

        if (1):
            image_id = 0
            DataLoaderTest = DataLoader(YoloDatasetTest, batch_size=BATCH_SIZE, shuffle=False,num_workers=0)
            with tqdm.tqdm_notebook(total = len(DataLoaderTest)*BATCH_SIZE) as pbar:
                for i,(X,Y) in enumerate(DataLoaderTest):
                    pbar.update(BATCH_SIZE)
                    yHat = testObj.test(X, Y, plot=False)                
                    [boxes_pred_batch, cls_indexs_pred_batch, probs_pred_batch] = yHat

                    for i_batch, boxes_pred_img in enumerate(boxes_pred_batch):
                        w = 448; h = 448;
                        result = []
                        for i,box in enumerate(boxes_pred_img):
                            x1        = int(box[0]*w)
                            x2        = int(box[2]*w)
                            y1        = int(box[1]*h)
                            y2        = int(box[3]*h)
                            cls_index = cls_indexs_pred_batch[i_batch][i]
                            cls_index = int(cls_index) # convert LongTensor to int
                            prob      = probs_pred_batch[i_batch][i]
                            prob      = float(prob)
                            class_name = self.VOC_CLASSES[cls_index]
                            preds[class_name].append([image_id, prob, x1,y1,x2,y2])

                        image_id += 1

        self.test_metrics(preds, target)                

    def test_metrics(self, preds, target, threshold=0.5, use_07_metric=False,):

        '''
        preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
        target {(image_id,class):[[],]}
        '''
        aps = []
        for i,class_ in enumerate(self.VOC_CLASSES):
            pred = preds[class_] #[[image_id,confidence,x1,y1,x2,y2],...]
            if len(pred) == 0: # (If there is no abnormality detected in this category)
                ap = -1
                print('---class {} ap {}---'.format(class_,ap))
                aps += [ap]
                break
            #print(pred)
            image_ids  = [x[0] for x in pred]
            confidence = np.array([float(x[1]) for x in pred])
            BB         = np.array([x[2:] for x in pred])
            
            # sort by confidence
            sorted_ind    = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB            = BB[sorted_ind, :]
            image_ids     = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            npos = 0.
            for (key1,key2) in target:
                if key2 == class_:
                    npos += len(target[(key1,key2)]) #Statistics of positive samples of this category, statistics will not be missed here.
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d,image_id in enumerate(image_ids):
                bb = BB[d] #预测框
                if (image_id,class_) in target:
                    BBGT = target[(image_id,class_)] #[[],]
                    for bbgt in BBGT:
                        # compute overlaps
                        # intersection
                        ixmin = np.maximum(bbgt[0], bb[0])
                        iymin = np.maximum(bbgt[1], bb[1])
                        ixmax = np.minimum(bbgt[2], bb[2])
                        iymax = np.minimum(bbgt[3], bb[3])
                        iw = np.maximum(ixmax - ixmin + 1., 0.)
                        ih = np.maximum(iymax - iymin + 1., 0.)
                        inters = iw * ih

                        union = (bb[2]-bb[0]+1.)*(bb[3]-bb[1]+1.) + (bbgt[2]-bbgt[0]+1.)*(bbgt[3]-bbgt[1]+1.) - inters
                        if union == 0:
                            print(bb,bbgt)
                        
                        overlaps = inters/union
                        if overlaps > threshold:
                            tp[d] = 1
                            BBGT.remove(bbgt) #这个框已经匹配到了，不能再匹配
                            if len(BBGT) == 0:
                                del target[(image_id,class_)] #删除没有box的键值
                            break
                    fp[d] = 1-tp[d]
                else:
                    fp[d] = 1
            fp   = np.cumsum(fp)
            tp   = np.cumsum(tp)
            rec  = tp/float(npos)
            prec = tp/np.maximum(tp + fp, np.finfo(np.float64).eps)
            #print(rec,prec)
            
            ap = self.test_metrics_vocap(rec, prec, use_07_metric)
            print('---class {} ap {}---'.format(class_,ap))
            aps += [ap]

        mAP = np.mean(aps)
        print('---map {}---'.format(mAP))
        return  mAP


if __name__ == "__main__":

    if (1):
        dir_main = './yolo'
        dir_annotations  = os.path.join(dir_main, 'data/VOCdevkit_test/VOC2007')
        file_annotations = os.path.join(dir_main, 'data/VOCdevkit_test/VOC2007/anno_test.txt')
        flagTrain        = False
        flagAug          = False
        BATCH_SIZE       = 2
        IMAGE_SIZE       = 448
        IMAGE_GRID       = 7

        YoloDatasetTest  = YoloDataset(dir_annotations, file_annotations
                                    , flagTrain
                                    , IMAGE_SIZE, IMAGE_GRID
                                    , flagAug
                                    , transform = [transforms.ToTensor()] )
        DataLoaderTest   = DataLoader(YoloDatasetTest, batch_size=BATCH_SIZE, shuffle=False,num_workers=0)
        print (' - 1. Dataset Loaded')
    
    if (1):
        model_name = 'yolov1'
        model = getYOLOv1(model_name)
        testObj = YOLOv1Test(model, model_chkp='')
        print (' - 2. Model Loaded')

    if (0):
        with torch.no_grad():
            for i,(X,Y) in enumerate(DataLoaderTest):
                print (' - 3. Image : ', i, ' || X:', X.shape)
                yHat = testObj.test(X, Y, plot=True)
                break
    else:
        testObj.getmAP(file_annotations, YoloDatasetTest)