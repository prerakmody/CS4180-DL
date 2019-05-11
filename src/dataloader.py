import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data

class YoloDataset(data.Dataset):
    
    def __init__(self, dir_data, file_annotations
                 , train
                 , image_size, grid_num
                 , flag_augm
                 , transform):
        
        self.dir_data   = dir_data
        self.dir_img    = os.path.join(dir_data, 'JPEGImages')
        self.train      = train
        self.transform  = transform
        
        self.fnames     = []
        self.boxes      = []
        self.labels     = []
        self.mean       = (123,117,104) # RGB ([How?])
        
        self.grid_num   = grid_num
        self.image_size = image_size
        self.flag_augm  = flag_augm

        with open(file_annotations) as f:
            for line in f.readlines():
                splited   = line.strip().split()
                self.fnames.append(splited[0])
                num_boxes = (len(splited) - 1) // 5
                box       = []
                label     = []
                for i in range(num_boxes):
                    x  = float(splited[1+5*i])
                    y  = float(splited[2+5*i])
                    x2 = float(splited[3+5*i])
                    y2 = float(splited[4+5*i])
                    c  = splited[5+5*i]
                    box.append([x,y,x2,y2])
                    label.append(int(c)+1)
                self.boxes.append(torch.Tensor(box))
                self.labels.append(torch.LongTensor(label))
                
        self.num_samples = len(self.boxes)
    
    def __getitem__(self,idx, verbose=0):
        fname  = self.fnames[idx]
        img    = cv2.imread(os.path.join(self.dir_img, fname), cv2.IMREAD_UNCHANGED)
        boxes  = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        
        if (0):
            print (' - fname :', fname)
            print (' - path : ', os.path.join(self.dir_img, fname))
            # plt.imshow(img)
            print (' - labels : ', labels)
        
        if self.train:
            if (self.flag_augm == 1):
                if (0):
                    img = self.random_bright(img)
                img, boxes       = self.random_flip(img, boxes)
                img,boxes        = self.randomScale(img,boxes)
                img              = self.randomBlur(img)
                img              = self.RandomBrightness(img)
                img              = self.RandomHue(img)
                img              = self.RandomSaturation(img)
                img,boxes,labels = self.randomShift(img,boxes,labels)
                img,boxes,labels = self.randomCrop(img,boxes,labels)

        h,w,_  = img.shape
        
        boxes  /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        img    = self.BGR2RGB(img) #because pytorch pretrained model use RGB
        #img    = self.subMean(img,self.mean) 
        img    = cv2.resize(img,(self.image_size,self.image_size))
        target = self.encoder(boxes,labels) # 7x7x30
        for t in self.transform:
            img = t(img)
        return img,target
    
    def __len__(self):
        return self.num_samples
    
    def encoder(self,boxes,labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 7x7x30
        '''
        
        target    = torch.zeros((self.grid_num, self.grid_num,30))
        cell_size = 1./self.grid_num
        wh        = boxes[:,2:] - boxes[:,:2]
        cxcy      = (boxes[:,2:] + boxes[:,:2])/2
        for i in range(cxcy.size()[0]):
            cxcy_sample                       = cxcy[i]
            ij                                = (cxcy_sample/cell_size).ceil()-1 #
            target[int(ij[1]),int(ij[0]),4]   = 1
            target[int(ij[1]),int(ij[0]),9]   = 1
            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1
            xy                                = ij*cell_size # The relative coordinates of the upper left corner of the matched mesh
            delta_xy                          = (cxcy_sample -xy)/cell_size
            target[int(ij[1]),int(ij[0]),2:4] = wh[i]
            target[int(ij[1]),int(ij[0]),:2]  = delta_xy
            target[int(ij[1]),int(ij[0]),7:9] = wh[i]
            target[int(ij[1]),int(ij[0]),5:7] = delta_xy
            
        return target
    
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    
    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr  = bgr - mean
        return bgr
    
    def display(self, X):
        plt.imshow(X.data.numpy().transpose(1,2,0))
    
    def display_anno(self, X,y):
        pass
    
    

if __name__ == "__main__":
    pass
    # dir_annotations  = 'yolo/data/VOCdevkit_trainval/VOC2007'
    # file_annotations = 'yolo/data/VOCdevkit_trainval/VOC2007/anno_trainval.txt'
    # image_size       = 448
    # grid_num         = 14
    # flag_augm        = 0
    # train            = True
    
    # YoloDatasetTrain = YoloDataset(dir_annotations, file_annotations
    #                             , train
    #                             , image_size, grid_num
    #                             , flag_augm
    #                             , transform = [transforms.ToTensor()] )
    
    # print (' - Total Images: ', len(YoloDatasetTrain))
    # if (1):
    #     idx = np.random.randint(len(YoloDatasetTrain))
    #     X,Y = YoloDatasetTrain[idx]
    #     YoloDatasetTrain.display(X)
        
    # DataLoaderTrain = DataLoader(YoloDatasetTrain, batch_size=1,shuffle=False,num_workers=0)
    # train_iter = iter(DataLoaderTrain)
    # for i in range(2):
    #     img,target = next(train_iter)
        # print(img,target) 