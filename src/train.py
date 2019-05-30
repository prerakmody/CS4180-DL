import tqdm
import time
import random
import math
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

import src.dataloader as dataloader
from src.nets2_utils import *
from src.nets import *
from src.predict import *


## --------------------------------------- YOLOV2 --------------------------------------- ##
class YOLOv2Train():

    def __init__(self):
        self.model       = ''
        self.optimizer   = ''

        self.trainlist   = ''
        self.testlist    = ''

        self.init_width        = ''
        self.init_height       = ''
        self.batch_size        = ''

    def train(self, PASCAL_DIR, PASCAL_TRAIN, PASCAL_VALID, TRAIN_LOGDIR, VAL_LOGDIR, VAL_OUTPUTDIR_PKL, VAL_PREFIX
                    , MODEL_CFG, MODEL_WEIGHT
                    , BATCH_SIZE
                    , LOGGER=''):
        net_options   = parse_cfg(MODEL_CFG)[0]
        self.trainlist     = PASCAL_TRAIN
        self.testlist      = PASCAL_VALID
        backupdir     = TRAIN_LOGDIR
        nsamples      = file_lines(self.trainlist)
        gpus          = "1"
        ngpus         = 1
        num_workers   = 4
        cfgfile       = MODEL_CFG
        weightfile    = MODEL_WEIGHT

        # model/net configuration
        self.batch_size    = BATCH_SIZE #int(net_options['batch'])
        max_batches   = int(net_options['max_batches'])
        learning_rate = float(net_options['learning_rate'])
        momentum      = float(net_options['momentum'])
        decay         = float(net_options['decay'])
        steps         = [float(step) for step in net_options['steps'].split(',')]
        scales        = [float(scale) for scale in net_options['scales'].split(',')]

        #Train parameters
        max_epochs    = int(max_batches*self.batch_size/nsamples+1)
        self.use_cuda      = True
        seed          = int(time.time())
        eps           = 1e-5
        save_interval = 10  # epoches
        dot_interval  = 70  # batches

        # Test parameters
        conf_thresh   = 0.25
        nms_thresh    = 0.4
        iou_thresh    = 0.5

        if not os.path.exists(backupdir):
            print (' - backupdir :', backupdir)
            os.mkdir(backupdir)

        ###############
        torch.manual_seed(seed)
        if self.use_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            torch.cuda.manual_seed(seed)

        self.model       = Darknet(cfgfile)
        region_loss      = self.model.loss

        self.model.load_weights(weightfile)
        # model.print_network()

        region_loss.seen  = self.model.seen
        processed_batches = int(self.model.seen/self.batch_size)

        self.init_width        = self.model.width
        self.init_height       = self.model.height
        init_epoch        = int(self.model.seen/nsamples) 

        kwargs = {'num_workers': num_workers, 'pin_memory': True} if self.use_cuda else {}
        

        if self.use_cuda:
            if ngpus > 1:
                self.model = torch.nn.DataParallel(self.model).cuda()
            else:
                self.model = self.model.cuda()

        params_dict = dict(self.model.named_parameters())
        params = []
        for key, value in params_dict.items():
            if key.find('.bn') >= 0 or key.find('.bias') >= 0:
                params += [{'params': [value], 'weight_decay': 0.0}]
            else:
                params += [{'params': [value], 'weight_decay': decay*self.batch_size}]
        optimizer = optim.SGD(self.model.parameters(), 
                                lr=learning_rate/self.batch_size, momentum=momentum,
                                dampening=0, weight_decay=decay*self.batch_size)
        
        print ('')
        print (' -- init_epoch : ', init_epoch)
        print (' -- max_epochs : ', max_epochs)

        for epoch in range(init_epoch, max_epochs): 
            
            print (' ---------------------------- EPOCH : ', epoch, ' ---------------------------------- ')
            ## ----------------------- TRAIN ------------------------
            # global processed_batches
            t0 = time.time()
            if ngpus > 1:
                cur_model = self.model.module
            else:
                cur_model = self.model
            
            train_loader = torch.utils.data.DataLoader(
                dataloader.VOCDatasetv2(self.trainlist, shape=(self.init_width, self.init_height),
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]),
                            train=True,
                            seen=cur_model.seen,
                            batch_size=self.batch_size),
                batch_size=self.batch_size, shuffle=False, **kwargs)               

            lr = self.adjust_learning_rate(optimizer, processed_batches, learning_rate, steps, scales, self.batch_size)
            self.model.train()
            train_loss_total       = 0.0
            with tqdm.tqdm_notebook(total = len(train_loader)*self.batch_size) as pbar:
                for batch_idx, (data, target) in enumerate(train_loader):
                    
                    if (1):
                        if (batch_idx > 1):
                            break

                        pbar.update(self.batch_size)
                        if (batch_idx == 0):
                            print ('  - [INFO] data (or X) : ', data.shape, ' || type : ', data.dtype)       # = torch.Size([1, 3, 416, 416]) torch.float32
                            print ('  - [INFO] target (or Y) : ', target.shape, ' || type : ', target.dtype) # = torch.Size([1, 250]) torch.float64
                            print ('  - [INFO] Total train points : ', len(train_loader), ' || nsamples : ', nsamples)
                    
                    if (1):
                        self.adjust_learning_rate(optimizer, processed_batches, learning_rate, steps, scales, self.batch_size)
                        processed_batches = processed_batches + 1
                    
                    if (1):
                        if self.use_cuda:
                            data   = data.cuda()
                            target = target.float() 
                            # target= target.cuda()
                        data, target = Variable(data), Variable(target)
                    
                    if (1):
                        optimizer.zero_grad()
                        output           = self.model(data)
                        region_loss.seen = region_loss.seen + data.data.size(0)
                        loss             = region_loss(output, target)
                        train_loss_total += loss.data;
                        loss.backward()
                        optimizer.step()
                        print (' - loss : ', loss, ' || loss.data : ', loss.data)

                    if (1):
                        pass
                        # print (' - region_loss.seen : ', region_loss.seen)
                        # print (' - [DEBUG] data : ', data.shape, data.dtype)
                        # print (' - [DEBUG] output : ', output.shape, output.dtype)
                        # print (' - [DEBUG] target : ', target.shape, target.dtype)
                        # print (' - [DEBUG] output : ',output)
                        # print ('-----------------------------------')
                        # print (' - [DEBUG] target : ',target)

            if LOGGER != '':
                train_loss_avg = train_loss_total / len(train_loader)
                print ('   -- train_loss_total : ', train_loss_total, ' || train_loss_avg :', train_loss_avg)
                LOGGER.save_value('Total Loss', 'Train Loss', epoch+1, train_loss_avg)
            t1 = time.time()
            # logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
            # if (epoch+1) % save_interval == 0:
            logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
            cur_model.seen = (epoch + 1) * len(train_loader.dataset)
            cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))

            ## ----------------------- TEST ------------------------
            # self.test(epoch)
            print (' -- Loss : ', self.model.loss)
            valObj = PASCALVOCEval(self.model, MODEL_CFG, MODEL_WEIGHT
                                        , PASCAL_DIR, PASCAL_VALID, VAL_LOGDIR, VAL_PREFIX, VAL_OUTPUTDIR_PKL
                                        , LOGGER, epoch)
            valObj.predict(BATCH_SIZE)
        # end for epoch

    def adjust_learning_rate(self, optimizer, batch, learning_rate, steps, scales, batch_size):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = learning_rate
        for i in range(len(steps)):
            scale = scales[i] if i < len(scales) else 1
            if batch >= steps[i]:
                lr = lr * scale
                if batch == steps[i]:
                    break
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr/batch_size
        return lr
