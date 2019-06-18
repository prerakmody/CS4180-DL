import pdb
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

runtime = 'online' # ['local', 'online']

if (runtime == 'online'):
    # print (' - [train.py] Online Runtime')
    import src.dataloader as dataloader
    from src.nets2_utils import *
    if (1):
        from src.predict import *
        from src.nets import *
        from src.pruning.weightPruning.utils import prune_rate, are_masks_consistent
        from src.pruning.weightPruning.methods import filter_prune, quick_filter_prune_v2, quick_filter_prune_v1

elif runtime == 'local':
    import dataloader as dataloader
    from nets2_utils import *
    from predict import *
    from nets import *


def parse_cfg(cfgfile, verbose=0):
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
                    , BATCH_SIZE, LEARNING_RATES, MAX_EPOCHS
                    , LOGGER='', DEBUG_EPOCHS=-1, verbose=0,pruning_perc=0.,pruning_method="weight"):

        # Step1 - Model Config        
        if (1):
            self.use_cuda = True
            cfgfile       = MODEL_CFG   
            weightfile    = MODEL_WEIGHT
            net_options   = parse_cfg(MODEL_CFG)[0]
            self.model    = Darknet(cfgfile)
            if self.use_cuda:
                self.model = self.model.cuda()
            self.model.load_weights(weightfile)
            # model.print_network()
            
        # Step2 - Dataset
        if (1):
            self.trainlist = PASCAL_TRAIN
            self.testlist  = PASCAL_VALID
            backupdir      = TRAIN_LOGDIR
            nsamples       = file_lines(self.trainlist)
            num_workers   = 4
            self.init_width        = self.model.width
            self.init_height       = self.model.height
            if not os.path.exists(backupdir):
                print (' - backupdir :', backupdir)
                os.mkdir(backupdir)
            img_backupdir = os.path.join(backupdir, 'grad_flow')
            if not os.path.exists(img_backupdir):
                print (' - img_backupdir :', img_backupdir)
                os.mkdir(img_backupdir)
            kwargs = {'num_workers': num_workers, 'pin_memory': True} if self.use_cuda else {}
        
        # Step3 - Training Params    
        if (1):
            self.batch_size  = BATCH_SIZE #int(net_options['batch'])
            max_epochs       = MAX_EPOCHS
            LRs              = LEARNING_RATES  #LR [0.00001, 0.0001] # 0.00025

            momentum        = float(net_options['momentum'])
            decay           = float(net_options['decay'])
            
            # max_batches     = MAX_BATCHES
            # steps           = [float(step) for step in net_options['steps'].split(',')]
            # scales          = [float(scale) for scale in net_options['scales'].split(',')]
            # eps             = 1e-5
            seed            = 42  #int(time.time())
            
            
            torch.manual_seed(seed)
            if self.use_cuda:
                torch.cuda.manual_seed(seed)
            # [TODO]
            # torch.backends.cudnn.deterministic=True
            
            region_loss       = self.model.loss
            region_loss.seen  = self.model.seen
            processed_batches = int(self.model.seen/self.batch_size)
            init_epoch        = int(self.model.seen/nsamples)
        
        # Step3.2 - Optimizer - 
        if (1):
            # params_dict = dict(self.model.named_parameters())
            # params = []
            # for key, value in params_dict.items():
            #     if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            #         params += [{'params': [value], 'weight_decay': 0.0}]
            #     else:
            #         params += [{'params': [value], 'weight_decay': decay*self.batch_size}]
            # optimizer = optim.SGD(self.model.parameters(), 
            #                         lr=learning_rate/self.batch_size, momentum=momentum,
            #                         dampening=0, weight_decay=decay*self.batch_size)
            
            optimizer = optim.SGD(self.model.parameters(), 
                                    lr=LRs[0], momentum=momentum,
                                    dampening=0, weight_decay=decay*self.batch_size)

        # Step4 - Model Saving
        if (1): 
            SAVE_INTERNAL = 10  # epoches
            dot_interval  = 70  # batches

        # Step5 -  Test parameters
        if (1):
            conf_thresh   = 0.25 # [currently default to 0.05]
            nms_thresh    = 0.4  # [currently default to 0.45]
            iou_thresh    = 0.5  # [currently default to 0.5]

        # Step 99 - Random Printing
        if (1):
            print ('')
            #print (' -- init_epoch : ', init_epoch)
            #print (' -- max_epochs : ', max_epochs)

        # Step 100 - PRUNE THAT THING...
        """
         - weight pruning : here we prune x% of all neurons on the basis of their abs value
         - filter pruning : here we prune x% of all filters in every layer of the model on the 
                               basis of their L2 norm
        """
        if (1):
            if pruning_perc > 0:
                print ('')
                print ('  -- [DEBUG][pruning] Finding things to prune .... ')
                if pruning_method == "filter":
                    if (1):
                        print ('  -- [DEBUG][pruning] quick_filter_prune_v2() : ')
                        # masks = quick_filter_prune_v2(self.model, pruning_perc, min_conv_id = 1, max_conv_id =12, verbose=0)
                        # masks = quick_filter_prune_v2(self.model, pruning_perc, min_conv_id = 14, max_conv_id =19, verbose=0)
                        masks = quick_filter_prune_v2(self.model, pruning_perc, min_conv_id = 1, max_conv_id =19, verbose=0)
                    else:
                        print ('  -- [DEBUG][pruning] quick_filter_prune_v1() : ')
                        masks = quick_filter_prune_v1(self.model, pruning_perc, verbose=0)
                        # masks = filter_prune(self.model, pruning_perc)

                elif pruning_method == 'weight':
                    masks = weight_prune(self.model, pruning_perc)
                
                self.model.set_masks(masks)
                
                if pruning_method == 'filter':
                    p_rate_filter = prune_rate(self.model, 'filter',True)
                    p_rate_weight = prune_rate(self.model, 'weight',True)
                    print('  -- [DEBUG][pruning] %s=pruned: %s' % ('filter', round(p_rate_filter,5)))
                    print('  -- [DEBUG][pruning] %s=pruned: %s' % ('weight', round(p_rate_weight,5)))
                elif pruning_method == 'weight':
                    p_rate_weight = prune_rate(self.model,pruning_method,True)
                    print('  -- [DEBUG][pruning] %s=pruned: %s' % (pruning_method, round(p_rate,5)))

                ## ----------------------- VALIDATE ------------------------ (check the new mAP after pruning)
                if (0):
                    valObj = PASCALVOCEval(self.model, MODEL_CFG, MODEL_WEIGHT, region_loss
                                                , PASCAL_DIR, PASCAL_VALID, VAL_LOGDIR, VAL_PREFIX, VAL_OUTPUTDIR_PKL
                                                , LOGGER, LOGGER_EPOCH=0)
                    valObj.predict(BATCH_SIZE)

        for epoch in range(init_epoch, max_epochs):
            
            if (1):
                if epoch > 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = LRs[epoch]
                # LR = 0.00001
                #lr = self.adjust_learning_rate(optimizer, processed_batches, learning_rate, steps, scales, self.batch_size)
                # lr = 0.00001

            if (1):
                train_loader = torch.utils.data.DataLoader(
                    dataloader.VOCDatasetv2(PASCAL_TRAIN, shape=(self.init_width, self.init_height),
                                shuffle=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]),
                                train=True,
                                seen=self.model.seen),
                    batch_size=self.batch_size, shuffle=False, **kwargs)      


            print (' ---------------------------- EPOCH : ', epoch, ' (LR : ',LRs[epoch],') ---------------------------------- ')
                
            self.model.train()
            with tqdm.tqdm_notebook(total = len(train_loader)*self.batch_size) as pbar:
                train_loss_total       = 0.0
                ave_grads = {n: 0 for n,p in self.model.named_parameters() if p.requires_grad and "bias" not in n}

                for batch_idx, (data, target) in enumerate(train_loader):
                    if (1):
                        if (DEBUG_EPOCHS > -1 and epoch == 0):
                            if batch_idx > DEBUG_EPOCHS:
                                break

                        pbar.update(self.batch_size)
                        if (batch_idx == 0 and epoch == 0):
                            print ('  - [INFO] data (or X) : ', data.shape, ' || type : ', data.dtype)       # = torch.Size([1, 3, 416, 416]) torch.float32
                            print ('  - [INFO] target (or Y) : ', target.shape, ' || type : ', target.dtype) # = torch.Size([1, 250]) torch.float64
                            print ('  - [INFO] Total train points : ', len(train_loader), ' || nsamples : ', nsamples)
                    
                    # if (1):
                        #self.adjust_learning_rate(optimizer, processed_batches, learning_rate, steps, scales, self.batch_size)
                        #processed_batches = processed_batches + 1
                    
                    if (1):
                        if self.use_cuda:
                            data   = data.cuda()
                            target = target.float() 
                            # target= target.cuda()
                        data, target = Variable(data), Variable(target)

                    if (1):
                        try:
                            with torch.autograd.detect_anomaly():
                                output           = self.model(data)
                                if (output != output).any():
                                    print ('  -- [DEBUG][train.py] We have some NaNs')
                                    pdb.set_trace()
                                # print ((output != output).any())
                                region_loss.seen = region_loss.seen + data.data.size(0)
                                train_loss       = region_loss(output, target)
                                train_loss_total += train_loss.data
                                    
                                optimizer.zero_grad()
                                train_loss.backward()
                                optimizer.step()

                                for n, p in self.model.named_parameters():
                                    if(p.requires_grad) and ("bias" not in n):
                                        ave_grads[n] += p.grad.abs().mean()

                                # pdb.set_trace()

                                if verbose:
                                    print (' - loss : ', train_loss)

                            # for name, param in self.model.named_parameters():
                            #     if param.requires_grad:
                            #         print ('  -- [DEBUG] : ', name, '\t  - \t', round(param.grad.data.sum().item(),3), '   [',param.shape,']')
                        except:
                            traceback.print_exc()
                            pdb.set_trace()

            if pruning_perc > 0:
                print('  -- [DEBUG][pruning] pruned: %s' % prune_rate(self.model,False))
                print('  -- [DEBUG][pruning] pruned weights consistent after retraining: %s ' % are_masks_consistent(self.model, masks, debug=0))
            if (epoch + 1) % 5 == 0:
                logging('save weights to %s/%s-pruned-%s-retrained_%06d.weights' % (backupdir, pruning_method, pruning_perc , epoch+1))
                self.model.save_weights('%s/%s-pruned-%s-retrained_%06d.weights' % (backupdir, pruning_method, pruning_perc, epoch+1))
            if (1):
                train_loss_avg = train_loss_total / len(train_loader)
                print ('   -- train_loss_total : ', train_loss_total, ' || train_loss_avg :', train_loss_avg)
            if LOGGER != '':
                LOGGER.save_value('Total Loss', 'Train Loss', epoch+1, train_loss_avg)
                LOGGER.save_value('Learning Rate', 'Learning Rate', epoch+1, LRs[epoch])
                train_loss_total       = 0.0

            self.model.seen = (epoch + 1) * len(train_loader.dataset)

            if (1):
                
                if (1):
                    # epoch_means = [ave_grads[n]/(batch_idx+1) for n in ave_grads]
                    layers      = [layer_name for layer_name in ave_grads if 'conv' in layer_name]
                    epoch_means = [ave_grads[layer_name]/(batch_idx+1) for layer_name in ave_grads if 'conv' in layer_name]
                    f,axarr     = plt.subplots(1, figsize=(15,15))
                    plt.plot(epoch_means, alpha=0.3, color="b")
                    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
                    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
                    plt.xlim(0, len(ave_grads))
                    plt.ylim(0, 0.06)
                    plt.xlabel("Layers")
                    plt.ylabel("Avg Gradient Flow")
                    plt.title("Gradient flow - {0}".format(pruning_method))
                    plt.grid(True)
                    # plt.show()
                    plt.savefig("%s/grad_flow-%s-pruned-%s_%06d.jpg" % (img_backupdir, pruning_method, pruning_perc, epoch+1))
                    plt.close(f)

                if (1):
                    layers      = [layer_name for layer_name in ave_grads if 'bn' in layer_name]
                    epoch_means = [ave_grads[layer_name]/(batch_idx+1) for layer_name in ave_grads if 'bn' in layer_name]
                    f,axarr     = plt.subplots(1, figsize=(15,15))
                    plt.plot(epoch_means, alpha=0.3, color="b")
                    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
                    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
                    plt.xlim(0, len(ave_grads))
                    plt.ylim(0, 0.06)
                    plt.xlabel("Layers")
                    plt.ylabel("Avg Gradient Flow")
                    plt.title("Gradient flow - {0}".format(pruning_method))
                    plt.grid(True)
                    # plt.show()
                    plt.savefig("%s/grad_flow-%s-pruned-%s_%06d.jpg" % (img_backupdir, pruning_method, pruning_perc, epoch+1))
                    plt.close(f)


            ## ----------------------- VALIDATE ------------------------
            valObj = PASCALVOCEval(self.model, MODEL_CFG, MODEL_WEIGHT, region_loss
                                        , PASCAL_DIR, PASCAL_VALID, VAL_LOGDIR, VAL_PREFIX, VAL_OUTPUTDIR_PKL
                                        , LOGGER, epoch)
            valObj.predict(BATCH_SIZE)

        if (1):
            logging('save weights to %s/%s-pruned-%s-retrained-final_%06d.weights' % (backupdir, pruning_method, pruning_perc , epoch+1))
            self.model.save_weights('%s/%s-pruned-%s-retrained-final_%06d.weights' % (backupdir, pruning_method, pruning_perc, epoch+1))
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

# if __name__ == "__main__":
#     torch.cuda.empty_cache()
    
#     if (torch.cuda.is_available()):
#         if (1):
#             DIR_MAIN         = os.path.abspath('../')
#             print (' - 1. DIR_MAIN :  ', DIR_MAIN)

#         if (1):
#             PASCAL_DIR   = os.path.join(DIR_MAIN, 'data/dataset/VOCdevkit/')
#             PASCAL_TRAIN = os.path.join(DIR_MAIN, 'data/dataset/VOCdevkit/voc_train.txt')
#             PASCAL_VALID = os.path.join(DIR_MAIN, 'data/dataset/VOCdevkit/2007_test.txt')
#             TRAIN_LOGDIR = os.path.join(DIR_MAIN, 'train_data')
#             VAL_LOGDIR   = os.path.join(DIR_MAIN, 'eval_data')
#             VAL_OUTPUTDIR_PKL = os.path.join(DIR_MAIN, 'eval_results')
#             MODEL_CFG    = os.path.join(DIR_MAIN, 'data/cfg/github_pjreddie/yolov2-voc.cfg')
#             MODEL_WEIGHT = os.path.join(DIR_MAIN, 'data/weights/github_pjreddie/yolov2-voc.weights')
#             print (' - 2. MODEL_WEIGHT :  ', MODEL_WEIGHT)

#         if (1):
#             VAL_PREFIX   = 'pretrained'
#             BATCH_SIZE   = 1;
#             print (' - 3. VAL_PREFIX : ', VAL_PREFIX)

#         if (1):
#             LOGGER = ''
#             print (' - 4. Logger : ', LOGGER)


#         if (1):
#             trainObj = YOLOv2Train()
#             trainObj.train(PASCAL_DIR, PASCAL_TRAIN, PASCAL_VALID, TRAIN_LOGDIR, VAL_LOGDIR, VAL_OUTPUTDIR_PKL, VAL_PREFIX
#                         , MODEL_CFG, MODEL_WEIGHT
#                         , BATCH_SIZE
#                         , LOGGER)
#     else:
#         print (' - GPU Issues!!')