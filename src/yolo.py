import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from nets import vgg16, vgg16_bn
from resnet_yolo import resnet50, resnet18
from yoloLoss import yoloLoss
from dataset import yoloDataset
from visualize import Visualizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_gpu = torch.cuda.is_available()



net 		   = vgg16_bn()
vgg            = models.vgg16_bn(pretrained=True)
new_state_dict = vgg.state_dict()
dd             = net.state_dict()
for k in new_state_dict.keys():
	print(k)
	if k in dd.keys() and k.startswith('features'):
		print('yes')
		dd[k] = new_state_dict[k]
net.load_state_dict(dd)
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

criterion = yoloLoss(7,2,5,0.5)
if use_gpu:
    net.cuda ()

net.train()

# different learning rate
params=[]
params_dict = dict(net.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)

# train_dataset = yoloDataset(root=file_root,list_file=['voc12_trainval.txt','voc07_trainval.txt'],train=True,transform = [transforms.ToTensor()] )
train_dataset = yoloDataset(root=file_root,list_file=['voc2012.txt','voc2007.txt'],train=True,transform = [transforms.ToTensor()] )
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
# test_dataset = yoloDataset(root=file_root,list_file='voc07_test.txt',train=False,transform = [transforms.ToTensor()] )
test_dataset = yoloDataset(root=file_root,list_file='voc2007test.txt',train=False,transform = [transforms.ToTensor()] )
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))
logfile = open('log.txt', 'w')

num_iter =  0
vis = Visualizer(env='xiong')
best_test_loss = np.inf

def myNet():
	pass

def train():
	pass

if __name__ == ""__main__":

	file_root = '/home/xzh/data/VOCdevkit/VOC2012/allimgs/'
	learning_rate = 0.001
	num_epochs =  50
	batch_size = 24

	for epoch in range(num_epochs):
		net.train()
		if epoch == 30:
			learning_rate=0.0001
		if epoch == 40:
			learning_rate=0.00001
		for param_group in optimizer.param_groups:
			param_group['lr'] = learning_rate
		
		print ( ' \ n \ n Starting epoch % d / % d '  % (epoch +  1 , num_epochs))
		print('Learning Rate for this epoch: {}'.format(learning_rate))
		
		total_loss = 0.
		
		
	for i,(images,target) in enumerate(train_loader):
			images = Variable(images)
			target = Variable(target)
			if use_gpu:
				images,target = images.cuda(),target.cuda()
			
			pred = net(images)
			loss = criterion(pred,target)
			total_loss += loss.data[0]
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if (i+1) % 5 == 0:
				print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
				%(epoch+1, num_epochs, i+1, len(train_loader), loss.data[0], total_loss / (i+1)))
				num_iter +  1
				vis.plot_train_val(loss_train=total_loss/(i+1))