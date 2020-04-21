import numpy as np
import skimage 
import os , sys, argparse, glob
import cv2
import tqdm, json
import matplotlib.pyplot as plt
import torch
# from tqdm import tqdm 
from torch import nn, optim
from torchvision import models ,transforms
from torch.utils.data import DataLoader, Dataset

# from glob import glob
from PIL import Image
from skimage.io import imsave
from natsort import natsorted
import random
# from skimage.util.shape import 
# from sklearn.feature_extraction 
# import image,view_as_blocks


from skimage.filters import threshold_otsu

from sklearn.feature_extraction import image
from skimage.util.shape import view_as_blocks

from torch.autograd import Variable

import torch.nn.functional as F
from my_network import *
from neuron_util import *
from transforms import RandomFlip, RandomRotate90, RandomRotate,ToTensor
# from transform_3d import 
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from metrics import Evaluator

from torch.autograd import Function


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class mydataset_3d(Dataset):
    def __init__(self,imageDir,labelDir,transform,L_transform,size,state=False):
        img = glob.glob(imageDir +'*')
        lab = glob.glob(labelDir +'*')

        self.images = natsorted(img)[0:size]
        self.labels = natsorted(lab)[0:size]

        self.transform = transform
        self.L_transform = L_transform
        self.state = state
        self.Flip = RandomFlip(np.random.RandomState())
        self.Rotate90 = RandomRotate90(np.random.RandomState())
        self.Rotate = RandomRotate(np.random.RandomState(), angle_spectrum=30)
        
        print(len(img),len(lab))
    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        
        image = skimage.io.imread(self.images[index])
        labels = skimage.io.imread(self.labels[index])

        length=len(image)//4
        
        zero = np.zeros_like(image[0])
        zero = np.expand_dims(zero, axis=0)

        lab_zero = np.zeros_like(labels[0])
        lab_zero = np.expand_dims(lab_zero, axis=0)
        while len(image) < 25:
            image = np.concatenate((image, zero),axis=0)
            labels = np.concatenate((labels, lab_zero),axis=0)
            
        
        stack_label = []
        for label in labels:
            gray_label = skimage.color.rgb2gray(label)

            mask0= gray_label < 0.3
            mask1= gray_label > 0.9
            mask2= np.logical_and(0.3 < gray_label , gray_label < 0.4)
            mask3=  np.logical_and(0.4 < gray_label , gray_label < 0.8)

            gray_label[mask0] = 0
            gray_label[mask1] = 1
            gray_label[mask2] = 2
            gray_label[mask3] = 3
            stack_label.append(gray_label)
        labels=np.array(stack_label).astype('uint8')
        
        # clip = torch.Tensor([self.L_transform(img) for img in image])
        # image = skimage.color.rgb2gray(image[:25])
        
        if self.state==True:
            image,labels = self.Flip(m = image,n = labels)
            image,labels = self.Rotate90(m = image,n = labels)
            image,labels = self.Rotate(m = image,n = labels)
            #  = self.transform()
            # # print(image.shape)
            # skimage.io.imsave('../pre_result_unet_loss_single2_3d_vnet_batch_test/image'+str(index)+'tif',image[12])
            # skimage.io.imsave('../pre_result_unet_loss_single2_3d_vnet_batch_test/labels'+str(index)+'tif',labels[12])
           
        clip = self.L_transform(image)

        return clip,labels

imageDir='./3d_data/3d_data_orginal_train/'
labelDir = './3d_data/3d_data_train_label/'

testDir ='./3d_data/3d_data_orginal_test2/'
tlabelDir = './3d_data/3d_data_test_label2/'


transform = transforms.Compose([transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
t_transform = transforms.Compose([RandomFlip(np.random),
                                RandomRotate90(np.random),
                                RandomRotate(np.random)])
L_transform = transforms.Compose([ToTensor()])

train_db=mydataset_3d(imageDir,labelDir,t_transform,L_transform,200,True)
test_db=mydataset_3d(testDir,tlabelDir,t_transform,L_transform,60,True)

Datatset = {'train' :  DataLoader(train_db,4, 
                          shuffle = True,
                          num_workers = 8),
            'test' : DataLoader(test_db,4, 
                          shuffle = True,
                          num_workers = 8)}


cuda0 = torch.device('cuda:0'if torch.cuda.is_available() else "else")


gen = VNet().cuda()


optimizerG = optim.Adam(gen.parameters(),lr=1e-4)    
scheduler = optim.lr_scheduler.StepLR(optimizerG,100)

#training
import shutil
path = '../pre_result_unet_loss_single2_3d_vnet_batch_test/'
print('----- Make_save_Dir-------------')
if not os.path.exists(path):
    os.makedirs(path )
# else:
#     print('----- remove_Dir-------------')
#     shutil.rmtree(path,ignore_errors=True)
#     print('----- Make_save_Dir-------------')
    # os.makedirs(path)

if os.path.exists(path+"bestsave_models{}.pth"):
    checkpoint = torch.load(path +"bestsave_models{}.pth")
    gen.load_state_dict(checkpoint['gen_model'])

gen = torch.nn.DataParallel(gen, device_ids=[0,1])

writer = SummaryWriter(path+'board/')
print("loading.......dataset")

#set loss
softm = torch.nn.LogSoftmax(1)
criterion = Custom_WeightedCrossEntropyLossV2().cuda()

evaluator_body = Evaluator(4)

epochs = 2000
evaluator_body.reset()

seg_loss_dend = 0
seg_loss_axon = 0
size =24
best = 0

print("start trainning!!!!")
for epoch in range(epochs):
    running_loss = 0
    TIOU = 0
    tdloss = 0
    scheduler.step()

    if epoch %50 == 0:
        phase = 'test'
        gen.train()
    else : 
        phase = 'train'
        gen.eval()

    print('*** learning_riate===>{:.4f}'.format(get_lr(optimizerG)))

    for i, batch in enumerate(tqdm.tqdm(Datatset[phase])):
        
        _input, _label = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
        
        _input = _input[:,:,0:24]
        _label = _label[:,0:24]
        # print(_input.shape)
        
        
        if phase == 'train':
            predict=gen(_input)
            optimizerG.zero_grad()
            seg_loss_body = criterion(predict,_label)
            seg_loss_body.backward(retain_graph = True)
            optimizerG.step()

        else:
            with torch.no_grad():
                predict=gen(_input)
                
                val_loss = criterion(predict,_label)

    if phase == 'test':
        pre_IOU = 0
        pre_ACC = 0
        print(_input.shape)
        for i in range(len(_input[0,0,:])):
            tlab = _label[:,i]
            predict_bod = predict[:,:,i]
            evaluator_body.add_batch(tlab.cpu().numpy(),torch.argmax(predict_bod,dim=1).cpu().numpy())
            IOU,Class_IOU,wo_back_MIoU = evaluator_body.Mean_Intersection_over_Union()
            Acc_class,Class_ACC,wo_back_ACC = evaluator_body.Pixel_Accuracy_Class()
            
            pre_IOU += Class_IOU
            pre_ACC += Class_ACC
            if i == 12:
                middle_IOU = Class_IOU
                middle_ACC = Class_ACC
                
        pre_IOU = [number / 24 for number in pre_IOU]
        pre_ACC = [number / 24 for number in pre_ACC]

    print("======{}/{}epochs======".format(epoch,epochs))
    if phase == 'train':
        print("================trainning=====================")    
        print("===> seg_loss_body: %.4f" % (seg_loss_body))
        writer.add_scalar('seg_loss_body',seg_loss_body,epoch)
    else: 
        print("================testing=====================")
        print("===> Class_IOU:" ,(pre_IOU))
        print("===> Class_ACC:" ,(pre_ACC))
        print("===> middle_IOU:" ,(middle_IOU))
        print("===> middle_ACC:" ,(middle_ACC))
        print("===> wo_back_MIoU: %.4f" % (wo_back_MIoU))
        print("===> wo_back_ACC: %.4f" % (wo_back_ACC))
        print("===> val_loss: %.4f" % (val_loss))
        writer.add_scalar('IOU_1',pre_IOU[1],epoch)
        writer.add_scalar('IOU_2',pre_IOU[2],epoch)
        writer.add_scalar('IOU_3',pre_IOU[3],epoch)
        writer.add_scalar('val_loss',val_loss,epoch)
        print("================testing=====================")
        
        if  pre_IOU[3] > best:
            torch.save({"gen_model":gen.state_dict(),
                        "optimizerG":optimizerG.state_dict(),
                        "epochs":epoch},
                        path+"bestsave_models{}.pth")
            print("update best!!!")
            best = pre_IOU[3]
            best_epoch = epoch
            
        torch.save({"gen_model":gen.state_dict(),
                    "optimizerG":optimizerG.state_dict(),
                    "epochs":epoch},
                    path+"lastsave_models{}.pth")

        pre_body=decode_segmap(ch_channel(predict),name='full')

        v_la=decode_segmap(_label.cpu().detach().numpy(),name='full')
        
        skimage.io.imsave(path+"pre_body"+"_"+str(epoch)+".tif",np.transpose(pre_body[0],[1,2,3,0]))    
        skimage.io.imsave(path+"labe_"+"_"+str(epoch)+".tif",np.transpose(v_la[0],[1,2,3,0]))
        skimage.io.imsave(path+"img"+"_"+str(epoch)+".tif",np.transpose(_input[0].detach().cpu().numpy(),[1,2,3,0]))



