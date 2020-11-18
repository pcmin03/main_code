import cv2
import skimage
import torch, glob
import numpy as np
#import pydensecrf.densecrf as dcrf
#import pydensecrf.utils as utils

from natsort import natsorted

import torch.nn.functional as F
from torch.utils.data import  Dataset
from torch import nn
from torch import einsum
from torch.autograd import Variable
from sklearn.model_selection import KFold
from scipy import ndimage

from skimage.transform import resize

from skimage.morphology import medial_axis, skeletonize

######################################################################

#---------------------------Gabor loss-------------------------------#

######################################################################
class Custom_Gabor(torch.nn.Module):
    
    def __init__(self,device='gpu',use_median=False,use_label=False):
        super(Custom_Gabor,self).__init__()
        self.use_label = use_label
        if self.use_label == True:
            self.gabor = GaborConv2d(in_channels=1, out_channels=300, kernel_size=51, stride=1,
                    padding=25, dilation=1 , padding_mode='zeros',device=device)
        elif self.use_label == False:
            self.gabor = GaborConv2d(in_channels=1, out_channels=300, kernel_size=51, stride=1,
                    padding=25, dilation=1 , padding_mode='zeros',device=device)
        
        self.use_median = use_median  
        
        if self.use_median == True:
            self.median = MedianPool2d()

    def make_mRMSE(self,feature):
        one_img = torch.ones_like(feature[:,1:2])
        Bimg = one_img- feature[:,1:2]
        Dimg = one_img- feature[:,2:3]
        Cimg = one_img- feature[:,3:4]
        return torch.abs(one_img - (Bimg + Dimg + Cimg))

    def forward(self, input_,label,activation='softmax'):
        # net_output = torch.sum(net_output[:,1:3],dim=1).unsqueeze(1)  

        if self.use_label == True:
            one_img = torch.zeros_like(input_)
            zero_img = torch.ones_like(input_)
            gt = torch.where(input_>0.2,zero_img,one_img) #binary image 
            if self.use_median == True:
                gt = self.median(gt)

            out_feature = self.gabor(gt) 
        
        else:
            if activation == 'sigmoid':
                net_output = self.make_mRMSE(label) #make binary image -> 2 channel
                out_feature = self.gabor(net_output) 
                
            elif activation == 'softmax':
                out_feature = self.gabor(label)

        input_feature = self.gabor(input_)
        return input_feature, out_feature


class Custom_Gabor_loss(torch.nn.Module):
    
    def __init__(self,device='gpu',weight=10,use_median=False,use_label=False):
        super(Custom_Gabor_loss,self).__init__()
        
        self.weight = float(weight)
        self.use_label = use_label
        if self.use_label == True:
            self.gabor = GaborConv2d(in_channels=1, out_channels=300, kernel_size=30, stride=3,
                    padding=0, dilation=1 , padding_mode='zeros',device=device)
        elif self.use_label == False:
            self.gabor = GaborConv2d(in_channels=1, out_channels=300, kernel_size=30, stride=3,
                    padding=0, dilation=1 , padding_mode='zeros',device=device)
            
        self.use_median = use_median  
        
        if self.use_median == True:
            self.median = MedianPool2d()
    def make_mRMSE(self,feature):
        one_img = torch.ones_like(feature[:,1:2])
        Bimg = one_img- feature[:,1:2]
        Dimg = one_img- feature[:,2:3]
        Cimg = one_img- feature[:,3:4]
        return one_img - (Bimg * Dimg * Cimg)

    def forward(self, net_output, gt,label,activation='softmax'):
        # net_output = torch.sum(net_output[:,1:3],dim=1).unsqueeze(1)  
        if self.use_label == True:
            if activation == 'sigmoid':
                out_feature = self.gabor(net_output) 
                gt_feature = self.gabor(label) 
            elif activation == 'softmax':
                one_torch= torch.ones_like(label)
                zero_torch= torch.zeros_like(label)
                back_gt = torch.where(label==0,one_torch,zero_torch).unsqueeze(1)
                body_gt = torch.where(label==1,one_torch,zero_torch).unsqueeze(1)
                dend_gt = torch.where(label==2,one_torch,zero_torch).unsqueeze(1) 
                axon_gt = torch.where(label==3,one_torch,zero_torch).unsqueeze(1)
                new_gt = torch.cat((back_gt, body_gt,dend_gt,axon_gt),dim=1).cuda().float()
                
                out_feature = self.gabor(net_output)
                #make mask image 
                gt_feature = self.gabor(new_gt)
            MAE = torch.abs(out_feature - gt_feature)
        else : 
            one_img = torch.zeros_like(gt)
            zero_img = torch.ones_like(gt)
            gt = torch.where(gt>0.35,zero_img,one_img) #binary image 

            gt = gt - label[:,1:2]

            if self.use_median == True:
                gt = self.median(gt)

            dend_feature = self.gabor(net_output[:,2:3])
            axon_feature = self.gabor(net_output[:,2:3])
            #make mask image 
            out_feature = axon_feature
            gt_feature = self.gabor(gt)
        
            DEMAE = torch.abs(gt_feature - dend_feature )
            AXMAE = torch.abs(gt_feature - axon_feature)

            DEMSE = torch.mul(DEMAE,DEMAE)
            AXMAE = torch.mul(AXMAE,AXMAE)

            MSE = torch.mean(DEMSE + AXMAE)
            
        return [MSE*self.weight,out_feature.float(),gt_feature.float()]

class Custom_Gabor_loss2(torch.nn.Module):
    
    def __init__(self,device='gpu',weight=10,use_median=False):
        super(Custom_Gabor_loss2,self).__init__()
        
        self.weight = float(weight)
        
        self.gabor = GaborConv2d(in_channels=4, out_channels=300, kernel_size=50, stride=2,
                 padding=0, dilation=1 , padding_mode='zeros',device=device)
        self.use_median = use_median  
        if self.use_median == True:
            self.median = MedianPool2d()
    def make_mRMSE(self,feature):
        one_img = torch.ones_like(feature[:,1:2])
        Bimg = one_img- feature[:,1:2]
        Dimg = one_img- feature[:,2:3]
        Cimg = one_img- feature[:,3:4]
        return one_img - (Bimg * Dimg * Cimg)

    def forward(self, net_output, gt,activation='softmax'):
        # net_output = torch.sum(net_output[:,1:3],dim=1).unsqueeze(1)  
        if activation == 'sigmoid':
            out_feature = self.gabor(net_output) 
            gt_feature = self.gabor(gt) 
        elif activation == 'softmax':
            back_gt = torch.where(gt==0,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
            body_gt = torch.where(gt==1,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
            dend_gt = torch.where(gt==2,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1) 
            axon_gt = torch.where(gt==3,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
            new_gt = torch.cat((back_gt, body_gt,dend_gt,axon_gt),dim=1).cuda().float()
            
            out_feature = self.gabor(new_gt)
            #make mask image 
            gt_feature = self.gabor(gt)
        
        # print(out_feature.shape,gt_feature.shape,out_feature)
        MAE = torch.abs(out_feature - gt_feature)
        MSE = torch.mean(torch.mul(out_feature,gt_feature)).float()
        RMSE = torch.sqrt(MSE)
        
        # MSE=torch.abs(out_feature - gt_feature)
        # RMSE = torch.mul(MSE,MSE).float()
        return [MSE*self.weight,out_feature.float(),gt_feature.float()]

class gabor_test(torch.nn.Module):
    
    def __init__(self,in_chan=1,out_chan=473,kernel_size=50,stride=1,padding=0,dilation=1,device='gpu'):
        super(gabor_test,self).__init__()
        self.out_chan = out_chan
        self.gabor = GaborConv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_size, stride=stride,
                 padding=padding, dilation=dilation , padding_mode='zeros',device=device)

    def forward(self, net_output):
        out_feature = self.gabor(net_output)
        return out_feature

