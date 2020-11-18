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

#-------------------------------RMSE---------------------------------#

######################################################################
class Custom_Adaptive_gausian_DistanceMap(torch.nn.Module):
    
    def __init__(self,weight,distanace_map=False,select_MAE='RMSE',treshold_value=0.35,back_filter=False,premask=True):
        super(Custom_Adaptive_gausian_DistanceMap,self).__init__()
        self.weight = weight
        self.dis_map = distanace_map
        self.select_MAE = select_MAE
        self.treshold_value = treshold_value
        self.back_filter = back_filter
        self.premask = premask
        self.MSE = nn.MSELoss()

    def gaussian_fn(self,predict,label,labmda,channel): 

        i = channel
        gau_numer = torch.pow(torch.abs(predict[:,i:i+1]-label[:,i:i+1]),2)
        
        gau_deno = 1
        ch_gausian = torch.exp(-1*float(self.weight)*(gau_numer))
        if channel == 0: 
            ch_one  = ((label[:,i:i+1])*(ch_gausian)).float()
            ch_zero = (1-label[:,i:i+1]).float()
        else : 
            ch_one  = ((1-label[:,i:i+1])*(ch_gausian)).float()
            ch_zero = (label[:,i:i+1]).float()
        return ch_one,ch_zero

    def forward(self, net_output, gt,mask_inputs):
        
        if gt.dim() == 3:
            back_gt = torch.where(gt==0,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
            body_gt = torch.where(gt==1,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
            dend_gt = torch.where(gt==2,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1) 
            axon_gt = torch.where(gt==3,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
            
            back_output = net_output[:,0:1,:,:]
            new_gt = torch.cat((back_gt, body_gt,dend_gt,axon_gt),dim=1).cuda().float()

        elif gt.dim() == 4 or gt.dim() == 5:
            # postive predict label
            new_gt = gt
            if self.premask == True:
                back_gt = torch.abs((1-mask_inputs))
                gt[:,0:1] = back_gt
            else:
                if self.back_filter == True:
                    zero_img = torch.zeros_like(mask_inputs)
                    one_img = torch.ones_like(mask_inputs)
                    mask_img = torch.where(mask_inputs>self.treshold_value,one_img,zero_img)
                    back_gt = torch.where(mask_inputs>self.treshold_value,zero_img,one_img)
                    # print(new_gt.shape,'self.back-filter')
                    new_gt[:,0:1] = mask_inputs

            back_one,back_zero = self.gaussian_fn(net_output,new_gt,1,0)
            body_one,body_zero = self.gaussian_fn(net_output,new_gt,1,1)
            dend_one,dend_zero = self.gaussian_fn(net_output,new_gt,1,2)
            axon_one,axon_zero = self.gaussian_fn(net_output,new_gt,1,3)
        
        MAE = torch.abs(net_output - new_gt) #L1 loss
        MSE = torch.mul(MAE,MAE).float() 

        # BEMAE,BOMAE,DEMAE,AXMAE = MAE[:,0:1],MAE[:,1:2],MAE[:,2:3],MAE[:,3:4]
        # BEMSE,BOMSE,DEMSE,AXMSE = MSE[:,0:1],MSE[:,1:2],MSE[:,2:3],MSE[:,3:4]

        BEMAE = torch.abs(net_output[:,0:1] - new_gt[:,0:1])
        BOMAE = torch.abs(net_output[:,1:2] - new_gt[:,1:2])
        DEMAE = torch.abs(net_output[:,2:3] - new_gt[:,2:3])
        AXMAE = torch.abs(net_output[:,3:4] - new_gt[:,3:4])
        
        BEMSE = torch.mul(BEMAE,BEMAE).float()
        BOMSE = torch.mul(BOMAE,BOMAE).float()
        DEMSE = torch.mul(DEMAE,DEMAE).float()
        AXMSE = torch.mul(AXMAE,AXMAE).float()

        BEloss = (back_one + back_zero) * BEMSE
        BOloss = (body_one + body_zero) * BOMSE
        DEloss = (dend_one + dend_zero) * DEMSE
        AXloss = (axon_one + axon_zero) * AXMSE

        if self.select_MAE == 'MAE':
            return torch.mean(MAE).float()
        elif self.select_MAE == 'MSE': 
            return torch.mean(MSE).float()

        elif self.select_MAE == 'RMSE' or self.select_MAE == 'SIGRMSE' or self.select_MAE == 'SIGMAE':

            if self.select_MAE == 'SIGRMSE':
                return torch.mean(BEMSE+BOMSE+DEloss+AXloss).float()
            elif self.select_MAE == 'SIGMAE':
                BEloss = (back_one + back_zero) * BEMAE
                BOloss = (body_one + body_zero) * BOMAE
                DEloss = (dend_one + dend_zero) * DEMAE
                AXloss = (axon_one + axon_zero) * AXMAE
                return torch.mean(BEMAE+BOMAE+DEloss+AXloss).float()
        
######################################################################

#-----------------------------TV loss--------------------------------#

######################################################################
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


######################################################################

#---------------------------Recon loss-------------------------------#

######################################################################

class Custom_RMSE_regularize(torch.nn.Module):
    def __init__(self,weight,distanace_map=False,treshold_value=0.2,
            select_MAE='RMSE',partial=False,premask=True,clamp=True):
        super(Custom_RMSE_regularize,self).__init__()
        self.weight = weight
        self.dis_map = distanace_map
        self.treshold_value = treshold_value
        self.select_MAE = select_MAE
        
        self.partial = partial 
        self.premask = premask
        self.clamp = clamp
        
        

    def make_mRMSE(self,feature):
        Bimg = feature[:,1:2]
        Dimg = feature[:,2:3]
        Cimg = feature[:,3:4]

        return torch.abs((1-feature[:,0:1]) - (Bimg + Dimg + Cimg))

    def forward(self,feature_output,mask_inputs,labels,activation='softmax'):
 
        #make mask image 
        if self.premask == True:
            mask_img = mask_inputs
        else:
            zero_img = torch.zeros_like(mask_inputs)
            one_img = torch.ones_like(mask_inputs)
            mask_img = torch.where(mask_inputs>self.treshold_value,one_img,zero_img)
            back_mask_img = torch.where(mask_inputs>self.treshold_value,zero_img,one_img)

        # L1 loss
        if self.partial == True:

            back_part = 1-mask_img
            body_part = (1-feature_output[:,0:1]) - (1-((1-feature_output[:,2:3]) * (1-feature_output[:,3:4] )))
            dend_part = (1-feature_output[:,0:1]) - (1-((1-feature_output[:,1:2]) * (1-feature_output[:,3:4] )))
            axon_part = (1-feature_output[:,0:1]) - (1-((1-feature_output[:,1:2]) * (1-feature_output[:,2:3] )))
            if self.clamp == True:
                dend_part = F.sigmoid(dend_part)
                axon_part = F.sigmoid(axon_part)
                
            BOMAE = torch.abs(body_part - feature_output[:,1:2])
            DEMAE = torch.abs(dend_part - feature_output[:,2:3])
            AXMAE = torch.abs(axon_part - feature_output[:,3:4])

            MAE = DEMAE + AXMAE
            BOMSE = torch.mul(BOMAE,BOMAE)
            DEMSE = torch.mul(DEMAE,DEMAE)
            AXMSE = torch.mul(AXMAE,AXMAE)

            sum_output = axon_part
            back_output = dend_part
            # sum_output  = dend_part
            
            MSE = torch.mean( DEMSE + AXMSE).float()
            

            if self.select_MAE == 'MAE':
                return [torch.mean(MAE).float() * float(self.weight),sum_output,back_output]
            elif self.select_MAE == 'RMSE':
                return [MSE.float() * float(self.weight),sum_output,back_output]
        else : 
            sum_output = self.make_mRMSE(feature_output)
            
        if self.select_MAE == 'MAE':
            return [torch.mean(sum_output).float() * float(self.weight),feature_output[:,2:3],feature_output[:,3:4]]
        elif self.select_MAE == 'RMSE':
            return [torch.mean(torch.pow(sum_output,2)).float() * float(self.weight),feature_output[:,2:3],feature_output[:,3:4]]

######################################################################

#--------------------------NCdice loss-------------------------------#

######################################################################

class NCDICEloss(torch.nn.Module):
    
    def __init__(self,r=1.5):
        super(NCDICEloss,self).__init__()
        self.r = r
        self.treshold_value = 0.3
    # def dice
    # def dice_coef(self,y_true, y_pred):
    #     y_true_f = y_true.contiguous().view(y_true.shape[0], -1)
    #     y_pred_f = y_pred.contiguous().view(y_pred.shape[0], -1)
    #     intersection = torch.sum(torch.pow(torch.abs(y_true_f - y_pred_f),self.r),dim=1)
    #     # print(y_pred_f.shape,y_true_f.shape)
    #     return intersection/(torch.sum(y_true_f.pow(2),dim=1) + torch.sum(y_pred_f.pow(2),dim=1) + 1e-5)

    def forward(self, feature_output,labels,mask_inputs):
        zero_img = torch.zeros_like(mask_inputs)
        one_img = torch.ones_like(mask_inputs)
        mask_img = torch.where(mask_inputs>self.treshold_value,one_img,zero_img)
        back_gt = torch.where(mask_inputs>self.treshold_value,zero_img,one_img)

        labels[:,0:1] = back_gt

        dice = BinaryDiceLoss()
        total_loss= 0
        for i in [0,1,2,3]:
            # if i != self.ignore_index:
            dice_loss = dice( feature_output[:, i],labels[:, i])
            total_loss += dice_loss
        # print(result)
        return total_loss/labels.shape[1]

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, p=1.5, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.pow(torch.abs(predict - target),1.5), dim=1)
        den = torch.sum(predict.pow(self.p), dim=1) + torch.sum(target.pow(self.p), dim=1) + self.smooth

        loss = num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

