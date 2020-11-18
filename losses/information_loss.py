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

#--------------------------Cross Entropy-----------------------------#

######################################################################
class Custom_CE(torch.nn.Module):
    def __init__(self,weight,Gaussian=True,active='softmax'):
        super(Custom_CE,self).__init__()
        self.weight = weight
        self.Gaussian = Gaussian
        self.active = active

    def make_loss(self,predict,target):
        predict = torch.clamp(predict,min=1e-8,max=1-1e-8)  
        back,p_back = target[:,0:1].requires_grad_(False),predict[:,0:1]
        body,p_body = target[:,1:2].requires_grad_(False),predict[:,1:2]
        dend,p_dend = target[:,2:3].requires_grad_(False),predict[:,2:3]
        axon,p_axon = target[:,3:4].requires_grad_(False),predict[:,3:4]

        no_back = 1- back
        weightback = Variable(back/(1+(int(weight)* torch.abs(p_back - back.float()))).float())
        weightbody = Variable(back/(1+(int(weight)* torch.abs(p_body - body.float()))).float())
        weightdend = Variable(back/(1+(int(weight)* torch.abs(p_dend - dend.float()))).float())
        weightaxon = Variable(back/(1+(int(weight)* torch.abs(p_axon - axon.float()))).float())


        back_bce = weightback*(- back * torch.log(p_back)) - (1 - back) * torch.log(1 - p_back)
        body_bce = - back * torch.log(p_back) - (1 - back) * torch.log(1 - p_back)
        dend_bce = - back * torch.log(p_back) - (1 - back) * torch.log(1 - p_back)
        axon_bce = - back * torch.log(p_back) - (1 - back) * torch.log(1 - p_back)
        return (back_bce + body_bce + dend_bce + axon_bce).mean(axis=0)

    def Adaptive_NLLloss(self,predict, target,weight,Gaussianfn =False):
        
        if active == 'softmax':
            target = target.unsqueeze(1)
            loss = predict.gather(1, target)
            # print(predict.shape,target.shape,predict.max(),target.max(),predict.min(),target.min())
            # with torch.no_grad():
            back = torch.where(target==0,torch.ones_like(target),torch.zeros_like(target)).float()
            no_back = torch.where(target==0,torch.zeros_like(target),torch.ones_like(target)).float()
        elif active == 'sigmoid':
            return torch.mean(self.make_loss(predict,target))

        if Gaussianfn == False:
            adptive_weight = Variable(back/(1+(int(weight)* torch.abs(predict[:,0:1] - back.float()))).float() +no_back)
            adptive_weight = adptive_weight
            # print(adptive_weight.max())
        elif Gaussianfn == True:

            gau_numer = float(self.weight) *torch.abs(predict[:,0] - target.float())
            gau_deno = torch.exp(torch.ones(1)).cuda().float()
            gaussian_fc = torch.exp( -(gau_numer/gau_deno))

            adptive_weight = (gaussian_fc) * back +no_back
            adptive_weight = adptive_weight.unsqueeze(1)
        
        num_classes = predict.size()[1]
        batch_size = predict.size()[0]

        weighted_logs  = (loss*adptive_weight).view(batch_size,-1)
        # print(weighted_logs.shape,adptive_weight.shape,predict.shape,loss.shape,target.shape)
        # print(adptive_weight.mean(),back.shape)

        i0 = 1
        i1 = 2
        pre = predict
        while i1 < len(predict.shape): # this is ugly but torch only allows to transpose two axes at once
            predict = predict.transpose(i0, i1)
            i0 += 1
            i1 += 1

        weighted_loss = weighted_logs.sum(1) / adptive_weight.view(batch_size,-1).sum(1)

        # predict = predict.contiguous()
        # predict = predict.view(-1, num_classes) #shape=(vox_num, class_num)
        # gt = gt.view(-1,)
        return -1 * weighted_loss.mean()

    def forward(self, net_output, gt,upsample=False):
    
        if upsample == True:
            gt = F.interpolate(gt.unsqueeze(1), net_output.size()[2:],mode='bilinear',align_corners =True).long()
            # gt = resize(gt,net_output.size()[2:])
            # print(gt.shape)
            new_gt = gt
        
        gt = gt.long()
        # if self.active = 'softmax':
        #     net_output = F.log_softmax(net_output,dim=1)
        # elif self.active = 'sigmoid':
        #     net_output = net_output
        # BCEloss = F.cross_entropy(net_output ,gt)
        
        if upsample == True:        
            return  [self.Adaptive_NLLloss(net_output,gt,self.weight,Gaussianfn = self.Gaussian),new_gt]
        else:
            if self.active == 'softmax':
                return  self.Adaptive_NLLloss(net_output,gt,self.weight,Gaussianfn = self.Gaussian)
            elif self.active == 'sigmoid':
                return  self.Adaptive_NLLloss(net_output,gt,self.weight,Gaussianfn = self.Gaussian)

class noiseCE(nn.Module):
    def __init__(self,weight,RCE=False,NCE=False,SCE=False,BCE=False,back_filter=True):
        super(noiseCE,self).__init__()
        self.weight = weight
        self.RCE = RCE
        self.NCE = NCE
        self.SCE = SCE
        self.BCE = BCE
        self.back_filter = back_filter
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.treshold_value = 0.3
        self.BCEloss = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self,net_output, gt,mask_inputs):

        # if self.back_filter == True:
        zero_img = torch.zeros_like(mask_inputs)
        one_img = torch.ones_like(mask_inputs)
        mask_img = torch.where(mask_inputs>self.treshold_value,one_img,zero_img)
        back_gt = torch.where(mask_inputs>self.treshold_value,zero_img,one_img)

        # gt[:,0:1] = back_gt

        # reverse cross entropy
        # log_gt = torch.clamp(gt, 1e-4, 1.0)
        # net_output_float = self.softmax(net_output)

        # net_output_float = torch.clamp(net_output_float, 1e-7, 1.0)
        # # net_output_max = torch.argmax(net_output_float,dim=1)
        # rce = (-1*torch.sum(net_output_float * torch.log(log_gt),dim=1)).mean()

        if self.RCE == True:    
            # cross entropy
            # log_output = self.softmax(net_output)
            gt = gt.long()
            gt = torch.argmax(gt,dim=1)
            
            ce = self.cross_entropy(net_output, gt)
            return ce + rce
        
        if self.BCE == True:    
            # cross entropy
            # log_output = self.softmax(net_output)
            total_ce = 0
            # print(gt.shape, net_output.shape)
            
            num_classes = net_output.size()[1]
            net_output = net_output.contiguous()
            net_output = net_output.view(-1, num_classes)

            gt = gt.view(-1,num_classes)
            # print(gt.shape, net_output.shape)
            for i in [0,1,2,3]:     
                total_ce += self.BCEloss(net_output[:,i].float(),gt[:,i])
            
            return total_ce

        elif self.NCE == True: 
            # reverse cross entropy
            # log_output = self.softmax(net_output)
            MAE = torch.abs(gt-self.softmax(net_output)) 
            
            gt = gt.long()
            total_ce = 0
            
            # print(gt.shape)
            for i in [0,1,2,3]:
                each_gt = gt[:,i]
                total_ce += self.loss(net_output,each_gt)

            gt = torch.argmax(gt,dim=1)         
            ce = self.cross_entropy(net_output, gt)   


            # ce = self.loss(log_output, gt)
            
            nce = ce/total_ce
            
            return rce + nce
            # return nce*float(0.1) + rce*float(1) 
        # elif self.SCE == True: 
        #     log_output = self.softmax(net_output)

            # return 
        

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

            # if activation == 'sigmoid':
            #     net_output = self.make_mRMSE(net_output) #make binary image -> 2 channel
            #     out_feature = self.gabor(net_output) 
            #     gt_feature = self.gabor(gt) 
            # elif activation == 'softmax':
                # out_feature = self.gabor(torch.sum(net_output[:,1:4],dim=1).unsqueeze(1))
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
        # print(out_feature.shape,gt_feature.shape,out_feature)
        # 
        # MSE = torch.mean(torch.mul(out_feature,gt_feature)).float()
        # RMSE = torch.sqrt(MSE)
        
        # MSE=torch.abs(out_feature - gt_feature)
        # RMSE = torch.mul(MSE,MSE).float()
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

