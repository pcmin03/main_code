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
        # self.weight = weight
        self.Gaussian = Gaussian
        self.active = active

    def make_loss(self,predict,target):
        predict = torch.clamp(predict,min=1e-8,max=1-1e-8)  
        back,p_back = target[:,0:1].requires_grad_(False),predict[:,0:1]
        body,p_body = target[:,1:2].requires_grad_(False),predict[:,1:2]
        dend,p_dend = target[:,2:3].requires_grad_(False),predict[:,2:3]
        axon,p_axon = target[:,3:4].requires_grad_(False),predict[:,3:4]

        no_back = 1 - back
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
        
        # if active == 'softmax':
        predict = F.log_softmax(predict)
        # target[:,1] = topr.zeros_like(target[:,1])
        loss = predict.gather(1, torch.argmax(target,1).unsqueeze(1))
        # weight = torch.from_numpy(np.array(weight))
        back = torch.where(target==0,torch.ones_like(target),torch.zeros_like(target)).float()
        no_back = torch.where(target==0,torch.zeros_like(target),torch.ones_like(target)).float()
        # elif active == 'sigmoid':
        #     return torch.mean(self.make_loss(predict,target))

        # if Gaussianfn == False:
        #     adptive_weight = Variable(back/(1+(int(weight)* torch.abs(predict[:,0:1] - back.float()))).float() +no_back)
        #     adptive_weight = adptive_weight
        #     # print(adptive_weight.max())
        # elif Gaussianfn == True:

        #     gau_numer = float(self.weight) *torch.abs(predict[:,0] - target.float())
        #     gau_deno = torch.exp(torch.ones(1)).cuda().float()
        #     gaussian_fc = torch.exp( -(gau_numer/gau_deno))

        #     adptive_weight = (gaussian_fc) * back +no_back
        #     adptive_weight = adptive_weight.unsqueeze(1)
        num_classes = predict.size()[1]
        batch_size = predict.size()[0]

        weighted_logs  = (loss*(1+weight)).view(batch_size,-1)
        # print((1+weight).max(),(1+weight).mean())
        # i0 = 1
        # i1 = 2
        # pre = predict
        # while i1 < len(predict.shape): # this is ugly but torch only allows to transpose two axes at once
        #     predict = predict.transpose(i0, i1)
        #     i0 += 1
        #     i1 += 1

        weighted_loss = weighted_logs.sum(1) / weight.view(batch_size,-1).sum(1)

        return -1 * weighted_loss.mean()

    def forward(self, net_output, gt,weight,upsample=False):
    
        
        if upsample == True:
            gt = F.interpolate(gt.unsqueeze(1), net_output.size()[2:],mode='bilinear',align_corners =True).long()
            new_gt = gt
        

        gt = gt.long()
        # print(weight.shape,'????')
        if upsample == True:        
            return  [self.Adaptive_NLLloss(net_output,gt,weight,Gaussianfn = self.Gaussian),new_gt]
        else:
            if self.active == 'softmax':
                return  self.Adaptive_NLLloss(net_output,gt,weight,Gaussianfn = self.Gaussian)
            elif self.active == 'sigmoid':
                return  self.Adaptive_NLLloss(net_output,gt,weight,Gaussianfn = self.Gaussian)

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
        self.CEloss = torch.nn.NLLLoss(ignore_index=254)

    def forward(self,net_output, gt,mask_inputs,phase):

        # if self.back_filter == True:
        # zero_img = torch.zeros_like(mask_inputs)
        # one_img = torch.ones_like(mask_inputs)
        # mask_img = torch.where(mask_inputs>self.treshold_value,one_img,zero_img)
        # back_gt = torch.where(mask_inputs>self.treshold_value,zero_img,one_img)

        # gt[:,0:1] = back_gt

        # reverse cross entropy
        # log_gt = torch.clamp(gt, 1e-4, 1.0)
        # net_output_float = self.softmax(net_output)

        # net_output_float = torch.clamp(net_output_float, 1e-7, 1.0)
        # # net_output_max = torch.argmax(net_output_float,dim=1)
        # rce = (-1*torch.sum(net_output_float * torch.log(log_gt),dim=1)).mean()
        # if  phase == 'train':
        #     zero_img = torch.zeros_like(mask_inputs)
        #     one_img = torch.ones_like(mask_inputs)
        #     mask_img = torch.where(mask_inputs>self.treshold_value,one_img,zero_img)
        #     back_gt = torch.where(mask_inputs>self.treshold_value,zero_img,one_img)
        #     gt[:,0:1] = back_gt

        if self.RCE == True:    
            # cross entropy
            # log_output = self.softmax(net_output)
            gt = gt.long()
            gt = torch.argmax(gt,dim=1)
            
            ce = self.cross_entropy(net_output, gt)
            return ce + rce
        
        if self.BCE == True:    
            # cross entropy

            # skimage.io.imsave('samplelab.tif',(gt[:100].detach().cpu().numpy()*100).astype('uint8')[...,np.newaxis])
            # skimage.io.imsave('sampleimg.tif',(net_output[:100].detach().cpu().numpy()*100).astype('uint8')[...,np.newaxis])
            
            # total_ce = 0

            stack_mask_inputs = torch.cat((mask_inputs,mask_inputs,mask_inputs,mask_inputs))
            forground = torch.sum(gt[:,1:4],axis=1).unsqueeze(1)
            forground = torch.clamp(forground, min=0, max=1)

            ignore_part = torch.abs((1-gt[:,0:1]) - forground)
            gt = torch.cat((gt,ignore_part),axis=1)

            gt = torch.argmax(gt,axis=1)
            # skimage.io.imsave('samplelab.tif',(gt[:100].detach().cpu().numpy()).astype('uint8')[...,np.newaxis])
            gt[gt==4] = 254
            # gt = gt.view(-1)
            # print(net_output.shape)
            num_classes = net_output.size()[1]
            # print(num_classes,'num_classes')
            net_output = net_output.contiguous()
            
            # net_output = net_output.view(-1, num_classes)
            

            total_ce = self.CEloss(net_output,gt)
            # print(total_ce)
            # for i in [0,1,2,3]:     
            #     total_ce += self.BCEloss(net_output[:,i].float(),gt[:,i])
            
            return total_ce

        elif self.NCE == True: 
            # reverse cross entropy
            MAE = torch.abs(gt-self.softmax(net_output)) 
            
            gt = gt.long()
            total_ce = 0
            for i in [0,1,2,3]:
                each_gt = gt[:,i]
                total_ce += self.loss(net_output,each_gt)

            gt = torch.argmax(gt,dim=1)         
            ce = self.cross_entropy(net_output, gt)   
            
            nce = ce/total_ce
            
            return rce + nce

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
    def forward(self, input, target,_input,phase):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        
        target = torch.argmax(target,dim=1)
        target = target.view(-1,1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

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
