import cv2
import skimage
import torch, glob
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

from natsort import natsorted

import torch.nn.functional as F
from torch.utils.data import  Dataset
from torch import nn
from torch import einsum
from torch.autograd import Variable
from sklearn.model_selection import KFold
from scipy import ndimage



from ND_Crossentory import CrossentropyND, TopKLoss, WeightedCrossEntropyLoss


# custom loss
class Custom_WeightedCrossEntropyLossV2(torch.nn.Module):
    """
    WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    Network has to have NO LINEARITY!
    copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L121
    """
    def forward(self, net_output, gt,Lambda=10):


        gt = gt.long()
        num_classes = net_output.size()[1]
        i0 = 1
        i1 = 2
        pre = net_output
        while i1 < len(net_output.shape): # this is ugly but torch only allows to transpose two axes at once
            net_output = net_output.transpose(i0, i1)
            i0 += 1
            i1 += 1

        net_output = net_output.contiguous()
        net_output = net_output.view(-1, num_classes) #shape=(vox_num, class_num)

        gt = gt.view(-1,)
        # print(net_output.shape,gt.shape)
        weight = torch.FloatTensor([1,1e-2 ,0.1 ,1e+1]).cuda()
        
        BCEloss = F.cross_entropy(net_output ,gt)
        # print(BCEloss * weight_MSEloss,'123123',weight_MSEloss)
        return  BCEloss 
class Custom_CE(torch.nn.Module):
    def __init__(self,weight):
        super(Custom_CE,self).__init__()
        self.weight = weight

    # def Adaptive_NLLloss(self,predict,target,weight):

    #     back = torch.where(target==0,torch.ones_like(target),torch.zeros_like(target))
    #     no_back = torch.abs(back -1 )

    #     zero_ch     = torch.diag(predict[:,target]) * back.float()
    #     not_zero_ch = torch.diag(predict[:,target]) * no_back.float()
        
    #     adptive_weight = 1/(1+int(weight)* torch.abs(predict[:,0] - target.float()))
        
    #     return torch.mean(adptive_weight * zero_ch + not_zero_ch)
    def Adaptive_NLLloss(self,predict, target,weight):
        


        loss = predict.gather(1, target.unsqueeze(1))

        with torch.no_grad():
            back = torch.where(target==0,torch.ones_like(target),torch.zeros_like(target)).float()
            no_back = torch.abs(back -1 )
            # zero_ch     = loss * back.float()
            # not_zero_ch = loss * no_back.float()
            adptive_weight = 1/(1+int(weight)* torch.abs(predict[:,0] - target.float())) * back +no_back
            adptive_weight = adptive_weight.unsqueeze(1)
        # (adptive_weight * zero_ch + not_zero_ch).mean()
        # print(loss.shape,adptive_weight.shape)
        return (loss*adptive_weight).mean() 
    # def my_cross_entropy(self,x, y):
    #     log_prob = -1.0 * F.log_softmax(x, 1)
    #     loss = log_prob.gather(1, y.unsqueeze(1))
    #     loss = loss.mean()
    #     return loss
    def forward(self, net_output, gt):
    

        gt = gt.long()
        num_classes = net_output.size()[1]
        i0 = 1
        i1 = 2
        pre = net_output
        while i1 < len(net_output.shape): # this is ugly but torch only allows to transpose two axes at once
            net_output = net_output.transpose(i0, i1)
            i0 += 1
            i1 += 1

        net_output = net_output.contiguous()
        net_output = net_output.view(-1, num_classes) #shape=(vox_num, class_num)

        gt = gt.view(-1,)
        net_output = -1 * F.log_softmax(net_output,dim=1)
        
        # BCEloss = F.cross_entropy(net_output ,gt)

        return  self.Adaptive_NLLloss(net_output,gt,self.weight)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,gt):
        back_gt = torch.where(gt==0,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
        body_gt = torch.where(gt==1,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
        dend_gt = torch.where(gt==2,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1) 
        axon_gt = torch.where(gt==3,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
        y = torch.cat((back_gt, body_gt,dend_gt,axon_gt),dim=1).cuda().float()

        loss = torch.sqrt(self.mse(yhat,y)+self.eps)
        # print(loss)
        return loss


class Custom_TM_CE(torch.nn.Module):
    """
    WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    Network has to have NO LINEARITY!
    copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L121
    """
    def forward(self, net_output, gt,Lambda=10):
        
        print(net_output.shape,'111')

        # splite prediction channel 
        batch,classnum,xsize = new_output.shape
        split_output=net_output.view(batch,2,classnum//2,xsize,xsize)
        split_ouptut = torch.split(split_output,2,dim=1)

        positive_predict = split_ouptut[0]
        negative_predict = split_output[1]

        # divide channel 
        back_gt = torch.where(gt==0,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
        
        # bac_gt = torch.where(gt==0, torch.zeros_likt(gt),torch.ones_like(gt)).unsqueeze(1)
        body_gt = torch.where(gt==1,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
        dend_gt = torch.where(gt==2,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1) 
        axon_gt = torch.where(gt==3,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
        
        new_gt = torch.cat((back_gt, body_gt,dend_gt,axon_gt),dim=1).cuda().float()

        
        gt = gt.long()
        num_classes = net_output.size()[1]
        i0 = 1
        i1 = 2
        pre = net_output
        while i1 < len(net_output.shape): # this is ugly but torch only allows to transpose two axes at once
            net_output = net_output.transpose(i0, i1)
            i0 += 1
            i1 += 1

        net_output = net_output.contiguous()
        net_output = net_output.view(-1, num_classes) #shape=(vox_num, class_num)

        gt = gt.view(-1,)
        # print(net_output.shape,gt.shape)
        weight = torch.FloatTensor([1,1e-2 ,10 ,10]).cuda()
        
        BCEloss = F.cross_entropy(net_output ,gt)
        # print(BCEloss * weight_MSEloss,'123123',weight_MSEloss)
        return  BCEloss 

class Custom_Adaptive_DistanceMap(torch.nn.Module):
    
    def __init__(self,weight,distanace_map=False):
        super(Custom_Adaptive_DistanceMap,self).__init__()
        self.weight = weight
        self.dis_map = distanace_map

    def forward(self, net_output, gt,stage='forward'):

        # postive predict label

        # divide channel 
        if stage == 'forward':
            back_gt = torch.where(gt==0,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
            body_gt = torch.where(gt==1,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
            dend_gt = torch.where(gt==2,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1) 
            axon_gt = torch.where(gt==3,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
        net_output = net_output
        back_output = net_output[:,0:1,:,:]

        new_gt = torch.cat((back_gt, body_gt,dend_gt,axon_gt),dim=1).cuda().float()

        # L1 loss
        back_gt = back_gt.float()
        back_output = back_output.float()
        
        back_one= back_gt/(1+ int(self.weight)*(torch.abs(back_output - back_gt))).float()
        back_zero = (1-back_gt).float()

        MSE = net_output - new_gt
        RMSE = torch.mul(MSE,MSE).float()
        
        return torch.mean((back_one*RMSE + back_zero*RMSE).float())

class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""
    
    def __init__(self, aux_weight=0.4, ignore_index=-1):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, H, W] -> [batch, 1, H, W]
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(
            1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        #return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)
        return loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight
#     if self.dis_map == True:
#     # zeros = torch.zeros_like(gt).unsqueeze(1)
#     # ones = torch.ones_like(gt).unsqueeze(1)
#     # new_gt = torch.cat((ones, ones,ones,axon_gt*self.weight),dim=1).cuda().float()
#     # Lambda = np.array([ndimage.distance_transform_edt(i) for i in new_gt.cpu().numpy()])
#     # _,_,_,size = Lambda.shape
#     # normalizedImg = np.zeros((size, size))
#     # Lambda = torch.Tensor(cv2.normalize(Lambda,  normalizedImg, 1, self.weight, cv2.NORM_MINMAX)).cuda().float()
#     # RMSE * new_gt 
#     # print(new_gt.max())
#     # class_weight = ones * self.weight
#     RMSE[:,3] = RMSE[:,3] *self.weight
#     # Lambda = new_gt
#     return torch.mean((back_one*RMSE + back_zero*RMSE).float())

# else:     
class Custom_Adaptive_class_DistanceMap(torch.nn.Module):
    
    def __init__(self,weight,distanace_map=False):
        super(Custom_Adaptive_class_DistanceMap,self).__init__()
        self.weight = weight
        self.dis_map = distanace_map

    def forward(self, net_output, gt,stage='forward'):

        # postive predict label

        # divide channel 
        if stage == 'forward':
            back_gt = torch.where(gt==0,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
            body_gt = torch.where(gt==1,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
            dend_gt = torch.where(gt==2,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1) 
            axon_gt = torch.where(gt==3,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
            net_output = net_output
            back_output = net_output[:,0:1,:,:]

            new_gt = torch.cat((back_gt, body_gt,dend_gt,axon_gt),dim=1).cuda().float()

        #distance Map (weight dendrite, axon)
        # L1 loss
        back_gt = back_gt.float()
        back_output = back_output.float()
        
        
        ad_weight = 100
        decrease_weight = 1/(1+ 100)

        back_one = back_gt/(1+ int(self.weight)*(torch.abs(back_output - back_gt))).float()
        back_zero = (1-back_gt).float()

        bodyMSE = net_output[:,1:2] - new_gt[:,1:2]
        dendMSE = net_output[:,2:3] - new_gt[:,2:3]
        axonMSE = net_output[:,3:4] - new_gt[:,3:4]

        MSE = net_output - new_gt
        RMSE = torch.mul(MSE,MSE).float()
        
        bodyRMSE = decrease_weight * torch.mul(bodyMSE,bodyMSE).float()
        dendRMSE = decrease_weight * torch.mul(dendMSE,dendMSE).float()
        axonRMSE = ad_weight * torch.mul(axonMSE,axonMSE).float()

        return torch.mean((back_one*RMSE + back_zero*(bodyRMSE+dendRMSE+axonRMSE)).float())

class Custom_Adaptive_Gaussian_DistanceMap(torch.nn.Module):
    
    def __init__(self,weight,distanace_map=False):
        super(Custom_Adaptive_Gaussian_DistanceMap,self).__init__()
        self.weight = weight
        self.dis_map = distanace_map


    def forward(self, net_output, gt,stage='forward',upsample=False):

        # postive predict label
        # back_one= back_gt * self.gaussian(back_output,back_gt,float(self.weight)).float()

        if upsample == True:
            gt = F.interpolate(gt.unsqueeze(1), net_output.size()[2:], mode='bilinear', align_corners=True).long()
        else : 
            gt =gt.unsqueeze(1)
        # divide channel 
        if stage == 'forward':
            back_gt = torch.where(gt==0,torch.ones_like(gt),torch.zeros_like(gt))
            body_gt = torch.where(gt==1,torch.ones_like(gt),torch.zeros_like(gt))
            dend_gt = torch.where(gt==2,torch.ones_like(gt),torch.zeros_like(gt)) 
            axon_gt = torch.where(gt==3,torch.ones_like(gt),torch.zeros_like(gt))
        
        back_output = net_output[:,0:1,:,:]
        #distance Map (weight dendrite, axon)

        #forground image & forground ground truth 
        new_gt = torch.cat((back_gt, body_gt,dend_gt,axon_gt),dim=1).cuda().float()

        #background image & background ground truth
        back_gt = back_gt.float()
        back_output = back_output.float()

        #Gaussian weight : weight = 0.1
        # gau_numer = -(torch.pow(back_output - back_gt, 2.))
        # gau_deno = float(self.weight)
        # gaussian_fc = torch.exp( gau_numer/gau_deno)

        gau_numer = float(self.weight) *(torch.pow(back_output - back_gt, 2.))
        gau_deno = torch.exp(torch.ones(1)).cuda().float()
        gaussian_fc = torch.exp( -(gau_numer/gau_deno))

        #background loss
        back_one = (back_gt * (gaussian_fc+float(0.0001))).float()
        #forground loss
        back_zero = (1-back_gt).float()

        MSE = net_output - new_gt
        RMSE = torch.mul(MSE,MSE).float()
        
        return torch.mean((back_one*RMSE + back_zero*RMSE).float())

        # return torch.mean((back_one*RMSE + back_zero*(bodyRMSE+dendRMSE+axonRMSE)).float())



        # adpative_loss[:,2:3] = adpative_loss[:,2:3]*((dend_gt.float() + 0.01) * 100)
        # adpative_loss[:,3:4] = adpative_loss[:,3:4]*((axon_gt.float() + 0.01) * 100)
def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)

        
class Custom_Adaptive_CE_RMSE(torch.nn.Module):
    def forward(self, net_output, gt,weight=10,stage='forward'):

        if stage == 'forward':
            back_gt = torch.where(gt==0,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
            body_gt = torch.where(gt==1,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
            dend_gt = torch.where(gt==2,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1) 
            axon_gt = torch.where(gt==3,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
            net_output = net_output[:,0:4]
            back_output = net_output[:,0:1,:,:]

            new_gt = torch.cat((back_gt, body_gt,dend_gt,axon_gt),dim=1).cuda().float()
        else:
            
            back_gt = torch.where(gt==0,torch.zeros_like(gt),torch.ones_like(gt)).unsqueeze(1)
            body_gt = torch.where(gt==1,torch.zeros_like(gt),torch.ones_like(gt)).unsqueeze(1)
            dend_gt = torch.where(gt==2,torch.zeros_like(gt),torch.ones_like(gt)).unsqueeze(1) 
            axon_gt = torch.where(gt==3,torch.zeros_like(gt),torch.ones_like(gt)).unsqueeze(1)
            
            net_output = net_output[:,4:8]
            # print(net_output.shape,'11111111111111111')
            back_output = net_output[:,0:1,:,:]
            # print(back_gt.shape,body_gt.shape,dend_gt.shape,axon_gt.shape)
            new_gt = torch.cat((back_gt, body_gt,dend_gt,axon_gt),dim=1).cuda().float()
            # print(new_gt.shape,'111')
        zeros = torch.zeros_like(gt).unsqueeze(1)
        
        zero_gt = torch.cat((zeros, zeros,dend_gt,axon_gt),dim=1).cuda().float()
        Lambda = np.array([ndimage.distance_transform_edt(i) for i in zero_gt.cpu().numpy()])
        _,_,_,size = Lambda.shape
        normalizedImg = np.zeros((size, size))
        Lambda = torch.Tensor(cv2.normalize(Lambda,  normalizedImg, 1, 100, cv2.NORM_MINMAX)).cuda().float()

    def NLLLoss(self,logs, targets):
        out = torch.zeros_like(targets, dtype=torch.float)
        for i in range(len(targets)):
            out[i] = logs[i][targets[i]]
        return -out.sum()/len(out)

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

class Custom_Adaptive(torch.nn.Module):
    def forward(self, net_output, gt,Lambda=10):
        
        
        batch,classnum,xsize,_ = net_output.shape
        split_output=net_output.view(batch,2,classnum//2,xsize,xsize)
        # print(split_output.shape)
        split_ = torch.split(split_output,1,dim=1)
        
        positive_predict = split_[0][:,0]
        negative_predict = split_[1][:,0]

        #postive predict label
        po_back_output = positive_predict[:,0:1,:,:]
        # gt = positive_predict
        back_gt = torch.where(gt==0,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
        # bac_gt = torch.where(gt==0, torch.zeros_likt(gt),torch.ones_like(gt)).unsqueeze(1)
        body_gt = torch.where(gt==1,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
        dend_gt = torch.where(gt==2,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1) 
        axon_gt = torch.where(gt==3,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
        # print(back_gt.shape,body_gt.shape)
        po_new_gt = torch.cat((back_gt, body_gt,dend_gt,axon_gt),dim=1).cuda().float()
        # print(po_new_gt.shape)
        # negative predict label
        ne_back_output = negative_predict[:,0:1,:,:]
        # gt = negative_predict
        back_gt = torch.where(gt==0,torch.zeros_like(gt),torch.ones_like(gt)).unsqueeze(1)
        # bac_gt = torch.where(gt==0, torch.zeros_likt(gt),torch.ones_like(gt)).unsqueeze(1)
        body_gt = torch.where(gt==1,torch.zeros_like(gt),torch.ones_like(gt)).unsqueeze(1)
        dend_gt = torch.where(gt==2,torch.zeros_like(gt),torch.ones_like(gt)).unsqueeze(1) 
        axon_gt = torch.where(gt==3,torch.zeros_like(gt),torch.ones_like(gt)).unsqueeze(1)

        ne_new_gt = torch.cat((back_gt, body_gt,dend_gt,axon_gt),dim=1).cuda().float()


        # gt = torch.cat()
        # back_gt = gt[:,0:1,:,:]
        back_gt = back_gt.float()
        # RMSE
        po_MSE = positive_predict - po_new_gt
        ne_MSE = negative_predict - ne_new_gt
        po_RMSE = torch.mean(torch.mul(po_MSE,po_MSE))
        ne_RMSE = torch.mean(torch.mul(ne_MSE,ne_MSE))
        
        result = torch.mean(po_RMSE + ne_RMSE)
        # print(result,'123687984321')
        #L1 loss
        # print(positive_predict.shape,po_new_gt.shape)
        # po_MSE = F.multilabel_margin_loss(positive_predict, po_new_gt)
        # ne_MSE = F.multilabel_margin_loss(negative_predict,ne_new_gt)

        # back_one= back_gt/(1+Lambda * (torch.abs(back_output - back_gt)))
        # back_zero = 1-back_gt

        # adpative_loss = (back_one*RMSE + back_zero*RMSE).float()
        
        # adpative_loss[:,2:3] = adpative_loss[:,2:3]*((dend_gt.float() + 0.01) * 100)
        # adpative_loss[:,3:4] = adpative_loss[:,3:4]*((axon_gt.float() + 0.01) * 100)

        return result
        

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)

class GDiceLossV2(nn.Module):
    def __init__(self, apply_nonlin=torch.nn.Softmax(dim=1), smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLossV2, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):

        # new_output = torch.argmax(net_output,dim=1)
        # new_output = torch.where(new_output>1,torch.ones_like(new_output),torch.zeros_like(new_output))
        # gt_ = torch.where(gt>1,torch.ones_like(gt),torch.zeros_like(gt))
        # MSEloss = F.mse_loss(new_output.float(),gt_.float())
        # weight_MSEloss = 1/(1+MSEloss)


        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)


        if self.apply_nonlin is not None:
            softmax_output = self.apply_nonlin(net_output)

        input = flatten(softmax_output)
        target = flatten(y_onehot)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.smooth), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.smooth)

        
#  kfold data
def divide_kfold(imageDir,labelDir,k=4,name='test'):
    images = np.array(natsorted(glob.glob(imageDir+'*')))
    labels = np.array(natsorted(glob.glob(labelDir+'*')))

    kfold = KFold(n_splits=k)

    train = dict()
    label  = dict()
    i = 0
    for train_index, test_index in kfold.split(images):
        print(f"train_index{train_index} \t test_index:{test_index}")
        img_train,img_test = images[train_index], images[test_index]
        lab_train,lab_test = labels[train_index], labels[test_index]
        i+=1
        train.update([('train'+str(i),img_train),(name+str(i),img_test)])
        label.update([('train'+str(i),lab_train),(name+str(i),lab_test)])
    return train,label
    
# prepocessing
class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)
        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        predict=predict[:,1:2].cuda().float()
        target = target.float()
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        gt = target
        back_gt = torch.where(gt==0,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
        body_gt = torch.where(gt==1,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
        # dend_gt = torch.where(gt==2,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1) 
        # axon_gt = torch.where(gt==3,torch.ones_like(gt),torch.zeros_like(gt)).unsqueeze(1)
        
        target = torch.cat((back_gt, body_gt),dim=1).cuda().float()
        # print(predict.shape,target.shape)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]
        
