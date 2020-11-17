import torch
import torch.nn.functional as F
from torch import nn

import torchvision
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import segmentation_models_pytorch as smp

from torch.autograd import Variable

from custom_module import *

def dont_train(net):
    '''
    set training parameters to false.
    '''
    for param in net.parameters():
        param.requires_grad = False
    return net
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

def init_model(args)

    if args.modelname =='unet':
        gen = pretrain_unet(3,5).to(device)

    elif args.modelname =='newunet':
        gen = pretrain_unet(1,4,active).to(device)
    
    elif args.modelname =='newunet_last':
        gen = pretrain_unet(1,4,active).to(device)
    
    elif args.modelname =='unet_final':
        gen = pretrain_unet(1,4,active).to(device)

    elif args.modelname =='unet_test':
        gen = pretrain_unet(1,4,active).to(device)
            
    elif args.modelname =='ResidualUNet3D':
        gen = ResidualUNet3D(1,4,final_sigmoid=active).to(device)
        
            
    elif args.modelname =='multinewunet':
            gen = pretrain_multi_unet(1,1,active).to(device)

    elif args.modelname =='newmultinewunet':
        gen = pretrain_multi_unet(1,1,active).to(device)
    
    elif args.modelname =='newunet_compare_new':    
        gen = pretrain_unet(1,4,active).to(device)
        
    elif args.modelname =='newunet_compare_new2':    
        gen = pretrain_unet(1,4,active).to(device)

    return gen
        
            