import torch
from .my_network import *  
# from .unet3d.model import UNet3D
from .unet3d_model.unet3d import UnetModel
def dont_train(net):
    '''
    set training parameters to false.
    '''
    for param in net.parameters():
        param.requires_grad = False
    return net
    
#weight initilize
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

#load training model
def init_model(args,device):

    if args.modelname =='unet':
        gen = pretrain_unet(3,5).to(device)

    elif args.modelname =='newunet':
        gen = pretrain_unet(1,4,args.activation).to(device)
    
    elif args.modelname =='newunet_last':
        gen = pretrain_unet(1,4,args.activation).to(device)
    
    elif args.modelname =='unet_final':
        gen = pretrain_unet(1,4,args.activation).to(device)

    elif args.modelname =='unet_test':
        gen = pretrain_unet(1,4,args.activation).to(device)
            
    elif args.modelname =='ResidualUNet3D':
        # gen = UNet3D(1,4,final_sigmoid=args.activation).to(device)
        gen = UnetModel(1,4).to(device)

    elif args.modelname =='multinewunet':
        gen = pretrain_multi_unet(1,1,args.activation).to(device)

    elif args.modelname =='newmultinewunet':
        gen = pretrain_multi_unet(1,1,args.activation).to(device)
    
    elif args.modelname =='newunet_compare_new':    
        gen = pretrain_unet(1,4,args.activation).to(device)
        
    elif args.modelname =='newunet_compare_new2':    
        gen = pretrain_unet(1,4,args.activation).to(device)

    return gen
        
            