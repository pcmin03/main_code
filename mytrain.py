import numpy as np
import skimage 
import os ,tqdm , random
from glob import glob
import torch
import torch.nn.functional as F
import yaml

from torch import nn, optim
from torchvision import models ,transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch import Tensor
#custom set#

from my_network3d import ResidualUNet3D
from utils.logger import Logger
from utils.mydataset import *
from utils.neuron_util import make_path
from models.mynetwork import init_model
from losses.my_custom_loss import select_loss

import Trainer

def main(args): 
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    #set devices
    print(torch.cuda.is_available(),'torch.cuda.is_available()')
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'else')

    #seed seeting
    resetseed(random_seed=2020)

    #inport network
    model = init_model(args,device)
    
    #select dataset
    Dirlist = select_data(args)

    #cross validation
    trainset,validset = divide_kfold(Dirlist,k=args.foldnum,dataname=args.data_name,cross_validation == args.cross_validation)

    #import dataset
    MyDataset = make_dataset(trainset,validset,args)

    #select loss
    loss_list,lossname = select_loss(args)
    
    # logger 
    main_path, valid_path = make_path(args)

    #set log
    logger = Logger(main_path,valid_path+lossname,delete=args.deleteall)

    # continuous training
    if os.path.exists(logger.log_dir+"lastsave_models{}.pth"):
        checkpoint = torch.load(logger.log_dir +"lastsave_models{}.pth")
        gen.load_state_dict(checkpoint['gen_model'])

    #import trainer
    Trainer = Trainer(model, Mydataset,loss_list,optimizer,scheduler,logger,args)
    
    Trainer.train(args)
    Trainer.test()


if __name__ == '__main__': 
    args = config
    print(adce,'adce')
    print(adrmse,'adrmse')
    print(recon_gau,'recon_gau')
    print(recon,'recon')
    print(gabor_loss,'gabor_loss')
    print('==============================')
    print(args.RCE,'RCE')
    print(args.NCE,'NCE')
    print(args.NCDICE,'NCDICE')
    print(args.BCE,'BCE')

    main(config)




