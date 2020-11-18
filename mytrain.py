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

from utils.logger import Logger
from utils.neuron_util import make_path


from datacode.mydataset import *

from models.mynetwork import init_model
from losses.my_custom_loss import select_loss
from myconfig import *
from trainer import Trainer

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
    trainset,validset = divide_kfold(Dirlist,args)

    #import dataset
    MyDataset = make_dataset(trainset,validset,args)

    #select loss
    loss_list,lossname = select_loss(args)
    print(lossname,'1111')
    # logger 
    main_path, valid_path = make_path(args)
    print(main_path,valid_path,'1111')
    #set log
    logger = Logger(main_path,valid_path+lossname,delete=args.deleteall)

    # continuous training
    if os.path.exists(logger.log_dir+"lastsave_models{}.pth"):
        checkpoint = torch.load(logger.log_dir +"lastsave_models{}.pth")
        gen.load_state_dict(checkpoint['gen_model'])

    #import trainer
    Learner = Trainer(model, MyDataset,loss_list,logger,args,device)
    
    Learner.train()
    # Trainer.test()


if __name__ == '__main__': 
    args = my_config()
    print(args.ADCE,'ADCE')
    print(args.RECONGAU,'RECONGAU')
    print(args.RECON,'RECON')
    print('==============================')
    print(args.RCE,'RCE')
    print(args.NCE,'NCE')
    print(args.NCDICE,'NCDICE')
    print(args.BCE,'BCE')
    main(args)




