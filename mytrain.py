import numpy as np
import skimage 
import os ,tqdm , random
from glob import glob


import torch
from utils.logger import Logger
from utils.neuron_util import make_path

from datacode.mydataset import *
from models.mynetwork import init_model
from losses.my_custom_loss import select_loss
from trainer import Trainer

from myconfig import *

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
    
    # logger 
    main_path, valid_path = make_path(args)

    #set log
    logger = Logger(main_path,valid_path+lossname,delete=args.deleteall)

    # continuous training
    if os.path.exists(logger.log_dir+"lastsave_models{}.pt"):
        print('==================load model==================')
        checkpoint = torch.load(logger.log_dir +"lastsave_models{}.pt")
        model.load_state_dict(checkpoint['self.model_model'])

    #import trainer
    Learner = Trainer(model, MyDataset,loss_list,logger,args,device)

    #use tarin
    if args.use_train == True: 
        Learner.train()
    Learner.test()

if __name__ == '__main__': 
    args = my_config()
    print(f'ADCE:{args.ADCE},RECONGAU:{args.RECONGAU},RECON:{args.RECON}')
    print('============Compare loss==================')
    print(f'RCE:{args.RCE},NCE:{args.NCE},NCDICE:{args.NCDICE},BCE:{args.BCE}')
    main(args)