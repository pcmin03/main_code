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
from trainer3D import Trainer3d

from myconfig import *

def main(args): 
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    #set devices
    print(torch.cuda.is_available(),'torch.cuda.is_available()')
    
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'else')
    # torch.cuda.set_device(torch.device('cuda:0'))
    print(torch.cuda.get_device_name(0))
    
    # seed seeting
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
    # if os.path.exists(logger.log_dir+"lastsave_models{}.pt"):
    if args.pretrain == True:
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
    print(f'RCE:{args.RCE},NCE:{args.NCE},NCDICE:{args.NCDICE},BCE:{args.BCE},FOCAL:{args.FOCAL}')
    main(args)


#     unet_sample_uint16_wise/5sigmoid_128_oversample_back_filter_seg_gauadaptive_SIGRMSE_100_part_reconloss2_RMSE_1.0
# _0.2_/