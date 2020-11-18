import numpy as np
import skimage ,numbers , random
from glob import glob
from natsort import natsorted
import torch

from torchvision import transforms

import torch.nn.functional as F
# set image smoothing
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import KFold

from numpy.lib.stride_tricks import as_strided as ast
from torch.utils.data import DataLoader

def resetseed(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
#  kfold data
def divide_kfold(Datadir,args):
    imageDir,labelDir,testDir,tlabelDir = Datadir
    images = np.array(natsorted(glob(imageDir+'*')))
    labels = np.array(natsorted(glob(labelDir+'*')))
    train,valid = dict(),dict()
    
    if args.cross_validation == True:
        # total_label = []
        # for label in labels: 
        #     total_label.append(np.array(natsorted(glob(label+'/*'))))
        # labels = np.array(total_label)
        kfold = KFold(n_splits=args.Kfold)
        i = 0
        
        # print(f"train_index{train_index} \t test_index:{test_index}")
        for train_index, test_index in kfold.split(images):
            img_train,img_test = images[train_index], images[test_index]
            lab_train,lab_test = labels[train_index], labels[test_index]
            i+=1
            
            train.update([('train'+str(i),img_train),('test'+str(i),img_test)])
            valid.update([('train'+str(i),lab_train),('test'+str(i),lab_test)])
        
        train_num, test_num = 'train'+str(args.knum), 'test'+str(args.knum)
        #train set
        image_train = train[train_num]
        label_train = valid[train_num]
        #valid set
        image_valid = train[test_num]
        label_valid = valid[test_num]
        
    else: 
        image_valid = np.array(natsorted(glob(testDir+'*')))
        label_valid = np.array(natsorted(glob(tlabelDir+'*')))
    # print([image_train,image_valid],[label_train,label_valid])
    return [image_train,image_valid],[label_train,label_valid]

def select_data(args):

    if args.datatype == 'uint8':
        #uint8 train
        imageDir= '../new_project_original_image/'
        labelDir = '../new_project_label_modify_image/'
        #uint8 test
        testDir ='../test_image/'
        tlabelDir = '../test_label/'

    elif args.datatype == 'uint16':
        #uint16 train
        imageDir= '../AIAR_orignal_data/train_project_image/'
        labelDir = '../AIAR_orignal_data/train_project_label/'
        #uint16 test
        testDir ='../AIAR_orignal_data/test_project_image/'
        tlabelDir = '../AIAR_orignal_data/test_project_label/'

    elif args.datatype == 'edge':
        #edge dataset train
        imageDir= '../AIAR_orignal_data/train_project_image/'
        labelDir = '../AIAR_orignal_data/train_boundary_label/'
        #edge dataset test
        testDir= '../AIAR_orignal_data/test_project_image/'
        tlabelDir = '../AIAR_orignal_data/test_boundary_label/'

    elif args.datatype == 'uint16_3d_wise':
        #uint16 train
        imageDir= '../AIAR_orignal_data/full_original_data/'
        labelDir = '../AIAR_orignal_data/full_color_label_data/'
        #edge dataset test
        testDir= '../AIAR_orignal_data/test_full_original_data/'
        tlabelDir = '../AIAR_orignal_data/test_full_color_label_data/'

    elif args.datatype == 'uint16_wise':
        #uint16 train
        imageDir= '../AIAR_orignal_data/train_project_image/'
        labelDir = '../AIAR_orignal_data/stack_train_label/'
        #edge dataset test
        testDir= '../AIAR_orignal_data/test_project_image/'
        tlabelDir = '../AIAR_orignal_data/stack_test_label/'

    elif args.datatype == 'uint16_xray':
        #uint16 train
        imageDir= '../xraydataset/train_image/'
        labelDir = '../xraydataset/train_label/'
        #edge dataset test
        testDir= '../xraydataset/test_image/'
        tlabelDir = '../xraydataset/test_label/'

    elif args.datatype == 'scribble':
        #edge dataset train
        imageDir= '../AIAR_orignal_data/train_project_scrrible_image/'
        labelDir = '../AIAR_orignal_data/train_stack_scribble_project_label/'
        
        #edge dataset test
        testDir= '../AIAR_orignal_data/test_project_scrrible_image/'
        tlabelDir = '../AIAR_orignal_data/stack_test_label/'
    
    return [imageDir,labelDir,testDir,tlabelDir]

def make_dataset(trainset,validset,args): 
    num_workers = 16
    if 'xray' in args.datatype:
        from .mydatasetxray import mydataset_xray
        
        MyDataset = {'train': DataLoader(mydataset_xray(trainset[0],validset[0],args.patchsize,
                            args.stride,args.oversample,phase='train'),
                            args.batch_size, 
                            shuffle = True,
                            num_workers = num_workers),
                    'valid' : DataLoader(mydataset_xray(trainset[1],validset[1],args.patchsize,
                            args.stride,False,phase='valid'),
                            1, 
                            shuffle = False,
                            num_workers = num_workers)}
    else:  
        from .mydataset2d import mydataset_2d 

        MyDataset = {'train': DataLoader(mydataset_2d(trainset[0],validset[0],args.patchsize,
                        args.stride,args.oversample,args.datatype,phase='train'),
                        args.batch_size, 
                        shuffle = True,
                        num_workers = num_workers),
                    'valid' : DataLoader(mydataset_2d(trainset[1],validset[1],args.patchsize,
                            args.stride,False,args.datatype,phase='valid'),
                            1, 
                            shuffle = False,
                            num_workers = num_workers)}
    return MyDataset