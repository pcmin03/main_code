import numpy as np
import skimage ,numbers
import glob, random

from natsort import natsorted
import torch

from torchvision import transforms

import torch.nn.functional as F
# set image smoothing
from scipy.ndimage import gaussian_filter

import custom_transforms
from numpy.lib.stride_tricks import as_strided as ast

def resetseed(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
#  kfold data
def divide_kfold(Datadir,k=4,dataname='uint16_xray',cross_validation == True):
    
    imageDir,labelDir,testDir,tlabelDir = Datadir
    image_train = np.array(natsorted(glob(imageDir+'*')))
    label_train = np.array(natsorted(glob(labelDir+'*')))
    train = dict()
    label  = dict()
    
    if cross_validation == True:
        total_label = []
        for label in label_train: 
            total_label.append(np.array(natsorted(glob(label+'/*'))))
        label_train = np.array(total_label)
        
        kfold = KFold(n_splits=k)
        i = 0
        
        # print(f"train_index{train_index} \t test_index:{test_index}")
        for train_index, test_index in kfold.split(image_train):
            img_train,img_test = image_train[train_index], image_train[test_index]
            lab_train,lab_test = label_train[:,train_index], label_train[:,test_index]
            i+=1
            train.update([('train'+str(i),img_train),('test'+str(i),img_test)])
            label.update([('train'+str(i),lab_train),('test'+str(i),lab_test)])
        
        train_num, test_num = 'train'+str(args.knum), 'test'+str(args.knum)
        #train set
        image_train = image[train_num]
        label_train = label[train_num]
        #valid set
        image_valid = image[test_num]
        label_valid = label[test_num]
        
    else: 
        image_valid = np.array(natsorted(glob(testDir+'*')))
        label_valid = np.array(natsorted(glob(tlabelDir+'*')))

    return [image_train,image_valid],[label_train,label_valid]

def select_data(args):
    if args.data_name == 'uint8':
        #uint8 train
        imageDir= '../../new_project_original_image/'
        labelDir = '../../new_project_label_modify_image/'
        #uint8 test
        testDir ='../../test_image/'
        tlabelDir = '../../test_label/'

    elif args.data_name == 'uint16':
        #uint16 train
        imageDir= '../../AIAR_orignal_data/train_project_image/'
        labelDir = '../../AIAR_orignal_data/train_project_label/'
        #uint16 test
        testDir ='../../AIAR_orignal_data/test_project_image/'
        tlabelDir = '../../AIAR_orignal_data/test_project_label/'

    elif args.data_name == 'edge':
        #edge dataset train
        imageDir= '../../AIAR_orignal_data/train_project_image/'
        labelDir = '../../AIAR_orignal_data/train_boundary_label/'
        #edge dataset test
        testDir= '../../AIAR_orignal_data/test_project_image/'
        tlabelDir = '../../AIAR_orignal_data/test_boundary_label/'

    elif args.data_name == 'uint16_3d_wise':
        #uint16 train
        imageDir= '../../AIAR_orignal_data/full_original_data/'
        labelDir = '../../AIAR_orignal_data/full_color_label_data/'
        #edge dataset test
        testDir= '../../AIAR_orignal_data/test_full_original_data/'
        tlabelDir = '../../AIAR_orignal_data/test_full_color_label_data/'

    elif args.data_name == 'uint16_wise':
        #uint16 train
        imageDir= '../../AIAR_orignal_data/train_project_image/'
        labelDir = '../../AIAR_orignal_data/stack_train_label/'
        #edge dataset test
        testDir= '../../AIAR_orignal_data/test_project_image/'
        tlabelDir = '../../AIAR_orignal_data/stack_test_label/'

    elif args.data_name == 'uint16_xray':
        #uint16 train
        imageDir= '../../xraydataset/train_image/'
        labelDir = '../../xraydataset/train_label/'
        #edge dataset test
        testDir= '../../xraydataset/test_image/'
        tlabelDir = '../../xraydataset/test_label/'

    elif args.data_name == 'scribble':
        #edge dataset train
        imageDir= '../../AIAR_orignal_data/train_project_scrrible_image/'
        labelDir = '../../AIAR_orignal_data/train_stack_scribble_project_label/'
        
        #edge dataset test
        testDir= '../../AIAR_orignal_data/test_project_scrrible_image/'
        tlabelDir = '../../AIAR_orignal_data/stack_test_label/'
    
    return [imageDir,labelDir,testDir,tlabelDir]

def make_dataset(args): 
        
    if 'xray' in args.data_name:
        from mydatasetxray import mydataset_xray
        
        MyDataset = {'train': DataLoader(mydataset_xray(trainset[0],validset[0],patchwise=patchwise,
                            phase='train',preprocessing=True,multiple_scale=multiple_scale,
                            patch_size=args.patchsize,stride=args.stride,oversampling = args.oversample,
                            dataname = args.data_name),
                            batch_size, 
                            shuffle = True,
                            num_workers = num_workers),
            'valid' : DataLoader(mydataset_xray(trainset[1],validset[1],False,patchwise=False,
                            phase='valid',preprocessing=False,multiple_scale=multiple_scale,
                            patch_size=args.patchsize,stride=args.stride,oversampling = False,
                            dataname = args.data_name),
                                1, 
                                shuffle = False,
                                num_workers = num_workers)}
    else:  
        from mydataset2d import mydataset_2d 

        MyDataset = {'train': DataLoader(mydataset_2d(trainset[0],validset[0],patchwise=patchwise,
                        phase='train',preprocessing=True,multiple_scale=multiple_scale,
                        patch_size=args.patchsize,stride=args.stride,oversampling = args.oversample,
                        dataname = args.data_name),
                        batch_size, 
                        shuffle = True,
                        num_workers = num_workers),
        'valid' : DataLoader(mydataset_2d(trainset[1],validset[1],False,patchwise=False,
                        phase='valid',preprocessing=False,multiple_scale=multiple_scale,
                        patch_size=args.patchsize,stride=args.stride,oversampling = False,
                        dataname = args.data_name),
                            1, 
                            shuffle = False,
                            num_workers = num_workers)}
    return MyDataset