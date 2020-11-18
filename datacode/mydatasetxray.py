import numpy as np
import skimage ,numbers, random
from glob import glob 
from tqdm import tqdm
from natsort import natsorted

from torch.utils.data import Dataset
import torch
from skimage.filters import threshold_otsu,threshold_yen , threshold_local
from torchvision import transforms

from skimage.io import imsave

from skimage.util.shape import view_as_blocks,view_as_windows
import cv2

import torch.nn.functional as F
# set image smoothing
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize, thin
from skimage.morphology import erosion, dilation, opening, closing,disk

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

import datacode.custom_transforms as custom_transforms

class mydataset_xray(Dataset):
    def __init__(self,imageDir,labelDir,patch_size,stride,oversampling,phase):
        
        
        self.phase = phase
        self.imageDir = imageDir
        self.labelDir = labelDir
        # useing scribble or edge label
        images = []
        labels = []
        #full_labels
        labs = []
        labels_backs = []
        labels_lungs = []
        labels_clavi = []
        labels_heart = []
        
        selem = disk(5)

        print(f"=====making dataset======")
        for i in tqdm(range(len(self.imageDir))):
            img = skimage.io.imread(self.imageDir[i])
            img = skimage.color.rgb2gray(img)
            img = resize(img,(patch_size,patch_size))
            
            lab = []
            for la in self.labelDir: 
                part_label = skimage.io.imread(la[i])
                lab.append(resize(part_label,(patch_size,patch_size),anti_aliasing=True))
            lab = np.array(lab)
            
            lab[1] += lab[3]
            lab[2] += lab[4]
            lab = lab[0:3]

            lab = np.where(lab>0.1,np.ones_like(lab),np.zeros_like(lab))

            if phase == 'train': 
                for i in range(len(lab)): 
                    # if i != 2: 
                    lab[i] = erosion(lab[i], selem)

            back_lab = lab.copy()
            back_lab = (back_lab[0] + back_lab[1] + back_lab[2])[np.newaxis]
            back_lab = np.where(back_lab>0,np.zeros_like(back_lab),np.ones_like(back_lab))

            lab = np.concatenate((back_lab,lab[0:1],lab[2:3],lab[1:2]),axis=0)

            lab = np.argmax(lab,axis=0)

            back_gt = np.where(lab==0,np.ones_like(lab),np.zeros_like(lab))[np.newaxis]
            body_gt = np.where(lab==1,np.ones_like(lab),np.zeros_like(lab))[np.newaxis]
            dend_gt = np.where(lab==2,np.ones_like(lab),np.zeros_like(lab))[np.newaxis]
            axon_gt = np.where(lab==3,np.ones_like(lab),np.zeros_like(lab))[np.newaxis]

            lab = np.concatenate((back_gt, body_gt,dend_gt,axon_gt),axis=0)

            ###########################################################
            ###############make noise label ###########################
            ###########################################################
            
            if phase=='train':  
                im_size = patch_size
                ch_size = int(lab.shape[0])
                images.append(view_as_blocks(img ,block_shape=(im_size,im_size)))
                labels.append(view_as_blocks(lab ,block_shape=(ch_size,im_size,im_size)))
            else:
                # this is validation case 
                images.append(img)
                labels.append(lab)
            
        print(f"====start patch image=====")
        self.labels = np.array(labels)
        self.imgs = np.array(images)

        self.mean = self.imgs.mean()
        self.std = self.imgs.std()
        if phase =='train':
            self.t_trans= custom_transforms.RandomGaussianBlur(90)
            self.t_trans2= custom_transforms.RandomHorizontalFlip(90)
            self.t_trans3= custom_transforms.RandomRotate(90)
            self.t_trans4= custom_transforms.RandomMultiple(90)
            # normalize3d = custom_transforms.Normalize_3d(0,65535)
            print(self.labels.shape)
            num,_,_,patch,ch_size,im_size,_ = self.labels.shape
            self.imgs = np.reshape(self.imgs,[num*patch*patch,im_size,im_size])    
            self.labels = np.reshape(self.labels,[num*patch*patch,ch_size,im_size,im_size])

            print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
            print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")
            print("=====totalpatch=======")

        print(self.imgs[0].dtype,'self.imgs[0].dtype') 
        self.L_transform =  transforms.Compose([
                        transforms.ToTensor()])

        print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
        print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")
        if oversampling==True:
            print("=====oversampling=======")
            self.imgs,self.labels = self.pre_oversampling(self.imgs,self.labels)
            print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
            print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")
         

    def __len__(self):
        self.number_img = len(self.imgs)
        return self.number_img
    
    def __getitem__(self,index):
        
        image = np.array(self.imgs[index])
        label = np.array(self.labels[index])

        if self.phase =='train':
            sample = {'image':image,'label':label}    
            sample = self.t_trans3(self.t_trans2(sample))
            image = sample['image']
            label = sample['label']

        clip = self.L_transform(image).float()
        _mask = self.L_transform(image)
        return clip,label,_mask

        