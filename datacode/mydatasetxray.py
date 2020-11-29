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
from skimage.transform import resize 

import datacode.custom_transforms as custom_transforms
import albumentations as A

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
        
        self.selem = disk(5)
        self.patch_size = patch_size
        self.phase = phase
        print(f"=====making dataset======")
        
        for i in tqdm(range(len(self.imageDir))):
            img = skimage.io.imread(self.imageDir[i]).astype('float32')
            
            img = skimage.color.rgb2gray(img)
            img = resize(img,(patch_size,patch_size))

            img = (img - img.min()) / (img.max() - img.min())
            lab = []
            for la in self.labelDir: 
                la = natsorted(glob(la+'/*'))
                part_label = skimage.io.imread(la[i])
                lab.append(resize(part_label,(patch_size,patch_size),anti_aliasing=True))
            lab = np.array(lab)
            lab[1] += lab[3]
            lab[2] += lab[4]
            lab = lab[0:3]

            lab = np.where(lab>0.1,np.ones_like(lab),np.zeros_like(lab))

            if phase == 'train': 
                for i in range(len(lab)):
                    lab[i] = erosion(lab[i], self.selem)

            back_lab = lab.copy()
            back_lab = (back_lab[0] + back_lab[1] + back_lab[2])[np.newaxis]
            back_lab = np.where(back_lab>0,np.zeros_like(back_lab),np.ones_like(back_lab))

            lab = np.concatenate((back_lab,lab[0:1],lab[2:3],lab[1:2]),axis=0)

            # lab = np.argmax(lab,axis=0)
            # back_gt = np.where(lab==0,np.ones_like(lab),np.zeros_like(lab))[np.newaxis]
            # body_gt = np.where(lab==1,np.ones_like(lab),np.zeros_like(lab))[np.newaxis]
            # dend_gt = np.where(lab==2,np.ones_like(lab),np.zeros_like(lab))[np.newaxis]
            # axon_gt = np.where(lab==3,np.ones_like(lab),np.zeros_like(lab))[np.newaxis]
            # lab = np.concatenate((back_gt, body_gt,dend_gt,axon_gt),axis=0)

            ###########################################################
            ###############make noise label ###########################
            ###########################################################
            # if phase=='train':  
            #     im_size = patch_size
            #     ch_size = int(lab.shape[0])
            #     images.append(view_as_blocks(img ,block_shape=(im_size,im_size)))
            #     labels.append(view_as_blocks(lab ,block_shape=(ch_size,im_size,im_size)))
            # else:
                # this is validation case 
                    
            images.append(img)
            labels.append(lab)
            
        print(f"====start patch image=====")
        self.labels = np.array(labels)
        self.imgs = np.array(images)
        if self.phase =='train':
            self.t_trans= custom_transforms.RandomGaussianBlur(90)
            self.t_trans2= custom_transforms.RandomHorizontalFlip(90)
            self.t_trans3= custom_transforms.RandomRotate(90)
            self.t_trans4= custom_transforms.RandomMultiple(90)
            # normalize3d = custom_transforms.Normalize_3d(0,65535)
            # print(self.labels.shape)
            # num,_,_,patch,ch_size,im_size,_ = self.labels.shape
            # self.imgs = np.reshape(self.imgs,[num*patch*patch,im_size,im_size])    
            # self.labels = np.reshape(self.labels,[num*patch*patch,ch_size,im_size,im_size])

            print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
            print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")
            print("=====totalpatch=======")
 
        self.L_transform =  transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.249945),(0.200682))])

        print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
        print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")


    def __len__(self):
        return len(self.imageDir)
    
    def __getitem__(self,index):
        image = np.array(self.imgs[index])
        label = np.array(self.labels[index])

        if self.phase == 'train':
            sample = {'image':image,'label':label}
            sample = self.t_trans3(self.t_trans2(sample))
            image = sample['image']
            label = sample['label']
        # image = image[]

        label = np.array(label).astype('float32')
        image = image.astype(np.float64)
        _mask = np.where(image > 0.25,np.zeros_like(image),np.ones_like(image))[np.newaxis]

        clip = self.L_transform(image).float()
        
        return clip,label,clip

        