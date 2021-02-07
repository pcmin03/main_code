import numpy as np
import skimage ,numbers, random
from glob import glob
from natsort import natsorted
from tqdm import tqdm
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
import albumentations as A

from scipy.ndimage.morphology import distance_transform_edt

# import cupy as cp
class mydataset_2d(Dataset):
    def __init__(self,imageDir,labelDir,patch_size,stride,oversampling,
    dataname,phase):
        
        self.phase = phase
        self.mlb = MultiLabelBinarizer
        self.dataname = dataname
        self.imageDir = imageDir
        self.labelDir = labelDir
        # useing scribble or edge label
        images = []
        labels = []
        distance_map = []
        
        # affine argumentation
        self.T_trans = A.Compose([
                        A.RandomCrop(128,128),
                        A.HorizontalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        A.Rotate(limit=30,p=0.5)])    
        # intensity argumentation 
        self.Ttranform = custom_transforms.Contrast_limited(128)

        #     #make distance map & zero-one normalization 
            
        print(f"====start patch image=====")
        self.imgs = np.array(images)
        self.labels = np.array(labels)
        self.distance_map = np.array(distance_map)

        if phase =='train':
            
            num,_,patch,im_size,_ = self.labels.shape
            self.imgs = np.reshape(self.imgs,[num*patch*patch,im_size,im_size])    
            self.labels = np.reshape(self.labels,[num*patch*patch,im_size,im_size])     
            self.distance_map = np.reshape(self.distance_map,[num*patch*patch,im_size,im_size])
     
            new_image = []
            new_label = []
            new_distance = []

            #data clean 
            for i,vlaue in enumerate(self.labels):
                count_num = np.sum(vlaue>0)
                if count_num >= ((im_size*im_size)*0.05):
                    new_label.append(vlaue)
                    new_image.append(self.imgs[i])
                    new_distance.append(self.distance_map[i])

            print(len(new_label),'lenght...',len(self.labels))
            self.imgs = np.array(new_image)
            self.labels = np.array(new_label)
            self.distance_map = np.array(new_distance)
            
        print("=====totalpatch=======")
        print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
        print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")
        
        if oversampling==True:
            print("=====oversampling=======")
            self.imgs,self.labels,self.distance_map = self.pre_oversampling(self.imgs,self.labels,self.distance_map)
            print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
            print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")
            
        self.mean = self.imgs.mean() 
        self.std = self.imgs.std()
        self.L_transform =  transforms.Compose([
                transforms.Lambda(lambda image: torch.tensor(np.array(image).astype(np.float32)).unsqueeze(0))])

    # def pre_oversampling(self,imgs,labeldata,distance_map):
    #     body,dend,axon,cross = [],[],[],[]
    #     dend_n,axon_n,cross_n = 0,0,0
    #     for num , lab in enumerate(labeldata):
    #         body_,dend_,axon_ = lab==1,lab==2,lab==3

    #         if  body_.any() == 1 :
    #             continue
            
    #         elif  dend_.any() > 0 and axon_.any() > 0:
    #             cross_n+=1
    #             cross.append(num)

    #         elif  dend_.any() >0:
    #             dend_n += 1
    #             dend.append(num)
                
    #         elif  axon_.any() > 0:
    #             axon_n += 1
    #             axon.append(num)

    #     dend = np.array(dend)
    #     axon  = np.array(axon)
    #     cross= np.array(cross)
    #     _ ,multipixel_dend = np.unique(labeldata[cross],return_counts=True)
    #     _,dend_pixel = np.unique(labeldata[dend],return_counts=True)
    #     _,axon_pixel = np.unique(labeldata[axon],return_counts=True)   
    #     print(f"Number of pixels : {multipixel_dend,dend_pixel,axon_pixel}")

    #     need_pixel = (dend_pixel[1]+multipixel_dend[1]) - (axon_pixel[1]+multipixel_dend[2])
        
    #     # make oversampling image
    #     add_image,add_label = [],[]
    #     add_distance_map = []
    #     total_axon_pixel = 0
    #     while need_pixel >= total_axon_pixel:
    #         num_axon = np.random.choice(axon,30)
    #         add_axon = labeldata[num_axon]
    #         _,axon_pixels = np.unique(add_axon,return_counts=True)
    #         total_axon_pixel += axon_pixels[1]
    #         add_image.append(imgs[num_axon])
    #         add_label.append(add_axon)
    #         add_distance_map.append(distance_map[num_axon])

    #     add_image,add_label= np.array(add_image), np.array(add_label)
    #     add_distance_map = np.array(add_distance_map)

    #     num,patch,im_size,_=add_image.shape
    #     add_image = np.reshape(add_image,(num*patch,im_size,im_size))
    #     add_label = np.reshape(add_label,(num*patch,im_size,im_size))
    #     add_distance_map = np.reshape(add_distance_map,(num*patch,im_size,im_size))

    #     imgs = np.concatenate((add_image,imgs),axis=0)
    #     labels = np.concatenate((add_label,labeldata),axis=0)
    #     distance_map = np.concatenate((add_distance_map,distance_map),axis=0)
        
    #     print(f"Number of pixels : {total_axon_pixel,need_pixel}")
    #     return imgs, labels,distance_map

    def __len__(self):
        self.number_img = len(self.imgs)
        return len(self.imgs)
    
    def __getitem__(self,index):
        
        image = np.array(self.imgs[index])
        # zero-one normalized
        image = (image - image.min()) / (image.max() - image.min())
        label = np.array(self.labels[index])
        
        dist_img = distance_transform_edt(label>0)
        dist_img = (dist_img - dist_img.min())/ (dist_img.max() - dist_img.min())
        
        # distance = np.array(self.distance_map[index])


                
        if self.phase =='train':
            if label.any() == 3:
                #apply affine transform 
                sample = self.T_trans(image=image,mask=label)
                image = sample['image']
                label = sample['mask']
                #apply intensity transform
                # image = self.Ttranform(image)

            label = [np.where((label==i),np.ones_like(label),np.zeros_like(label)) for i in range(4)]
            mask = np.where(image>0.3,np.zeros_like(image),np.ones_like(image))
            label[0] = mask
        else:
            label = [np.where((label==i),np.ones_like(label),np.zeros_like(label)) for i in range(4)]
            
        label = np.array(label)
        
        backgt,bodygt,dendgt,axongt = label[0],label[1],label[2],label[3]
        image = np.where(label[1],np.zeros_like(image),image)
        label = np.stack([backgt,dendgt,axongt],axis=0)

        clip = self.L_transform(image.astype(np.float64))
    
        return clip.float(),torch.from_numpy(label),torch.from_numpy(distance), torch.from_numpy(bodygt)

