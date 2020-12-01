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
        #full_labels
        global_thresh = 0
        
        #set add noise
        self.selem = disk(5)
        print(f"=====making dataset======")
        for i in tqdm(range(len(self.imageDir))):
            img = skimage.io.imread(self.imageDir[i]).astype('float32')
            #noirmlize 0 to 1 image
            img = (img - img.min()) / (img.max() - img.min())

            lab = skimage.io.imread(self.labelDir[i])
            lab = skimage.color.rgb2gray(lab)

            #make mask label
            global_thresh += threshold_yen(img)

            #maks 1024 patch image
            if img.shape[1] > 1024:
                center = img.shape[1]//2
                if img.ndim == 3: 
                    img = img[:,center-512:center+512,center-512:center+512]
                elif img.ndim == 2: 
                    img = img[center-512:center+512,center-512:center+512]
                
                lab = lab[:,center-512:center+512,center-512:center+512]
            
            #set labeling
            mask1= (lab > 0.85) & (lab < 0.94)
            mask2= (lab > 0.3) & (lab < 0.4)
            mask3= (lab > 0.4) & (lab < 0.84) 
            mask4= (lab < 0.2) 
            lab[mask1],lab[mask2],lab[mask3] = 1,1,1 # cellbody # dendrite # axon
            lab[0] = np.where(np.sum(lab,axis=0)>0,np.zeros_like(lab[0]),np.ones_like(lab[0]))
            
            #add noise
            # if self.phase == 'train': 
                # lab[1] = dilation(lab[1], self.selem)
                # lab[2] = dilation(lab[2], self.selem)
                # lab[3] = dilation(lab[3], self.selem)
            
            #make patch dataset
            if phase=='train':  
                im_size = patch_size
                ch_size = int(lab.shape[0])
                images.append(view_as_windows(img ,(im_size,im_size),stride))
                labels.append(view_as_windows(lab ,(ch_size,im_size,im_size),stride))
                    
            else:
                # this is validation case 
                images.append(img)
                labels.append(lab)
        
        print(f"====start patch image=====")
        self.mean_thresh = global_thresh/len(self.imageDir)
        self.imgs = np.array(images)
        self.labels = np.array(labels)
        
        mean = self.imgs.mean() 
        std = self.imgs.std()
        if phase =='train':
            
            #set 3d custom transform  
            # self.t_trans= custom_transforms.RandomGaussianBlur(90)
            self.t_trans2= custom_transforms.RandomHorizontalFlip(90)
            self.t_trans3= custom_transforms.RandomRotate(90)
            self.t_trans4= custom_transforms.RandomMultiple(90)
            
            self.T_trans = transforms.Compose([custom_transforms.RandomRotate(90),
                                                custom_transforms.RandomHorizontalFlip(90),
                                                custom_transforms.Contrast_limited(128)]) 
            num,_,_,patch,ch_size,im_size,_ = self.labels.shape
            self.imgs = np.reshape(self.imgs,[num*patch*patch,im_size,im_size])    
            self.labels = np.reshape(self.labels,[num*patch*patch,ch_size,im_size,im_size])
            print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
            print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")
            print("=====totalpatch=======")
            
            new_image = []
            new_label = []

            #data clean 
            for i,vlaue in enumerate(self.labels):
                count_num = np.sum(1-vlaue[0])
                if count_num >= ((patch*patch)*0.5):
                    new_label.append(vlaue)
                    new_image.append(self.imgs[i])
            print(len(new_label),'lenght...',len(self.labels))
            self.imgs = np.array(new_image)
            self.labels = np.array(new_label)

        self.L_transform =  transforms.Compose([
                        transforms.ToTensor()])

        print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
        print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")
        print(f"totalmean: {mean},totalstd:{std}")
        print("=====totalpatch=======")

        if oversampling==True:
            print("=====oversampling=======")
            self.imgs,self.labels = self.pre_oversampling(self.imgs,self.labels)
            print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
            print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")

    def pre_oversampling(self,imgs,labeldata):
        dend = []
        axon = []
        cross = []
        dend_n = 0
        axon_n = 0
        cross_n = 0
        for num , (i,j) in enumerate(zip(labeldata[:,2],labeldata[:,3])):

            if  i.any() >0 and j.any() > 0:
                cross_n+=1
                cross.append(num)

            elif  i.any() >0:
                dend_n += 1
                dend.append(num)
            elif  j.any() > 0:
                axon_n += 1
                axon.append(num)

        dend = np.array(dend)
        axon  = np.array(axon)
        cross= np.array(cross)
        
        _ ,multipixel_dend = np.unique(labeldata[cross,2],return_counts=True)
        _,multipixel_axon = np.unique(labeldata[cross,3],return_counts=True)
        _,dend_pixel = np.unique(labeldata[dend,2],return_counts=True)
        _,axon_pixel = np.unique(labeldata[axon,3],return_counts=True)
        print(f"Number of pixels : {multipixel_dend,multipixel_axon,dend_pixel,axon_pixel}")

        need_pixel = (dend_pixel[1]+multipixel_dend[1]) - (axon_pixel[1])
        
        # make oversampling image
        add_image = []
        add_label = []
        total_axon_pixel = 0
        while need_pixel >= total_axon_pixel:
            num_axon = np.random.choice(axon,10)
            add_axon = labeldata[num_axon,3:4]
            _,axon_pixels = np.unique(add_axon,return_counts=True)
            total_axon_pixel += axon_pixels[1]
            add_label.append(add_axon)
            add_image.append(imgs[num_axon])
        

        add_image = np.array(add_image)
        add_label = np.array(add_label)
        zeros_label = np.zeros_like(add_label)
        add_label = np.concatenate((zeros_label,zeros_label,zeros_label,add_label),axis=2)

        batch,zstack,channel,img_size,_= add_label.shape
        add_image = add_image.reshape(batch*zstack,img_size,img_size)
        add_label = add_label.reshape(batch*zstack,channel,img_size,img_size)

        # add dataset
        imgs = np.concatenate((add_image,imgs),axis=0)
        labels = np.concatenate((add_label,labeldata),axis=0)
        return imgs, labels

    def __len__(self):
        self.number_img = len(self.imgs)
        return len(self.imgs)
    
    def __getitem__(self,index):
        
        image = np.array(self.imgs[index])
        label = np.array(self.labels[index])
        
        if self.phase =='train':
            sample = {'image':image,'label':label}    
            # sample = self.t_trans3(self.t_trans2(sample))
            sample = self.T_trans(sample)
            image = sample['image']
            label = sample['label']

        label = np.array(label).astype('float32')
        image = image.astype(np.float64)
        _mask = np.where(image > 0.3,np.zeros_like(image),np.ones_like(image))[np.newaxis]
        label[0] = _mask
        clip = self.L_transform(image)
        return clip.float(),label,_mask

