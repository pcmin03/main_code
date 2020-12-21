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

            lab = np.argmax(lab,axis=0)
            #make patch dataset
            if phase=='train':  
                im_size = patch_size
                ch_size = int(lab.shape[0])
                images.append(view_as_windows(img ,(im_size,im_size),stride))
                labels.append(view_as_windows(lab ,(im_size,im_size),stride))
                    
            else:
                # this is validation case 
                images.append(img)
                labels.append(lab)

        print(f"====start patch image=====")
        self.mean_thresh = global_thresh/len(self.imageDir)
        self.imgs = np.array(images)
        self.labels = np.array(labels)
        if phase =='train':
            
            #set 3d custom transform  
            self.T_trans = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.Rotate(limit=80,p=0.5)])

            # self.T_trans = transforms.Compose([custom_transforms.RandomRotate(90),
            #                                     custom_transforms.RandomHorizontalFlip(90)]) 
            #                                     custom_transforms.Contrast_limited(128)
            # self.Ttranform = custom_transforms.Contrast_limited(128)
            num,_,patch,im_size,_ = self.labels.shape
            self.imgs = np.reshape(self.imgs,[num*patch*patch,im_size,im_size])    
            self.labels = np.reshape(self.labels,[num*patch*patch,im_size,im_size])            
            new_image = []
            new_label = []

            #data clean 
            for i,vlaue in enumerate(self.labels):
                count_num = np.sum(vlaue>0)
                if count_num >= ((patch*patch)*0.5):
                    new_label.append(vlaue)
                    new_image.append(self.imgs[i])
            print(len(new_label),'lenght...',len(self.labels))
            self.imgs = np.array(new_image)
            self.labels = np.array(new_label)

        print("=====totalpatch=======")
        print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
        print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")
        
        if oversampling==True:
            print("=====oversampling=======")
            self.imgs,self.labels = self.pre_oversampling(self.imgs,self.labels)
            print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
            print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")

        self.mean = self.imgs.mean() 
        self.std = self.imgs.std()
        self.L_transform =  transforms.Compose([
                transforms.Lambda(lambda image: torch.tensor(np.array(image).astype(np.float32)).unsqueeze(0))])

        print(f"totalmean: {self.mean},totalstd:{self.std}")

    def pre_oversampling(self,imgs,labeldata):
        body,dend,axon,cross = [],[],[],[]
        dend_n,axon_n,cross_n = 0,0,0
        for num , lab in enumerate(labeldata):
            body_,dend_,axon_ = lab==1,lab==2,lab==3

            if  body_.any() == 1 :
                continue
            
            elif  dend_.any() > 0 and axon_.any() > 0:
                cross_n+=1
                cross.append(num)

            elif  dend_.any() >0:
                dend_n += 1
                dend.append(num)
                
            elif  axon_.any() > 0:
                axon_n += 1
                axon.append(num)

        dend = np.array(dend)
        axon  = np.array(axon)
        cross= np.array(cross)
        _ ,multipixel_dend = np.unique(labeldata[cross],return_counts=True)
        _,dend_pixel = np.unique(labeldata[dend],return_counts=True)
        _,axon_pixel = np.unique(labeldata[axon],return_counts=True)   
        print(f"Number of pixels : {multipixel_dend,dend_pixel,axon_pixel}")

        need_pixel = (dend_pixel[1]+multipixel_dend[1]) - (axon_pixel[1]+multipixel_dend[2])
        
        # make oversampling image
        add_image,add_label = [],[]
        total_axon_pixel = 0
        while need_pixel >= total_axon_pixel:
            num_axon = np.random.choice(axon,30)
            add_axon = labeldata[num_axon]
            _,axon_pixels = np.unique(add_axon,return_counts=True)
            total_axon_pixel += axon_pixels[1]
            add_image.append(imgs[num_axon])
            add_label.append(add_axon)

        add_image,add_label= np.array(add_image), np.array(add_label)

        num,patch,im_size,_=add_image.shape
        add_image = np.reshape(add_image,(num*patch,im_size,im_size))
        add_label = np.reshape(add_label,(num*patch,im_size,im_size))
        
        imgs = np.concatenate((add_image,imgs),axis=0)
        labels = np.concatenate((add_label,labeldata),axis=0)
        # skimage.io.imsave('sampleaxon.tif',imgs[num_axon].astype('uint16')[...,np.newaxis])  
        # skimage.io.imsave('samplededn.tif',add_axon.astype('uint8')[...,np.newaxis])
              
        img_sam,lab_sam = [],[]
        for im,lab in zip(imgs,labels):
            sample = {'image':im,'mask':lab}
            # sample = self.T_trans(image=im, mask =lab)
            img_sam.append(sample['image'])
            lab_sam.append(sample['mask'])
        imgs,labels = np.array(img_sam),np.array(lab_sam)
        
        print(f"Number of pixels : {total_axon_pixel,need_pixel}")
        return imgs, labels

    def __len__(self):
        self.number_img = len(self.imgs)
        return len(self.imgs)
    
    def __getitem__(self,index):
        
        image = np.array(self.imgs[index])
        label = np.array(self.labels[index])
        
        if self.phase =='train':
            if label.any() == 2:
                sample = self.T_trans(image=image,mask=label)
                image = sample['image']
                label = sample['mask']
            # _mask = np.where(image > 65535*0.2,np.zeros_like(image),np.ones_like(image))
            # image = (image -0.9669762930910421)/0.17558176262605804
            # image = self.Ttranform(image)
            label = [np.where((label==i),np.ones_like(label),np.zeros_like(label)) for i in range(4)]
            # label[0] = _mask
        else:
            label = [np.where((label==i),np.ones_like(label),np.zeros_like(label)) for i in range(4)]
        
        label = np.array(label)
        image = image.astype(np.float64)
        mask = np.where(image>0.3,np.ones_like(image),np.zeros_like(image))
        mask = self.L_transform(mask)
        clip = self.L_transform(image)
        
        # _mask = self.L_transform(_mask).permute(1,0,2)
        
        return clip.float(),torch.from_numpy(label),mask

