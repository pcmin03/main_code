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

# import imgaug.augmenters as iaa
# aug = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))

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
        labs = []
        labels_back = []
        labels_body = []
        labels_dend = []
        labels_axon = []
        global_thresh = 0
        
        selem = disk(2)
        # skeleton_lee = skeletonize(blobs, method='lee')
        print(f"=====making dataset======")
        for i in tqdm(range(len(self.imageDir))):
            img = skimage.io.imread(self.imageDir[i]).astype('float32') /65535.
            lab = skimage.io.imread(self.labelDir[i])
                        
            lab = skimage.color.rgb2gray(lab)

            mask1= (lab > 0.85) & (lab < 0.94)
            mask2= (lab > 0.3) & (lab < 0.4)
            mask3= (lab > 0.4) & (lab < 0.84) 
            mask4= (lab < 0.2) 

            # elif label.ndim == 3:
            lab[mask1],lab[mask2],lab[mask3] = 1,1,1 # cellbody # dendrite # axon
            lab[0] = np.where(np.sum(lab,axis=0)>0,np.zeros_like(lab[0]),np.ones_like(lab[0]))

            if phase == 'train': 
                for i in range(len(lab)): 
                    if i == 1:
                        lab[i] = dilation(lab[i], selem)
                    elif i != 0: 
                        lab[i] = skeletonize(lab[i], method='lee')/255
                        lab[i] = dilation(lab[i], selem)
            global_thresh += threshold_otsu(img)
            #if 3d data imabe duplicate z axis
            if img.shape[1] > 1024:
                center = img.shape[1]//2
                # print(img.shape)
                if img.ndim == 3: 
                    img = img[:,center-512:center+512,center-512:center+512]
                elif img.ndim == 2: 
                    img = img[center-512:center+512,center-512:center+512]
                
                lab = lab[:,center-512:center+512,center-512:center+512]
                
            
            skimage.io.imsave('sample.tif',lab[...,np.newaxis].astype('uint8')*255.)
            
            if phase=='train':  
                im_size = patch_size
                ch_size = int(lab.shape[0])

                if '3d' in self.dataname:
                    img = img[4:-3]
                    lab = lab[4:-3]
                    images.append(view_as_blocks(img ,block_shape=(8,im_size,im_size)))
                    labels.append(view_as_blocks(lab ,block_shape=(8,im_size,im_size)))
                    
                else:
                    images.append(view_as_blocks(img ,block_shape=(im_size,im_size)))
                    labels.append(view_as_blocks(lab ,block_shape=(ch_size,im_size,im_size)))
                    
            else:
                if '3d' in self.dataname:
                    # make zero patch for z axis
                    # fix z axis == 24 during training 
                    print(img.shape,'img.shape')
                    if img.shape[0] < 24: 
                        img = np.concatenate((img,np.zeros_like(img)[0:1]),axis=0)
                        lab = np.concatenate((lab,np.zeros_like(img)[0:1]),axis=0)
                    img = img[:24]
                    lab = lab[:24]
                        
                    images.append(view_as_blocks(img ,block_shape=(24,256,256)))
                    labels.append(view_as_blocks(lab ,block_shape=(24,256,256)))

                else: 
                    # this is validation case 
                    images.append(img)
                    labels.append(lab)
        
        print(f"====start patch image=====")
        print(global_thresh/len(self.imageDir))
        self.imgs = np.array(images)
        print(self.imgs.shape)
        self.labels = np.array(labels)
        
        print(self.imgs.shape)
        mean = self.imgs.mean() 
        std = self.imgs.std()
        if phase =='train':
            self.t_trans= custom_transforms.RandomGaussianBlur(90)
            self.t_trans2= custom_transforms.RandomHorizontalFlip(90)
            self.t_trans3= custom_transforms.RandomRotate(90)
            self.t_trans4= custom_transforms.RandomMultiple(90)
            # normalize3d = custom_transforms.Normalize_3d(0,65535)
            if '3d' in self.dataname:
                num,_,patch,_,z_size,im_size,_=self.imgs.shape
                self.imgs = np.reshape(self.imgs,[num*patch*patch,z_size,im_size,im_size])    
                self.labels = np.reshape(self.labels,[num*patch*patch,z_size,im_size,im_size])
            else: 
                print(self.labels.shape)
                num,_,_,patch,ch_size,im_size,_ = self.labels.shape
                self.imgs = np.reshape(self.imgs,[num*patch*patch,im_size,im_size])    
                self.labels = np.reshape(self.labels,[num*patch*patch,ch_size,im_size,im_size])
            print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
            print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")
            print("=====totalpatch=======")
            
            new_image = []
            new_label = []

            for i,vlaue in enumerate(self.labels):
                mask= (vlaue[0] ==0)
                count_num = np.sum(mask)
                
                if count_num >= ((im_size*im_size)*0.02):
                    new_label.append(vlaue)
                    new_image.append(self.imgs[i])
            print(len(new_label),'lenght...',len(self.labels))
            self.imgs = np.array(new_image)
            self.labels = np.array(new_label)

        else:
            if '3d' in self.dataname:
                numi,_,pa_im,_,zsize,xysize,_  = self.imgs.shape
                self.imgs = self.imgs.reshape(numi*pa_im*pa_im,zsize,xysize,xysize)
                self.labels = self.labels.reshape(numi*pa_im*pa_im,zsize,xysize,xysize)

        self.L_transform =  transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((mean),(std))])

        self.to_tensor = transforms.ToTensor()
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
        if '3d' in self.dataname:
            mask1= (labeldata > 0.85) & (labeldata < 0.94)
            mask2= (labeldata > 0.3) & (labeldata < 0.4)
            mask3= (labeldata > 0.4) & (labeldata < 0.84) 
            mask4= (labeldata < 0.2) 
            
            body_label = labeldata.copy()
            body_label[mask4] = 0 #
            body_label[mask1] = 1 # cellbody
            body_label[mask2] = 0 # dendrite
            body_label[mask3] = 0 # axon
            
            dend_label = labeldata.copy()
            dend_label[mask4] = 0 #
            dend_label[mask1] = 0 # cellbody
            dend_label[mask2] = 1 # dendrite
            dend_label[mask3] = 0 # axon
            
            axon_label = labeldata.copy() 
            axon_label[mask4] = 0 #
            axon_label[mask1] = 0 # cellbody
            axon_label[mask2] = 0 # dendrite
            axon_label[mask3] = 1# axon
            
            back_label = labeldata.copy()
            back_label[mask4] = 1 #
            back_label[mask1] = 0 # cellbody
            back_label[mask2] = 0 # dendrite
            back_label[mask3] = 0 # axon
            labeldata= np.stack((back_label,body_label,dend_label,axon_label),axis=1)
            
            labeldata = np.array(labeldata).astype('float32')
       
        dend,axon,cross = [],[],[]
        dend_n,axon_n,cross_n = 0,0,0
        for num , (i,j) in enumerate(zip(labeldata[:,2],labeldata[:,3])):

            if  i.any() >0 and j.any() > 0:
                cross_n+=1
                cross.append(num)
            elif  i.any() > 0:
                dend_n += 1
                dend.append(num)
            elif  j.any() > 0:
                axon_n += 1
                axon.append(num)

        dend  = np.array(dend)
        axon  = np.array(axon)
        cross = np.array(cross)

        _,dend_multipixel = np.unique(labeldata[cross,2],return_counts=True)
        _,axon_multipixel = np.unique(labeldata[cross,3],return_counts=True)
        _,dend_pixel = np.unique(labeldata[dend,2],return_counts=True)
        _,axon_pixel = np.unique(labeldata[axon,3],return_counts=True)
        
        print(f"label variance : {np.unique(labeldata)}")
        print(f"Number of pixels : {dend_multipixel,axon_multipixel,dend_pixel,axon_pixel}")
        need_pixel = (dend_pixel[1]+dend_multipixel[1]) - (axon_pixel[1]+axon_multipixel[1])
         
        add_image = []
        add_label = []
        total_axon_pixel = 0

        skimage.io.imsave('axontest.tif',np.swapaxes(labeldata[axon],1,3)[...,3:4].astype('uint8')*255.)
        skimage.io.imsave('dedntest.tif',np.swapaxes(labeldata[axon],1,3)[...,2:3].astype('uint8')*255.)
        print( axon)

        while need_pixel >= total_axon_pixel:
            num_axon = np.random.choice(axon,10)
            add_axon = labeldata[num_axon]
            _,axon_pixels = np.unique(add_axon,return_counts=True)
            total_axon_pixel += axon_pixels[1]
            add_label.append(add_axon)
            add_image.append(imgs[num_axon])
        print(need_pixel, total_axon_pixel)
        add_image = np.array(add_image)
        add_label = np.array(add_label)
        if add_label.ndim == 5: 
            batch,zstack,channel,img_size,_= add_label.shape
            add_image = add_image.reshape(batch*zstack,img_size,img_size)
            add_label = add_label.reshape(batch*zstack,channel,img_size,img_size)
        elif add_label.ndim == 6: 
            batch,sample,channel,zstack,img_size,_= add_label.shape
            add_image = add_image.reshape(batch*sample,zstack,img_size,img_size)
            add_label = add_label.reshape(batch*sample,channel,zstack,img_size,img_size)

        imgs = np.concatenate((imgs,add_image),axis=0)
        labels = np.concatenate((labeldata,add_label),axis=0)

        # _,dend_multipixel = np.unique(labels[cross,2],return_counts=True)
        # _,axon_multipixel = np.unique(labels[cross,3],return_counts=True)
        # print(dend_multipixel,axon_multipixel)
        return imgs, labels

    def __len__(self):
        self.number_img = len(self.imgs)
        return self.number_img
    
    def __getitem__(self,index):
        
        image = np.array(self.imgs[index])
        label = np.array(self.labels[index])

        if self.phase =='train':
            if label.ndim == 3:
                if 3 in np.unique(np.argmax(label,axis=0)) and np.sum(label[3]) > 200:
                    sample = {'image':image,'label':label}    
                    sample = self.t_trans3(self.t_trans2(sample))
                    image = sample['image']
                    label = sample['label']

        if self.dataname == 'scribble':
            mask0= label > 0.95
            mask1= (label > 0.85) & (label < 0.94)
            mask2= (label > 0.3) & (label < 0.4)
            mask3= (label > 0.4) & (label < 0.84) 
            mask4= (label < 0.2) 

            label[mask0],label[mask1],label[mask2],label[mask3],label[mask4] = 0,1,1,1,0 
            # ignore part
            # label[mask1] = 1 # cellbody
            # label[mask2] = 1 # dendrite
            # label[mask3] = 1 # axon
            # label[mask4] = 0 # background
            label[0] = np.where(np.sum(label,axis=0)>0,np.zeros_like(label[0]),np.ones_like(label[0]))
        else:
            #give ful_label
            # label = label[...,np.newaxis]
            # mask1= (label > 0.85) & (label < 0.94)
            # mask2= (label > 0.3) & (label < 0.4)
            # mask3= (label > 0.4) & (label < 0.84) 
            # mask4= (label < 0.2) 
            
            if '3d' in self.dataname:
                body_label = label.copy()
                body_label[mask4] = 0 #
                body_label[mask1] = 1 # cellbody
                body_label[mask2] = 0 # dendrite
                body_label[mask3] = 0 # axon
                
                dend_label = label.copy()
                dend_label[mask4] = 0 #
                dend_label[mask1] = 0 # cellbody
                dend_label[mask2] = 1 # dendrite
                dend_label[mask3] = 0 # axon
                
                axon_label = label.copy() 
                axon_label[mask4] = 0 #
                axon_label[mask1] = 0 # cellbody
                axon_label[mask2] = 0 # dendrite
                axon_label[mask3] = 1# axon
                
                back_label = label.copy()
                back_label[mask4] = 1 #
                back_label[mask1] = 0 # cellbody
                back_label[mask2] = 0 # dendrite
                back_label[mask3] = 0 # axon
                label= np.stack((back_label,body_label,dend_label,axon_label),axis=0)
                # print(label.shape,'32323')
                # ch,zsize,xysize,yxsize= label.shape
                # sample = label.reshape(ch*zsize,xysize,yxsize)[:,:,:,np.newaxis]
                # skimage.io.imsave('sample.tif',sample.astype('uint8'))
                # label[0] = np.where(np.sum(label,axis=0)>0,np.zeros_like(label[0]),np.ones_like(label[0]))
                label = np.array(label).astype('float32')
            
            # elif label.ndim == 2:
            #     label[mask1] = 1 # cellbody
            #     label[mask2] = 1 # dendrite
            #     label[mask3] = 1 # axon
            #     label[mask4] = 0 # background
            #     # print(label.shape,'shape')

                label = np.array(label).astype('float32')
        image = image.astype(np.float64)
        global_thresh = threshold_otsu(image)
        _mask = np.where(image > global_thresh/len(self.imageDir),np.zeros_like(image),np.ones_like(image))[np.newaxis]
        
        # print(_mask.max(),_mask.min(),_mask.shape)
        # self.L_transform(
        clip = self.L_transform(image)
        # print(clip.max(),clip.min(),clip.shape)
        #make binary image
        # _mask = self.L_transform(image)
        # print(clip.max(),clip.min())
        # b_threshold = threshold_otsu(clip.cpu().numpy())
        # print(b_threshold)
        # label[0:1] = clip.cpu().numpy()>0.1
        # label[0] = _mask 
        return clip.float(),label,_mask

