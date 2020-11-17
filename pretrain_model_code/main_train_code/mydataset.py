import numpy as np
import skimage ,numbers
import glob, random

from natsort import natsorted
# from glob import glob
from numpy.lib.stride_tricks import as_strided
from torch.utils.data import Dataset
import torch
from skimage.filters import threshold_otsu
from torchvision import transforms
from transforms3d import RandomFlip, RandomRotate90, RandomRotate,ToTensor
from skimage.io import imsave

from skimage.util.shape import view_as_blocks
import cv2
import custom_transform as tr
import torch.nn.functional as F
# set image smoothing
from scipy.ndimage import gaussian_filter

class mydataset_2d(Dataset):
    def __init__(self,imageDir,labelDir,usetranform=True,patchwise=True,threshold=0.1,phase='train',multichannel=False,isDir=True,preprocessing=True,multiple_scale=False):
        
        self.preprocessing = preprocessing
        self.multichannel = multichannel
        self.isDir = isDir
        self.multiple_scale = multiple_scale
        if isDir == True:
            self.imageDir = imageDir
            self.labelDir = labelDir
            images = []
            labels = []          
            gaussian_filter

            print(f"=====making dataset======")
            for i in range(len(self.imageDir)):
                
                
                img = skimage.io.imread(self.imageDir[i])
                lab = skimage.io.imread(self.labelDir[i])
                lab = skimage.color.rgb2gray(lab)
                #set hard constraiant 
                if self.preprocessing == True:
                    img = gaussian_filter(img, sigma=1)
                    # img_threshold = skimage.filters.threshold_yen(img)
                    img_threshold = skimage.filters.threshold_otsu(img)
                    img = (img > img_threshold) * img

                # normalizedImg = np.zeros((1024, 1024))
                # img = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
                # img = img.astype('uint8')
                
                if phase=='train':  
                    if patchwise == True:
                        images.append(view_as_blocks(img ,block_shape=(128,128)))
                        labels.append(view_as_blocks(lab ,block_shape=(128,128)))
                    else:    
                        images.append(self.view_as_windows(img ,(128,128),20,threshold=threshold))
                        labels.append(self.view_as_windows(lab ,(128,128),20,threshold=threshold))
                else:
                    images.append(img)
                    labels.append(lab)

            self.images = np.array(images)
            self.labels = np.array(labels)
            
            if phase =='train':
                
                num,patch,_,im_size,_=self.images.shape
                self.images = np.reshape(self.images,[num*patch*patch,im_size,im_size])
                self.labels = np.reshape(self.labels,[num*patch*patch,im_size,im_size])

                new_image = []
                new_label = []
                for i,vlaue in enumerate(self.labels):
                    count_num = np.sum(np.array(vlaue) > 0)
                    
                    if count_num >= ((patch*patch)*threshold):
                        new_label.append(vlaue)
                        new_image.append(self.images[i])
                self.images = np.array(new_image)
                self.labels = np.array(new_label)
                print(self.images.shape,'1111111111111111111111111',self.images[0].dtype)
            if self.images[0].dtype == 'uint8':
                # self.L_transform = transforms.Compose([transforms.ToTensor(),
                #                 transforms.Normalize([0.5], [0.5])])
                self.L_transform = transforms.Lambda(lambda image: torch.tensor(np.array(image).astype(np.float32)).unsqueeze(0))
                
            elif self.images[0].dtype == 'uint16':
                self.L_transform = transforms.Lambda(lambda image: torch.tensor(np.array(image).astype(np.float32)).unsqueeze(0))
                # self.L_transform = transforms.Compose([
                #                 transforms.Lambda(lambda image: torch.tensor(np.array(image).astype(np.float32)).unsqueeze(0)),
                #                 transforms.Normalize([0.5], [0.5])])
        else:
            self.images = np.array(natsorted(glob.glob(imageDir+'*')))
            self.labels = np.array(natsorted(glob.glob(labelDir+'*')))
            self.L_transform = transforms.Lambda(lambda image: torch.tensor(np.array(image).astype(np.float32)).unsqueeze(0))
                
            # self.L_transform = transforms.Compose([transforms.ToTensor(),
            #     transforms.Normalize([0.5], [0.5])])
        self.transform = transforms.Compose([tr.RandomHorizontalFlip()])

            # self.L_transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.Normalize([0.5], [0.5])])
        
        self.usetranform = usetranform

        print(len(self.images),len(self.labels))

    def view_as_windows(self,arr_in, window_shape, step=1,threshold=0.05):
        if not isinstance(arr_in, np.ndarray):
            raise TypeError("`arr_in` must be a numpy ndarray")

        ndim = arr_in.ndim
        
        if isinstance(window_shape, numbers.Number):
            window_shape = (window_shape,) * ndim
        if not (len(window_shape) == ndim):
            raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

        if isinstance(step, numbers.Number):
            if step < 1:
                raise ValueError("`step` must be >= 1")
            step = (step,) * ndim
        if len(step) != ndim:
            raise ValueError("`step` is incompatible with `arr_in.shape`")

        arr_shape = np.array(arr_in.shape)
        window_shape = np.array(window_shape, dtype=arr_shape.dtype)

        if ((arr_shape - window_shape) < 0).any():
            raise ValueError("`window_shape` is too large")

        if ((window_shape - 1) < 0).any():
            raise ValueError("`window_shape` is too small")

        # -- build rolling window view
        slices = tuple(slice(None, None, st) for st in step)
        window_strides = np.array(arr_in.strides)

        indexing_strides = arr_in[slices].strides

        win_indices_shape = (((np.array(arr_in.shape) - np.array(window_shape))
                            // np.array(step)) + 1)

        new_shape = tuple(list(win_indices_shape) + list(window_shape))
        strides = tuple(list(indexing_strides) + list(window_strides))

        arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
        # print(arr_out.shape)
        return arr_out
    # def 
    # num,_,win_shape,_=arr_out.shape
    # arr_out = np.reshape(arr_out,[num * num,win_shape,win_shape])
    
    # new_image = []

    # for i,vlaue in enumerate(arr_out):
    #     count_num = np.sum(np.array(vlaue) > 0)
    #     if count_num >= (4096*threshold):
    #         new_image.append(vlaue)        
    # new_image = np.array(new_image)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        
        image = self.images[index]
        label = self.labels[index]
        if self.isDir == False:
            image = skimage.io.imread(image)
            label = skimage.io.imread(label)
            label = skimage.color.rgb2gray(label)
            if self.preprocessing == True:
                image = gaussian_filter(image, sigma=1)
                # img_threshold = skimage.filters.threshold_yen(image)
                img_threshold = skimage.filters.threshold_otsu(image)
                image = (image > img_threshold) * image

        image = np.array(image)
        label = np.array(label)
        # print(image.dtype)
        

        # print(image.dtype)
        #give label
        mask0= label < 0.3
        mask1= label > 0.9
        mask2= (label >0.3) & (label < 0.4)
        mask3= (label >0.4) & (label < 0.8) 

        label[mask0] = 0
        label[mask1] = 1
        label[mask2] = 2
        label[mask3] = 3

        # mask_full = label >0.2
        
        # label[mask_full==0] = 0
        # label[mask_full==1] = 1
        # print(image.max(),'211')
        
        if self.usetranform == 'train':
            toPIL = transforms.ToPILImage()
            image = toPIL(image)
            label = toPIL(label)
            # image = image.convert('rgb')
            image,label = t_transform([image,label])
            if random.random() > 0.5:        
                angle = random.randint(0, 3)
                roate = tr.RandomRotate(angle*90)
                image,label = roate([image,label])
        # print(image.max(),'111')
        # image = image.convert('rgb')
        # print(image.dtype)
        if self.isDir == False:
            if image.dtype == 'uint16':
                self.L_transform = transforms.Lambda(lambda image: torch.tensor(np.array(image).astype(np.float32)).unsqueeze(0))
                
                clip = self.L_transform(image)
            elif image.dtype=='uint8':
                self.L_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])
                clip = self.L_transform(image)

        else:
            # print(image.dtype,image.shape)
        
            # self.L_transform = transforms.Compose([
            #     transforms.Lambda(lambda image: torch.tensor(np.array(image).astype(np.float32)).unsqueeze(0)),
            #     transforms.Normalize((0.5), (0.5))])
            clip = self.L_transform(image)
        # print(clip.max(),'311')
        label = np.array(label)
        if self.multichannel == True:
            back_lable = np.where(label==0,np.ones_like(label),np.zeros_like(label))
            body_lable = np.where(label==1,np.ones_like(label),np.zeros_like(label))
            dend_lable = np.where(label==2,np.ones_like(label),np.zeros_like(label))
            axon_lable = np.where(label==3,np.ones_like(label),np.zeros_like(label))
            label = [back_lable, body_lable,dend_lable,axon_lable]
        label = np.array(label).astype('float32')

        if self.multiple_scale == True:
            size = image.shape[0] * 2
            # print(size *4)
            # image = skimage.transform.resize(image,(size,size)).astype('uint16') 
            # print(clip.shape,label.shape)
            clip = F.interpolate(clip.unsqueeze(1), (size,size))[:,0] 
            
            label = skimage.transform.resize(label,(size,size))
            # print(clip.shape,label.shape)
        # print(clip.dtype)
        return clip,label
