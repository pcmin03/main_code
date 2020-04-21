import numpy as np
import skimage ,numbers
import glob, random

from natsort import natsorted

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

class mydataset_3d(Dataset):
    def __init__(self,imageDir,labelDir,size,state=False):
        img = glob.glob(imageDir +'*')
        lab = glob.glob(labelDir +'*')

        self.images = natsorted(img)[0:size]
        self.labels = natsorted(lab)[0:size]

        self.state = state

        self.Flip = RandomFlip(np.random.RandomState())
        self.Rotate90 = RandomRotate90(np.random.RandomState())
        self.Rotate = RandomRotate(np.random.RandomState(), angle_spectrum=30)

        self.L_transform = transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        
        print(len(img),len(lab))
    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        
        image = skimage.io.imread(self.images[index])
        labels = skimage.io.imread(self.labels[index])

        length=len(image)//4
        
        zero = np.zeros_like(image[0])
        zero = np.expand_dims(zero, axis=0)

        lab_zero = np.zeros_like(labels[0])
        lab_zero = np.expand_dims(lab_zero, axis=0)
        while len(image) < 25:
            image = np.concatenate((image, zero),axis=0)
            labels = np.concatenate((labels, lab_zero),axis=0)
            
        
        stack_label = []
        for label in labels:
            gray_label = skimage.color.rgb2gray(label)

            mask0= gray_label < 0.3
            mask1= gray_label > 0.9
            mask2= np.logical_and(0.3 < gray_label , gray_label < 0.4)
            mask3=  np.logical_and(0.4 < gray_label , gray_label < 0.8)

            gray_label[mask0] = 0
            gray_label[mask1] = 1
            gray_label[mask2] = 2
            gray_label[mask3] = 3
            stack_label.append(gray_label)
        labels=np.array(stack_label).astype('uint8')
        
        # clip = torch.Tensor([self.L_transform(img) for img in image])
        # image = skimage.color.rgb2gray(image[:25])
        
        if self.state==True:
            image,labels = self.Flip(m = image,n = labels)
            image,labels = self.Rotate90(m = image,n = labels)

        # print(np.expand_dims(image,axis=3).shape)
        # imsave('test2'+str(index)+'.tif',np.expand_dims(image,axis=3))
        # print(image.shape)
        clip = self.L_transform(image).permute(1,0,2,3)
        # print(clip.shape,'111')
        # im = np.transpose(clip.cpu().numpy(),[1,2,3,0])
        # imsave('test'+str(index)+'.tif',np.expand_dims(labels,axis=3))
        # imsave('test2_'+str(index)+'.tif',im)
        
        
        return clip,labels

class mydataset_2d(Dataset):
    def __init__(self,imageDir,labelDir,usetranform=True,patchwise=True,threshold=0.1,phase='train',multi=False):

        self.imageDir = imageDir
        self.labelDir = labelDir
        self.multi = multi
        images = []
        labels = []
          
        print(f"=====making dataset======")
        for i in range(len(self.imageDir)):
            
            img = skimage.io.imread(self.imageDir[i])
            lab = skimage.io.imread(self.labelDir[i])
            lab = skimage.color.rgb2gray(lab)
            
            normalizedImg = np.zeros((1024, 1024))
            img = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
            img = img.astype('uint8')
            
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

        self.transform = transforms.Compose([tr.RandomHorizontalFlip()])
        self.L_transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])])
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

        image = np.array(image).astype('uint8')
        label = np.array(label)
        
        #give label
        mask0= label < 0.3
        mask1= label > 0.9
        mask2= (label >0.3) & (label < 0.4)
        mask3= (label >0.4) & (label < 0.8) 

        label[mask0] = 0
        label[mask1] = 1
        label[mask2] = 2
        label[mask3] = 3


        if self.usetranform == 'train':
            toPIL = transforms.ToPILImage()
            image = toPIL(image.astype('uint8'))
            label = toPIL(label.astype('uint8'))
            # image = image.convert('rgb')
            image,label = t_transform([image,label])
            if random.random() > 0.5:        
                angle = random.randint(0, 3)
                roate = tr.RandomRotate(angle*90)
                image,label = roate([image,label])
                
        # image = image.convert('rgb')

        clip = self.L_transform(image)
        label = np.array(label)
        if self.multi == True:
            back_lable = np.where(label==0,np.ones_like(label),np.zeros_like(label))
            body_lable = np.where(label==1,np.ones_like(label),np.zeros_like(label))
            dend_lable = np.where(label==2,np.ones_like(label),np.zeros_like(label))
            axon_lable = np.where(label==3,np.ones_like(label),np.zeros_like(label))
            label = [back_lable, body_lable,dend_lable,axon_lable]
        label = np.array(label)

        return clip,label

class prjection_mydataset(Dataset):
    def __init__(self,imageDir,labelDir,usetranform=True,kfold_cross=True):
        
        if kfold_cross == True:
            self.images = imageDir
            self.labels = labelDir
            
        elif kfold_cross == False: 
            img = glob.glob(imageDir +'*')
            lab = glob.glob(labelDir +'*')

            self.images = natsorted(img)
            self.labels = natsorted(lab)

        self.transform = transforms.Compose([tr.RandomHorizontalFlip()])
        self.L_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.usetranform = usetranform

        print(len(self.images),len(self.labels))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        
        image = skimage.io.imread(self.images[index])
        label = skimage.io.imread(self.labels[index])

        image = np.array(image).astype('uint8')
        label = np.array(label)

        label = np.max(label,axis=0).astype('uint8')
        label2 = skimage.color.rgb2gray(label)
        
        #give label
        mask0= label2 < 0.3
        mask1= label2 > 0.9
        mask2= (label2 >0.3) & (label2 < 0.4)
        mask3= (label2 >0.4) & (label2 < 0.8) 

        label2[mask0] = 0
        label2[mask1] = 1
        label2[mask2] = 2
        label2[mask3] = 3
        image= np.max(image, axis=0)
        image = skimage.color.gray2rgb(image)
        if self.usetranform == 'train':
            toPIL = transforms.ToPILImage()
            image = toPIL(image.astype('uint8'))
            label2 = toPIL(label2.astype('uint8'))
            # image = image.convert('rgb')
            image,label2 = t_transform([image,label2])
            if random.random() > 0.5:        
                angle = random.randint(0, 3)
                roate = tr.RandomRotate(angle*90)
                image,label2 = roate([image,label2])
                
        # image = image.convert('rgb')

        clip = self.L_transform(image)
        label2 = np.array(label2)

        return clip,label2

class nested_mydataset(Dataset):
    def __init__(self,imageDir,labelDir,usetranform=True,kfold_cross=True):
        if kfold_cross == False:
            img = glob.glob(imageDir +'*')
            lab = glob.glob(labelDir +'*')

            self.images = natsorted(img)
            self.labels = natsorted(lab)

        if kfold_cross == True: 
            self.images = imageDir
            self.labels = labelDir

        self.transform = transforms.Compose([tr.RandomHorizontalFlip()])
        self.L_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.usetranform = usetranform

        print(len(self.images),len(self.labels))
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        
        image = skimage.io.imread(self.images[index])
        label = skimage.io.imread(self.labels[index])

        image = np.array(image).astype('uint8')
        label = np.array(label)

        label = np.max(label,axis=0).astype('uint8')
        label2 = skimage.color.rgb2gray(label)
        
        #give label
        mask0= label2 < 0.3
        mask1= label2 > 0.9
        mask2= (label2 >0.3) & (label2 < 0.4)
        mask3= (label2 >0.4) & (label2 < 0.8) 

        label2[mask0] = 0
        label2[mask1] = 1
        label2[mask2] = 2
        label2[mask3] = 3
        image= np.max(image, axis=0)
        image = skimage.color.gray2rgb(image)
        if self.usetranform == 'train':
            toPIL = transforms.ToPILImage()
            image = toPIL(image.astype('uint8'))
            label2 = toPIL(label2.astype('uint8'))
            # image = image.convert('rgb')
            image,label2 = t_transform([image,label2])
            if random.random() > 0.5:        
                angle = random.randint(0, 3)
                roate = tr.RandomRotate(angle*90)
                image,label2 = roate([image,label2])
                
        # image = image.convert('rgb')

        clip = self.L_transform(image)
        label2 = np.array(label2)

        return clip,label2


# class kfold_dataset(Dataset):
#     def __init__(self,imageDir):
#         data = glob.glob(imageDir+'*')

#         self.data = natsorted(data)

#     def __len__(self):
#         return len(self,self.data)
        