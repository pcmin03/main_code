import numpy as np
import skimage ,numbers, random
import torch
import cv2

from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.util.shape import view_as_blocks,view_as_windows
# import cupy as cp
class mydataset_3d(Dataset):
    def __init__(self,imageDir,labelDir,patch_size,stride,oversampling,
    dataname,phase):
        
        
        self.phase = phase
        self.dataname = dataname
        self.imageDir = imageDir
        self.labelDir = labelDir
        # useing scribble or edge label
        images = []
        labels = []
        #full_labels
        global_thresh = 0
        
        print(f"=====making dataset======")
        for i in tqdm(range(len(self.imageDir))):
            img = skimage.io.imread(self.imageDir[i]).astype('float32')
            img = (img - img.min()) / (img.max() - img.min())
            
            lab = skimage.io.imread(self.labelDir[i])
            lab = skimage.color.rgb2gray(lab)
            
            #select label
            mask1= (lab > 0.85) & (lab < 0.94)
            mask2= (lab > 0.3) & (lab < 0.4)
            mask3= (lab > 0.4) & (lab < 0.84) 
            mask4= (lab < 0.2) 

            # elif label.ndim == 3:
            lab[mask1],lab[mask2],lab[mask3] = 1,2,3 # cellbody # dendrite # axon
                
            # global_thresh += threshold_yen(img)
            #if 3d data imabe duplicate z axis
            if img.shape[1] > 1024:
                center = img.shape[1]//2
                img = img[:,center-512:center+512,center-512:center+512]
                lab = lab[:,center-512:center+512,center-512:center+512]
            
            if phase=='train':  
                im_size = patch_size
                ch_size = int(lab.shape[0])
                
                img = img[4:-3]
                lab = lab[4:-3]
                #if data need more dataset 
                images.append(view_as_windows(img ,(8,im_size,im_size),stride))
                labels.append(view_as_windows(lab ,(8,im_size,im_size),stride))
                
            else:
                if img.shape[0] < 24: 
                    img = np.concatenate((img,np.zeros_like(img)[0:1]),axis=0)
                    lab = np.concatenate((lab,np.zeros_like(img)[0:1]),axis=0)
                img = img[:24]
                lab = lab[:24]
                    
                images.append(view_as_blocks(img ,(24,256,256)))
                labels.append(view_as_blocks(lab ,(24,256,256)))

        print(f"====start patch image=====")
        self.mean_thresh = global_thresh/len(self.imageDir)
        self.imgs = np.array(images)
        self.labels = np.array(labels)
        
        mean = self.imgs.mean() 
        std = self.imgs.std()
        if phase =='train':
            num,_,patch,_,z_size,im_size,_=self.imgs.shape
            self.imgs = np.reshape(self.imgs,[num*patch*patch,z_size,im_size,im_size])    
            self.labels = np.reshape(self.labels,[num*patch*patch,z_size,im_size,im_size])

            print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
            print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")
            print("=====totalpatch=======")
            
            new_image = []
            new_label = []

            #data preprocessing 
            for i,vlaue in enumerate(self.labels):
                mask= vlaue > 0.2
                count_num = np.sum(mask)
                if count_num >= (8*(im_size*im_size)*0.02):
                    new_label.append(vlaue)
                    new_image.append(self.imgs[i])
            print(len(new_label),'lenght...',len(self.labels))
            self.imgs = np.array(new_image)
            self.labels = np.array(new_label)

        else:
            numi,_,pa_im,_,zsize,xysize,_  = self.imgs.shape
            self.imgs = self.imgs.reshape(numi*pa_im*pa_im,zsize,xysize,xysize)
            self.labels = self.labels.reshape(numi*pa_im*pa_im,zsize,xysize,xysize)

        self.L_transform =  transforms.Compose([
                        transforms.Lambda(lambda image: torch.tensor(np.array(image).astype(np.float32)).unsqueeze(0))])
        
        self.to_tensor = transforms.ToTensor()
        print(f"imglen: {len(self.imgs)},imgshape:{self.imgs.shape}")
        print(f"lablen:{len(self.labels)},labshape:{self.labels.shape}")
        print(f"totalmean: {mean},totalstd:{std}")
        print("=====totalpatch=======")

        print(self.labels.shape)


    def __len__(self):
        self.number_img = len(self.imgs)
        return self.number_img
    
    def __getitem__(self,index):
        
        image = np.array(self.imgs[index])
        label = np.array(self.labels[index])

        clip = self.L_transform(image)
        label = np.stack(([np.where(label==i, np.ones_like(label),np.zeros_like(label)) for i in range(4)]))
        print(image.max(),image.min())
        return clip.float(),label,clip

