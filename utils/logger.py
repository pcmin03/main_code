import os, shutil, random, glob
import numpy as np
import skimage,cv2
from torch.utils.tensorboard import SummaryWriter
from skimage.io import imsave
from natsort import natsorted
import pandas as pd

class Logger(object):
    ### save dictionary ###
    def __init__(self, main_path,valid_path,delete=False):
        self.log_dir =  main_path + valid_path+'/'
        #make deploy path
        if not os.path.exists(self.log_dir):
            print(f'Make_deploy_Dir{self.log_dir}')
            os.makedirs(self.log_dir)
        
        merge_path = main_path+'merge_path/'
            
        if not os.path.exists(merge_path):
            print(f'Make_logger:{merge_path}')
            os.makedirs(merge_path)

        merge_path += valid_path+'/'
        if not os.path.exists(merge_path):
            print(f'Make_logger:{merge_path}')
            os.makedirs(merge_path)

        if delete == True:
            print(f'======================remove_Dir:{merge_path,self.log_dir}======================')
            print('======================remove_Dir:{merge_path,self.log_dir}======================')
            shutil.rmtree(merge_path,ignore_errors=True)
            # shutil.rmtree(self.log_dir,ignore_errors=True)

        print(merge_path,self.log_dir )
        self.writer = SummaryWriter(merge_path)

    def make_full_image(self,images_dict,s_point = 4): 

        for i, img in enumerate(images_dict):
            
            images = images_dict[img][:16]
            himag =[]
            for j in range(s_point):
                full_image=np.concatenate([images[j*4],images[j*4+1],images[j*4+2],images[j*4+3]],axis=-1)
                himag.append(full_image)
                
            himag = np.array(himag)
            full_image = np.concatenate([himag[0],himag[1],himag[2],himag[3]],axis=-2)
            full_image = full_image[np.newaxis]
            
            images_dict[img] = full_image[0]
            
        return images_dict
        
    def summary_images(self,images_dict,step,phase):
        ### list of image ###
        for i, img in enumerate(images_dict):
            if images_dict[img].dtype == 'uint16':
                normalizedImg = (1024,1024)
                images_dict[img] = cv2.normalize(images_dict[img],  normalizedImg, 0,255 , cv2.NORM_MINMAX).astype('uint8')
            if images_dict[img].ndim == 4: 
                self.writer.add_images(str(phase)+'/'+str(img),images_dict[img],step,dataformats='NHWC')
            elif images_dict[img].ndim == 3:
                self.writer.add_image(str(phase)+'/'+str(img),images_dict[img],step,dataformats='HWC')

    def list_summary_scalars(self,scalar_list,step,phase='valid'):
        ### list of scaler ###
        Mavg_dict,IOU_scalar,precision_scalar,recall_scalr,F1score_scalar = scalar_list
        
        self.summary_scalars(IOU_scalar,step,'IOU',phase)
        self.summary_scalars(precision_scalar,step,'precision',phase)
        self.summary_scalars(recall_scalr,step,'recall',phase)
        self.summary_scalars(F1score_scalar,step,'F1',phase)        
        self.summary_scalars(Mavg_dict,step,'mean',phase)
        

    def summary_scalars(self,scalar_dict,step,tag='loss',phase='valid'):
        ### list of scaler ###
        for i, scalar in enumerate(scalar_dict):
            if tag in scalar:
                self.writer.add_scalar(str(tag)+'/'+str(phase)+str(scalar),scalar_dict[scalar],step)

            # elif 'loss' in scalar:
            #     self.writer.add_scalar(str(phase)+'/loss/'+str(scalar),scalar_dict[scalar],step)
            else:
                self.writer.add_scalar(str(phase)+'/'+str(scalar),scalar_dict[scalar],step)
            
    def changedir(self,changedir='result',delete=True):
        
        save_dir = self.log_dir + changedir +'/'
        self.log_dir = save_dir
        
        if delete == True:
            # print('----- remove_Dir-------------')
            shutil.rmtree(self.log_dir,ignore_errors=True)
            # os.remove(self.log_dir+'*')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def save_images(self,images_dict,step):
        ### save images ###
        num = 0
        save_dir = self.log_dir
        
        for i, img in enumerate(images_dict):
            print(images_dict[img].shape,img)
            imsave(save_dir+str(img)+str(step)+'.tif',images_dict[img])
    
    def make_stack_image(self,image_dict):
        for i, img in enumerate(image_dict):
            print(image_dict[img].shape,img)
            image_dict[img] = np.transpose(image_dict[img],(1,2,3,0))[...,0:1]
            if 'input' in img: 
                image_dict[img] = cv2.normalize(image_dict[img],(1024,1024), 0,65535, cv2.NORM_MINMAX).astype('uint16')

        return image_dict 

    def print_value(self,vlaues,state='train'):
        print(f'================{state}=====================')   
        for i, val in enumerate(vlaues):
            print(f"========{val}=>{vlaues[val]}")
    
    def save_csv_file(self,Class,name):
        import pandas
        df = pd.DataFrame(Class,columns =['back','dend','axon'])
        df.to_csv(self.log_dir+'/'+str(name)+'.csv')
        