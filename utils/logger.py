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

        merge_path += valid_path
        if not os.path.exists(merge_path):
            print(f'Make_logger:{merge_path}')
            os.makedirs(merge_path)

        if delete == True:
            print(f'remove_Dir:{merge_path,self.log_dir}')
            shutil.rmtree(self.log_dir+'*',ignore_errors=True)  
            shutil.rmtree(merge_path,ignore_errors=True)
        print(merge_path,self.log_dir )
        self.writer = SummaryWriter(merge_path)
        
    def summary_images(self,images_dict,step):
        ### list of image ###
        for i, img in enumerate(images_dict):
            if images_dict[img].dtype == 'uint16':
                normalizedImg = (1024,1024)
                images_dict[img] = cv2.normalize(images_dict[img],  normalizedImg, 0,255 , cv2.NORM_MINMAX).astype('uint8')
            if images_dict[img].ndim == 4: 
                self.writer.add_images(str(img),images_dict[img],step,dataformats='NHWC')
            elif images_dict[img].ndim == 3:
                self.writer.add_image(str(img),images_dict[img],step,dataformats='HWC')

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
            
            #change NCHW to NHWC save stack_image of TIF file
            #3d image
            print(images_dict[img].shape,img)
            # if images_dict[img].ndim == 4:
            #     result_image = np.transpose(images_dict[img])
            #2d image
            # elif images_dict[img].ndim ==3:
                # result_image = np.transpose(images_dict[img])

            imsave(save_dir+str(img)+str(step)+'.tif',images_dict[img])
    
    def make_stack_image(self,image_dict):
        for i, img in enumerate(image_dict):
            print(image_dict[img].shape,img)
            image_dict[img] = image_dict[img].detach().cpu().numpy()
            image_dict[img] = np.transpose(image_dict[img],(1,2,3,0))[...,0:1]
            # image_dict[img] = np.swapaxes(image_dict[img],0,-1)[...,0:1]

            print(image_dict[img].shape,img)
        image_dict['_input'] = cv2.normalize(image_dict['_input'],(1024,1024), 0, 65535 , cv2.NORM_MINMAX).astype('uint16')
        image_dict['prediction_map'] = cv2.normalize(image_dict['prediction_map'],(1024,1024), 0, 65535 , cv2.NORM_MINMAX)
        image_dict['predict_channe_wise'] = cv2.normalize(image_dict['predict_channe_wise'],(1024,1024), 0, 65535 , cv2.NORM_MINMAX).astype('uint16')

        return image_dict 

    def print_value(self,vlaues,state='train'):
        if state == 'train':
            print("================trainning=====================")   
            for i, val in enumerate(vlaues):
                print(f"========{val}=>{vlaues[val]}")
        
        else :
            print("================testing=====================")
            for i, val in enumerate(vlaues):
                print(f"========{val}=>{vlaues[val]}")

    def make_full_image(self,imagename):
        re_totals = natsorted(glob.glob(self.log_dir+imagename+'*'))

        sample = skimage.io.imread(re_totals[0])
        
        width,_,_ = sample.shape
        interval = int(1024/width)
        re_t = []
        re_total = []

        for i in range(len(re_totals)):
            img = skimage.io.imread(re_totals[i])
            re_total.append(img)
            if (i+1)%(int(interval*interval))==0:
                re_t.append(np.array(re_total))
                re_total = []
        re_total = np.array(re_t)

        new_image = []
        # new_image = dict()
        for i in range(len(re_total)):
            himag =[]
            one_image=re_total[i]
            
            for j in range(len(one_image)//interval):
                full_image = cv2.hconcat([one_image[j*interval+num] for num in range(interval)])
                # full_image=cv2.hconcat([one_image[j*4],one_image[j*4+1],one_image[j*4+2],one_image[j*4+3]])
                himag.append(full_image)
                if j==0:
                    continue
                elif j%(interval-1) == 0:
                    new=np.array(himag)

                    full_image2=cv2.vconcat([new[num] for num in range(interval)])
                    # full_image2=cv2.vconcat([new[0],new[1],new[2],new[3]])
                    # new_image.update({'full_image'+str(i):full_image2})
                    new_image.append(full_image2)

                
        imsave(self.log_dir+imagename+'_full_image.tif',np.array(new_image))

    def save_csv_file(self,Class,name):
        import pandas
        # for num,name in enumerate(Class):
        df = pd.DataFrame(Class,columns =['back','body','dend','axon'])
        
        # df = pd.DataFrame(Class,columns =['back','forward'])
        df.to_csv(self.log_dir +str(name)+'.csv')
        
        
        


    # def make_full_image(self,imageDir):
    #     re_totals = natsorted(glob.glob(imageDir))

    #     re_t = []
    #     re_total = []
    #     for i in range(len(re_totals)):
    #         img = skimage.io.imread(re_totals[i])
    #         re_total.append(img)
    #         if (i+1)%16==0:
    #     #         print(np.array(re_total).shape)
    #             re_t.append(np.array(re_total))
    #             re_total = []
    #     re_total = np.array(re_t)

    #     new_image =[]
    #     for i in range(len(re_total)):
    #         himag =[]
    #         one_image=re_total[i]
    #         for j in range(len(one_image)//4):
    #             full_image=cv2.hconcat([one_image[j*4],one_image[j*4+1],one_image[j*4+2],one_image[j*4+3]])
    #             himag.append(full_image)
    #             if j==0:
    #                 continue
    #             elif j%3 == 0:
    #                 new=np.array(himag)
    #                 full_image2=cv2.vconcat([new[0],new[1],new[2],new[3]])
    #                 new_image.append(full_image2)
    #     final = dict()
    #     full_image = np.array(new_image)
    #     for i in range(len(full_image)):
    #         final.update(['full_image'+str(i),full_image[i]])

    #     save_images(self,final,0,True)