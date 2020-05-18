import os, shutil, random, glob
import numpy as np
import skimage,cv2
from torch.utils.tensorboard import SummaryWriter
from skimage.io import imsave
from natsort import natsorted
import pandas as pd

class Logger(object):
    ### save dictionary ###
    def __init__(self, log_dir,batch_size,delete=False,num='0',name='weight'):
        self.log_dir = log_dir
        self.batch_size =batch_size
        # self.board_dir = self.log_dir+name+'board/'
        merge_path = './merge_path/board'+name+'/'+name+str(num)+'/'
        if not os.path.exists(self.log_dir):
            print('----- Make_save_Dir-------------')
            os.makedirs(self.log_dir)
            print(self.log_dir)
        if delete == True:

            print('----- remove_Dir-------------')
            shutil.rmtree(self.log_dir+'*',ignore_errors=True)
            shutil.rmtree(merge_path,ignore_errors=True)
        
        
        self.writer = SummaryWriter(merge_path)
        
    def summary_images(self,images_dict,step):
        ### list of image ###
        for i, img in enumerate(images_dict):
            self.writer.add_image(str(img),images_dict[img],step)

    def summary_scalars(self,scalar_dict,step,tag='loss'):
        ### list of scaler ###
        for i, scalar in enumerate(scalar_dict):

            if tag in scalar:
                self.writer.add_scalar(str(tag)+'/'+str(scalar),scalar_dict[scalar],step)

            elif 'loss' in scalar:
                self.writer.add_scalar('loss/'+str(scalar),scalar_dict[scalar],step)
            
    def summary_3dimages(self,images_dict,step):
        ### list of stack_images ###
        for i, img in enumerate(images_dict):
            self.writer.add_images(str(img),images_dict[img],step)

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
        num = random.randint(0, self.batch_size-1)
        num = 0
        save_dir = self.log_dir
        
        for i, img in enumerate(images_dict):
            
            #change NCHW to NHWC save stack_image of TIF file
            #3d image
            if images_dict[img][num].ndim == 4:
                result_image = np.transpose(images_dict[img][num],[1,2,3,0])
            #2d image
            elif images_dict[img][num].ndim ==3:
                result_image = np.transpose(images_dict[img][num],[1,2,0])

            imsave(save_dir+str(img)+str(step)+'.tif',result_image)

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
            
            
        
    def make_full_3dimage(self,file_name='result'):
        
        save_dir = self.log_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        re_totals = natsorted(glob.glob(save_dir+file_name+'*'))
        re_t = []
        re_total = []
        final_dict = dict()
        
        for i in range(len(re_totals)):
        
            img = skimage.io.imread(re_totals[i])
            re_total.append(img)
            if (i+1)%16==0:
        #         print(np.array(re_total).shape)
                re_t.append(np.array(re_total))
                re_total = []
        re_total = np.array(re_t)
        
        new_image =[]
        final = []
        dimaag = []
        for i in range(len(re_total)):
            new_image =[] 
            one_image=re_total[i]
            for k in range(len(one_image[0,:])):
                one_=one_image[:,k]
                himag =[]

                for j in range(len(one_)//4):
                    full_image=cv2.hconcat([one_[j*4],one_[j*4+1],one_[j*4+2],one_[j*4+3]])
                    himag.append(full_image)
                    if j==0:
                        continue
                    elif j%3 == 0:
                        new=np.array(himag)
                        full_image2=cv2.vconcat([new[0],new[1],new[2],new[3]])
                        new_image.append(full_image2)
                new_img = np.array(new_image)
            #     dimage = np.array(dimaag)[0]
                #     print(dimage.shape)
            final = np.transpose(np.expand_dims(new_img,axis=1),[1,4,0,2,3])
            project = np.max(final,axis=2)
            final_dict.update([('final'+str(i),final),('project'+str(i),project)])
        self.save_images(final_dict,0)
    def save_csv_file(self,Class,name):
        import pandas
        # for num,name in enumerate(Class):
        df = pd.DataFrame(Class,columns =['back','body','dend','axon'])
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