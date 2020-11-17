import numpy as np
import skimage ,cv2
import os , sys, argparse, glob
import pandas as pd
import tqdm,shutil
import torch

from torch import nn, optim
from torchvision import transforms

from skimage.io import imsave
from natsort import natsorted
from neuron_util import * 

import segmentation_models_pytorch as smp

class model():
    def __init__(self,model_path='./neuron_model/',nd2file='#16_2.nd2',gpu_id =0):
        
        #make new direction 
        self.model_path = model_path
        self.network_path = self.model_path +'model/'
        self.newpath = self.model_path +'result/'
        if not os.path.exists(self.newpath):
            os.makedirs(self.newpath)

        #set GPU
        self.cuda0 = torch.device('cuda:'+str(gpu_id)if torch.cuda.is_available() else "else")
    
        #select nd2 file
        file_root = 'sample_file/'
        
        print('----- Preprocessing loding-------------')
        self.patchimg,self.mitoimg = preprocessing(self.model_path+file_root+nd2file)
        self.save_dict = dict()
        
    def pretrain_unet(self,in_channel,out_channel=4):    
        return smp.Unet('resnet34',in_channels=in_channel,classes=out_channel,activation='softmax')

    def segmentataion(self,post_proces=False):
        #load pretrained segmentataion model
        print('----- Segmentation model loding-------------')
        segmentataion_model = self.pretrain_unet(1,4).to(self.cuda0)

        
        checkpoint = torch.load(self.network_path+"lastsave_models{}.pth")
        segmentataion_model.load_state_dict(checkpoint['gen_model'])
        segmentataion_model.eval()

        img = self.patchimg
        origin_img = img
        #change torch datatype
        origin_img = np.array(origin_img)
        
        if img.dtype == 'uint16':
            self.L2_transform = transforms.Compose([
                    transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)).unsqueeze(0)),
                    transforms.Normalize([0.5], [0.5])])
        else :
            self.L2_transform = transforms.Compose([
                    transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)).unsqueeze(0)),
                    transforms.Normalize([0], [255])])
        img = self.L2_transform(origin_img)
        
        #add batch axis
        img = img.cuda().unsqueeze(0)
        
        #evaluation
        out = segmentataion_model(img)
        predict=out.float()
        sample = ch_channel(predict)
        skimage.io.imsave('./test.tif',sample[0])
        v_pre = decode_segmap(ch_channel(predict),name='full')
        print(img.shape,predict.shape,origin_img.dtype)
            
        # update images
        print(v_pre.shape)
        v_pre = np.transpose(v_pre[0],[1,2,0])
        print(v_pre.shape)
        self.full_image = v_pre
        self.save_dict.update({'result_image':np.array(self.full_image),
                            'origine_img':np.array(origin_img)})
    
    def detection(self,detection_network=False):
        print('----- detection loding-------------')
        
        ### detection part ###
        self.mi = self.mitoimg
        binary = self.mitoimg > 0
       
        
        # devide label
        dend_channel = divide_getlabel(self.full_image,binary,'dendrite')
        axon_channel = divide_getlabel(self.full_image,binary,'axon')
        full_channel = divide_getlabel(self.full_image,binary,'full')
        
        bi = binary* full_channel

        #using binart connectied component
        print(full_channel.max())
        ful_boxes, self.ful_detectimage = locateComponents(full_channel * 255)
        dend_boxes, self.dend_detectimage = locateComponents(dend_channel* 255)
        axon_boxes, self.axon_detectimage =locateComponents(axon_channel* 255)
        
        self.mitoimg = skimage.color.gray2rgb((binary * full_channel).astype('uint8') * self.mitoimg)
        
        self.ful_detectimage = self.ful_detectimage+ self.mitoimg
        self.dend_detectimage = self.dend_detectimage + self.mitoimg
        self.axon_detectimage = self.axon_detectimage + self.mitoimg
        
        detect_image = {'mitoimg':self.mi,'full_detect':self.ful_detectimage,
                        'dend_detect':self.dend_detectimage,'axon_detect':self.axon_detectimage}
            
        self.save_dict.update(detect_image)
        return {'dend_boxes':np.array(dend_boxes), 'axon_boxes':np.array(axon_boxes),
                'full_boxes':np.concatenate((np.array(axon_boxes),np.array(dend_boxes)))}

    def save_image(self):
        for num,img in enumerate(self.save_dict):
            print(self.save_dict[img].dtype)
            skimage.io.imsave(self.newpath+str(img)+'.tif',self.save_dict[img])
            

    def save_csv_file(self,boxes):
        import pandas
        for num,name in enumerate(boxes):
            df = pd.DataFrame(boxes[name],columns =['xmin','ymin','xmax','ymax'])
            df.to_csv(self.newpath+str(name)+'.csv')


def main():
    # MFF overexpression/60X_2.nd2
    seg_detec = model(model_path='./neuron_model/',nd2file='control/60x_4.nd2') #self,model_path='../neuron_model/',nd2file='#12_2.nd2'
    seg_detec.segmentataion(post_proces=True)
    boxes = seg_detec.detection(detection_network=False)
    seg_detec.save_image()
    seg_detec.save_csv_file(boxes)
    
if __name__ =='__main__':
    main()
