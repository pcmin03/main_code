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
from neuron_network import *

class model():
    def __init__(self,model_path='./neuron_model/',nd2file='#12_2.nd2',gpu_id =0):
        
        #make new direction 
        self.model_path = model_path
        self.network_path = self.model_path +'model/'
        self.newpath = self.model_path +'result/'
        print('----- Make_save_Dir-------------')
        if not os.path.exists(self.newpath):
            os.makedirs(self.newpath)

        #set GPU
        self.cuda0 = torch.device('cuda:'+str(gpu_id)if torch.cuda.is_available() else "else")

        #change to Tensor
        self.L_transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        to_tensor = transforms.ToTensor()

        #select nd2 file
        file_root = 'sample_file/'
        
        self.patchimg,self.mitoimg = preprocessing(self.model_path+file_root+nd2file)

        self.save_dict = dict()

    def segmentataion(self,post_proces=True):
        #load segmentataion model
        
        segmentataion_model = Segmentataion_resnet101unet().to(self.cuda0)
        checkpoint = torch.load(self.network_path+"segmentataion_model.pth")
        segmentataion_model.load_state_dict(checkpoint['gen_model'])
        segmentataion_model.eval()
        
        full_img = []
        origin_img = []
        for img in tqdm.tqdm(self.patchimg):
            # if img.ndim == 3:
            #     img = np.max(img, axis=0)
            #     img = skimage.color.gray2rgb(img)
            origin_img.append(img)
            
            #change torch datatype
            img = self.L_transform(img.astype('uint8'))
            img = img.cuda().unsqueeze(0)
            
            out = segmentataion_model(img)
            predict=out.float()
            v_pre = decode_segmap(ch_channel(predict),name='full')

            
            if post_proces == True:
                #===================post processing DRF ==============#
                postprocessor = DenseCRF(
                                    iter_max=10,
                                    pos_xy_std=1,
                                    pos_w=3,
                                    bi_xy_std=67,
                                    bi_rgb_std=3,
                                    bi_w=4 )
                
                num = 0
                prob = F.softmax(out,dim=1)[num].detach().cpu().numpy()
                result_crf = postprocessor(img[num].permute(1,2,0).cpu().numpy().astype(np.ubyte), prob)
                result_crf = np.argmax(result_crf, axis=0)
                result_crf = np.array(decode_segmap(result_crf,name='full'))
                result_crf = np.transpose(result_crf,[0,2,1])
                #===================post processing DRF ==============#
                full_img.append(result_crf)
            
            else:
                v_pre = np.transpose(v_pre[0],[1,2,0])
                full_img.append(v_pre)
                
        self.full_image = make_full_image(np.array(full_img))
        
        self.save_dict.update({'full_image':make_full_image(np.array(full_img)),
                            'orgine_img':make_full_image(np.array(origin_img))})
    
    def detection(self,detection_network=False):
        ### detection part ###
        self.mitoimg = self.mitoimg.astype('uint8')
        self.mi = self.mitoimg
        binary = self.mitoimg > 0
       
        
        # devide label
        dend_channel = divide_getlabel(self.full_image,binary,'dendrite')
        axon_channel = divide_getlabel(self.full_image,binary,'axon')
        full_channel = divide_getlabel(self.full_image,binary,'full')
        
        bi = binary* full_channel
        if detection_network == True:
            self.mitoimg = skimage.color.gray2rgb(self.mitoimg)
            # load detection networks
            detection_model = get_model_instance_segmentation(2).cuda()
            detection_checkpoint = torch.load(self.network_path+"detection_model.pth")
            detection_model.load_state_dict(detection_checkpoint['gen_model'])
            detection_model.eval()

            # change to Tensor
            full_mito_image = self.L_transform(self.mitoimg)
            full_mito_image = full_mito_image.cuda().unsqueeze(0)
            
            dend_mitoimg = self.L_transform(self.mitoimg * skimage.color.gray2rgb(dend_channel))
            axon_mitoimg = self.L_transform(self.mitoimg * skimage.color.gray2rgb(axon_channel))

            dend_mitoimg = dend_mitoimg.cuda().unsqueeze(0)
            axon_mitoimg = axon_mitoimg.cuda().unsqueeze(0)

            # pass the network
            ful_predict = detection_model(full_mito_image)
            dend_predict = detection_model(dend_mitoimg)
            axon_predict = detection_model(axon_mitoimg)

            zero = np.zeros((1024,1024))
            self.mitoimg = cv2.normalize(self.mitoimg,zero,0,255,cv2.NORM_MINMAX).astype('uint8')
            self.ful_detectimage,ful_boxes = detecion_box(self.mitoimg, ful_predict)
            self.dend_detectimage,dend_boxes = detecion_box(self.mitoimg, dend_predict)
            self.axon_detectimage,axon_boxes = detecion_box(self.mitoimg, axon_predict)
            
        else:
            #using binart connectied component
            ful_boxes, self.ful_detectimage = locateComponents(full_channel*255)
            dend_boxes, self.dend_detectimage = locateComponents(dend_channel*255)
            axon_boxes, self.axon_detectimage =locateComponents(axon_channel*255)
            
            self.mitoimg = skimage.color.gray2rgb((binary * full_channel).astype('uint8') * self.mitoimg)
            
            self.ful_detectimage = self.ful_detectimage +self.mitoimg
            self.dend_detectimage = self.dend_detectimage + self.mitoimg
            self.axon_detectimage = self.axon_detectimage + self.mitoimg
        
        detect_image = {'mitoimg':self.mi,'full_detect':self.ful_detectimage,
                        'dend_detect':self.dend_detectimage,'axon_detect':self.axon_detectimage}
            
        self.save_dict.update(detect_image)
        return {'dend_boxes':np.array(dend_boxes), 'axon_boxes':np.array(axon_boxes)}

    def save_image(self):
        for num,img in enumerate(self.save_dict):
            skimage.io.imsave(self.newpath+str(img)+'.tif',self.save_dict[img])
            

    def save_csv_file(self,boxes):
        import pandas
        for num,name in enumerate(boxes):
            df = pd.DataFrame(boxes[name],columns =['xmin','ymin','xmax','ymax'])
            df.to_csv(self.newpath+str(name)+'.csv')
def main():
    
    seg_detec = model(model_path='./neuron_model/',nd2file='#7.nd2') #self,model_path='../neuron_model/',nd2file='#12_2.nd2'
    seg_detec.segmentataion(post_proces=True)
    boxes = seg_detec.detection(detection_network=True)
    seg_detec.save_image()
    seg_detec.save_csv_file(boxes)
    
if __name__ =='__main__':
    main()
