import numpy as np
import skimage 
import os ,tqdm , random
from glob import glob
import torch
import torch.nn.functional as F
import yaml

from torch import nn, optim
from torchvision import models ,transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch import Tensor
#custom set#
from my_network import *
from neuron_util import *
from neuron_util import channel_wise_segmentation
from my_custom_loss import *
import config
from mydataset import mydataset_2d
from mydataset_xray import mydataset_xray
from my_network3d import ResidualUNet3D
from self.logger import Logger
from metrics import *
import argparse
import torch.autograd as autograd

from HED import HED
from RCF import RCF

class Trainer():
    total_train_iter = 0

    def __init__(model, Mydataset,loss_list,optimizer,scheduler,args)
        self.args = args
        self.model = model 
        self.Mydataset = Mydataset
        self.self.logger = self.logger
        self.loss_list = loss_list
        self.best_axon_recall = 0

        # LR 
        self.optimizer = optim.Adam(self.model.parameters(),lr=args.start_lr)    
        # evaluator
        self.evaluator = Evaluator(eval_channel)
        # scheuler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_mult=2,T_0=2,eta_min=args.end_lr)

        self.summary_loss = AverageMeter()

    def train_one_epoch(self,phase):
        print(f"{epoch}/{epochs}epochs,IR=>{get_lr(optimizerG)},best_epoch=>{best_epoch},phase=>{phase}")
        print(f"==>{path}<==")
        self.summary_loss.reset()

        for i, batch in enumerate(tqdm.tqdm(MyDataset[phase])):

            self._input, self._label = Variable(batch[0]).to(device), Variable(batch[1]).to(device)
            mask_ = Variable(batch[2]).to(device)         
            
            optimizerG.zero_grad()
            torch.autograd.set_detect_anomaly(True)

            ##### train with source code #####
            with torch.set_grad_enabled(phase == 'train'):

                self.predict,self.prediction_map=self.model(self._input)

                loss = lossdict['mainloss'](self.predict,self._label,mask_)
                self.total_loss.append(criterion(self.predict,self._label,mask_))
                self.summary_loss.update(loss,self._input.size(0))          

                if args.RECON:
                    recon_loss,self.sum_output,self.back_output = lossdict['reconloss'](self.predict,mask_,self._label,active)
                    loss += recon_loss  
                    seg_loss += CE_loss.item() * self._input.size(0) 
                # print(recon_loss,CE_loss)
                if phase == 'train':
                    loss.backward()
                    optimizerG.step()
                    scheduler.step()
                
                t_loss = seg_loss + cls_loss + tvl

                self.predict = torch.argmax(self.predict,dim=1).cpu().numpy()

            if self._label.dim() == 4 or self._label.dim() == 5:
                self.evaluator.add_batch(torch.argmax(self._label,dim=1).cpu().numpy(),self.predict)
            elif self._label.dim() == 3:
                self.evaluator.add_batch(self._label.cpu().numpy(),self.predict)
            
            result_dicts = self.evaluator.update()
            result_dicts.update{'loss',loss}

            self.logger.print_value(result_dicts,phase)
            #update self.logger
            total_train_iter += 1

            return result_dicts
    

    def deployresult(self):
        v_la = self._label.unsqueeze(0)
        v_la = self._label.unsqueeze(2)
        v_la = v_la.cpu().detach().numpy() * 50.
        print(v_la.max(),v_la.min())
        # v_la = cv2.normalize(v_la,  normalizedImg, 0, 255 , cv2.NORM_MINMAX).astype('uint8')
        zero_img = torch.zeros_like(self._input)
        one_img = torch.ones_like(self._input)
        mask_ = mask_.detach().cpu().numpy()

        self._input = self._input.detach().cpu().numpy()
        self._input = cv2.normalize(self._input,  normalizedImg, 0, 65535 , cv2.NORM_MINMAX)
        pre_body = decode_segmap(self.predict)[:,np.newaxis]
        
        self.prediction_map = self.prediction_map.unsqueeze(2).cpu().numpy()
        self.prediction_map = cv2.normalize(self.prediction_map,  normalizedImg, 0, 255 , cv2.NORM_MINMAX)

        print(f"label shape : {v_la.shape},featuer shape:,{self.prediction_map.shape},self.prediction shape:{precision.shape}")
        print(f"pre_body shape : {pre_body.shape}")

        save_stack_images = {'mask_':mask_.astype('uint8'),'v_la':v_la.astype('uint8'),'self._input':self._input.astype('uint16'),
                            'precision':precision.astype('uint8'),'self.prediction_map':self.prediction_map.astype('uint16')}

        if self.args.recon == True:
            self.sum_output = self.sum_output.detach().cpu().numpy()
            self.back_output = self.back_output.detach().cpu().numpy()
            self.sum_output = cv2.normalize(self.sum_output,  normalizedImg, 0, 65535 , cv2.NORM_MINMAX)
            self.back_output = cv2.normalize(self.back_output,  normalizedImg, 0, 65535 , cv2.NORM_MINMAX)
            
            if self.sum_output.ndim == 5: 
                self.sum_output = np.transpose(self.sum_output,(0,2,1,3,4))
                self.back_output = np.transpose(self.back_output,(0,2,1,3,4))
            save_stack_images.update({'self.sum_output':self.sum_output.astype('uint16'),'self.back_output':self.back_output.astype('uint16')})    
            save_stack_images.update({'pre_body':pre_body})

        self.logger.save_images(save_stack_images,epoch)

    def save_model(self)
        if  self.evaluator.Class_F1score[3] > best_axon_recall :
            torch.save({"self.model_model":self.model.state_dict(),
                    "optimizer":self.optimizer.state_dict(),
                    "epochs":self.epoch},
                    path+"bestsave_models{}.pt")
            print('save!!!')
            best_axon = self.evaluator.Class_IOU[3]
            best_axon_recall = self.evaluator.Class_F1score[3]
            F1best = self.evaluator.Class_F1score[3]
            best_epoch = epoch
                    
        torch.save({"self.model_model":self.model.state_dict(),
                "optimizerG":self.optimizer.state_dict(),
                "epochs":self.epoch},
                path+"lastsave_models{}.pt")

    def train(self,epochs): 

        print("start trainning!!!!")
        for epoch in range(epochs):
            self.evaluator.reset()
            phase = 'train'
            self.model.train() 
            result_dict = self.train_one_epoch(phase)
            
            if epoch %changestep == 0:
                phase = 'valid'
                self.model.eval()            
                result_dict = self.train_one_epoch(phase)
                self.deployresult()
                self.save_model()


            IOU_scalar = dict()
            precision_scalar = dict()
            recall_scalr = dict()
            Fbetascore_scalar = dict()
            
            for i in range(classnum):
                IOU_scalar.update({'IOU_'+str(i):result_dict['IOU'][i]})
                precision_scalar.update({'precision_'+str(i):result_dict['precision'][i]})
                recall_scalr.update({'recall_'+str(i):result_dict['recall'][i]})
                F1score_scalar.update({'F1_'+str(i):result_dict['F1']i]})
                
            self.logger.summary_scalars(IOU_scalar,total_train_iter,'IOU',phase)
            self.logger.summary_scalars(precision_scalar,total_train_iter,'precision',phase)
            self.logger.summary_scalars(recall_scalr,total_train_iter,'recall',phase)
            self.logger.summary_scalars(F1score_scalar,total_train_iter,'F1',phase)
            self.logger.summary_scalars(result_dict['loss'],total_train_iter,'Losses',phase)
            self.logger.summary_scalars({'IR':get_lr(optimizerG)},total_train_iter,tag='IR',phase=phase)
            
                
                # self.logger.summary_scalars({'IR':get_lr(optimizerG)},epoch,'IR',phase)
                
    # def deploy3dresult(self):
    #     # _,_,cha,zsize,xysize,yxsize = v_la.shape
    #     totalself._label = np.array(totalself._label)
    #     totalself._label = cv2.normalize(totalself._label,  normalizedImg, 0, 255 , cv2.NORM_MINMAX).astype('uint8')
    #     v_la = np.concatenate((totalself._label[:,:,0],totalself._label[:,:,1],totalself._label[:,:,2],totalself._label[:,:,3]),axis=-3)
    #     v_la = np.swapaxes(make_full_image(v_la),1,2)
    #     print(v_la.shape,'11111111111111111111111111111111111')

    #     totalself._input = np.array(totalself._input)
    #     self._input = make_full_image(totalself._input)[0]
    #     print(self._input.shape,'self._input')
        
    #     valid_self.predicts = np.array(valid_self.predicts)
    #     self.predict = make_full_image(valid_self.predicts)[0]
    #     print(self.predict.shape,'np.array(valid_self.predicts).shape')
    #     # ful_v_la = decode_segmap(_fulself._label.cpu().detach().numpy(),name='full_4')
    #     pre_body = decode_segmap(self.predict)

    #     zero_img = np.zeros_like(self._input)
    #     one_img = np.ones_like(self._input)
    #     # mask_ = torch.where(self._input>bi_value,zero_img,one_img)
    #     print(self._input.max(),'self._input.max()',self._input.min())
    #     mask_ = np.where(self._input>0.1,zero_img,one_img)
        
    #     # self._input = self._input>bi_value
    #     self._input = cv2.normalize(self._input,  normalizedImg, 0, 255 , cv2.NORM_MINMAX)

    #     mask_ = np.transpose(mask_,(0,2,1,3,4)).astype('uint8') *255.
    #     pre_body = np.transpose(pre_body,(0,1,2,3,4))
    #     self._input = np.transpose(self._input,(0,2,1,3,4))

    #     self.prediction_map = make_full_image(np.array(valid_self.predictionmap)) 
    #     precision = make_full_image(np.array(valid_precision))
    #     print(self.prediction_map.shape,'np.array(self.prediction_map).shape')
    #     print(precision.shape,'np.array(precision).shape')

    #     self.prediction_map = np.concatenate((self.prediction_map[:,:,0],self.prediction_map[:,:,1],self.prediction_map[:,:,2],self.prediction_map[:,:,3]),axis=-3)
    #     print(self.prediction_map.shape,'np.array(self.prediction_map).shape')
    #     self.prediction_map = cv2.normalize(self.prediction_map,  normalizedImg, 0, 65535 , cv2.NORM_MINMAX)[0]
    #     self.prediction_map = self.prediction_map[:,:,np.newaxis]

    #     precision = np.concatenate((precision[:,:,0],precision[:,:,1],precision[:,:,2],precision[:,:,3]),axis=-3)
    #     precision = cv2.normalize(precision,  normalizedImg, 0, 255 , cv2.NORM_MINMAX)[0]
    #     precision = precision[:,:,np.newaxis]
    #     precision_chanelwise = precision 