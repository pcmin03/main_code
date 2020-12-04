import numpy as np
import os ,tqdm , random
from glob import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch import Tensor
#custom set#

from utils.matrix import Evaluator, AverageMeter
from utils.neuron_util import decode_segmap
import torch.autograd as autograd

from utils.pytorchtools import EarlyStopping

from trainer import Trainer
from models.unet3d.metrics import MeanIoU

class Trainer3d(Trainer):
    
    def __init__(self,model, Mydataset,loss_list,logger,args,device):
        self.model = model 
        self.Mydataset = Mydataset
        self.logger = logger
        self.loss_list = loss_list
        
        self.epochs = args.epochs
        self.args = args
        self.device = device
        # LR 
        self.optimizer = optim.Adam(self.model.parameters(),lr=args.start_lr)    
        # evaluator
        self.evaluator = Evaluator(args.out_class)
        self.evaluator3d = MeanIoU()
        # scheuler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_mult=2,T_0=2,eta_min=args.end_lr)

        self.t_loss = AverageMeter(args.out_class)
        self.recon_loss = AverageMeter(args.out_class)

        self.early_stopping = EarlyStopping(patience = 50, verbose = True,save_path=self.logger.log_dir)

    def train_one_epoch(self,epoch,phase):
        print(f"{epoch}/{self.epochs}epochs,IR=>{self.get_lr(self.optimizer)},best_epoch=>{self.best_epoch},phase=>{phase}")
        print(f"==>{self.logger.log_dir}<==")
        self.t_loss.reset()
        self.recon_loss.reset()
        self.evaluator.reset()

        for i, batch in enumerate(tqdm(self.Mydataset[phase])):
            
            self._input, self._label = batch[0].to(self.device), Variable(batch[1].to(self.device))
            mask_ = Variable(batch[2]).to(self.device)
            self.optimizer.zero_grad()
            
            # torch.autograd.set_detect_anomaly(True)
            
            ##### train with source code #####
            with torch.set_grad_enabled(phase == 'train'):
                
                self.predict,self.prediction_map=self.model(self._input)
                loss = self.loss_list['mainloss'](self.predict,self._label,mask_)
                if self.args.RECON:
                    recon_loss,self.sum_output,self.back_output = self.loss_list['reconloss'](self.predict,mask_,self._label,self.args.activation)
                    self.recon_loss.update(recon_loss.detach().item(),self._input.size(0))
                    loss += recon_loss  
                    
                self.t_loss.update(loss.detach().item(),self._input.size(0))          
                
                # print(recon_loss,CE_loss)
                self.evaluator.add_batch(torch.argmax(self._label,dim=1).cpu().numpy(),torch.argmax(self.predict,dim=1).cpu().numpy())
                result_dicts = self.evaluator.update(self.evaluator3d(self.predict,torch.argmax(self._label,dim=1)))
                #update self.logger
                self.t_loss.reset_dict()
                total_score = self.t_loss.update_dict(result_dicts)
                
                if self.scheduler.__class__.__name__ != 'ReduceLROnPlateau':
                    self.scheduler.step()

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()
                    self.logger.list_summary_scalars(total_score,self.total_train_iter,phase)
                    self.logger.summary_scalars({'loss':self.t_loss.avg},self.total_train_iter,'Losses',phase)
                    self.logger.summary_scalars({'IR':self.get_lr(self.optimizer)},self.total_train_iter,tag='IR',phase=phase)
                    self.logger.summary_scalars({'reconloss':self.recon_loss.avg},self.total_train_iter,'RLosses',phase)               
                    self.total_train_iter += 1
                    
                elif phase == 'valid': 
                    
                    self.logger.list_summary_scalars(total_score,self.total_valid_iter,phase)
                    self.logger.summary_scalars({'loss':self.t_loss.avg},self.total_valid_iter,'Losses',phase)
                    self.logger.summary_scalars({'reconloss':self.recon_loss.avg},self.total_valid_iter,'RLosses',phase)
                    self.logger.summary_scalars({'IR':self.get_lr(self.optimizer)},self.total_valid_iter,tag='IR',phase=phase)
                    self.total_valid_iter += 1
                    self.logger.print_value(result_dicts,phase)
                    
                    self.t_loss.stack_result([self.predict,self.prediction_map,self._label,self._input])
                
        self.logger.print_value(result_dicts,phase)
        return result_dicts
    
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def deployresult3d(self,epoch,phase):
        
        self.predict,self.prediction_map,self._label,self._input= self.t_loss.predicts,self.t_loss.precision,self.t_loss.total_label,self.t_loss.total_input
        # print(f"label shape : {self._label.shape},featuer shape:,{self.prediction_map.shape},self.predict shape:{self.predict.shape}")
        
        save_stack_images = {'_label':np.array(self._label) * 255. ,
                            '_input':np.array(self._input) * 65535,
                            'prediction_map':np.array(self.prediction_map),
                            'predict_channe_wise':np.array(self.predict)}
        
        save_stack_images = self.logger.make_full_image(save_stack_images)
        save_stack_images['predict_channe_wise'] = decode_segmap(np.argmax(save_stack_images['predict_channe_wise'],axis=1))
        save_stack_images = self.logger.make_stack_image(save_stack_images)
        
        # pre_body = np.transpose(pre_body,(0,2,3,1))

        save_stack_images['_label'] = save_stack_images['_label'].astype('uint8')
        save_stack_images['prediction_map'] = save_stack_images['prediction_map'].astype('uint16')
        # save_stack_images.update({'prediction_result':pre_body.astype('uint8')})
        
        self.logger.summary_images(save_stack_images,epoch,phase)
        if phase == 'valid' or  phase== 'test':
            self.logger.save_images(save_stack_images,epoch)

    def testing(self,phase):
        print(f"/{self.epochs}epochs,IR=>{self.get_lr(self.optimizer)},best_epoch=>{self.best_epoch},phase=>{phase}")
        print(f"==>{self.logger.log_dir}<==")
        self.t_loss.reset()
        self.recon_loss.reset()
        self.evaluator.reset()

        for i, batch in enumerate(tqdm(self.Mydataset['valid'])):
            
            self._input, self._label = Variable(batch[0]).to(self.device), Variable(batch[1]).to(self.device)
            mask_ = Variable(batch[2]).to(self.device)
            with torch.no_grad():
            ##### train with source code #####
                self.predict,self.prediction_map=self.model(self._input)                    
                # self.t_loss.update(loss.detach().item(),self._input.size(0))          
    
                # print(recon_loss,CE_loss)
                self.evaluator.add_batch(torch.argmax(self._label,dim=1).cpu().numpy(),torch.argmax(self.predict,dim=1).cpu().numpy())
                result_dicts = self.evaluator.update()
                #update self.logger
                self.t_loss.reset_dict()
                total_score = self.t_loss.update_dict(result_dicts)
                
                self.deployresult(i,phase)
                self.logger.list_summary_scalars(total_score,i,phase)
                self.logger.print_value(result_dicts,phase)

        return result_dicts
    
    def train(self): 

        print("start trainning!!!!")
        for epoch in range(self.epochs):
            self.evaluator.reset()
            phase = 'train'
            self.model.train() 
            result_dict = self.train_one_epoch(epoch,phase)
            self.deployresult(epoch,phase)

            if epoch % self.args.changestep == 0:
                phase = 'valid'
                self.evaluator.reset()
                self.model.eval()            
                result_dict = self.train_one_epoch(epoch,phase)
                self.deployresult3d(epoch,phase)
                self.save_model(epoch)
                self.early_stopping(self.t_loss.IOU_scalar['IOU_1'], self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    def test(self): 
        print("start testing")
        self.model.eval()
        phase = 'test'
        result_dict = self.testing(phase)
                # self.logger.summary_scalars({'IR':get_lr(optimizerG)},epoch,'IR',phase)
                