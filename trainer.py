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
import skimage
class Trainer:
    total_train_iter = 0
    total_valid_iter = 0
    
    best_epoch = 0 
    best_axon_recall = 0
    
    
    def __init__(self,model, Mydataset,loss_list,logger,args,device):
        
        self.model = model 
        self.Mydataset = Mydataset
        self.logger = logger
        self.loss_list = loss_list
        
        self.epochs = args.epochs
        self.args = args
        self.device = device

        # LR 
        self.optimizer = optim.Adam(self.model.parameters(),lr=args.start_lr,weight_decay=args.weight_decay)    
        # evaluator
        self.evaluator = Evaluator(args.out_class)
        # scheuler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,2,T_mult=2,eta_min=args.end_lr)

        self.t_loss = AverageMeter(args.out_class)
        self.recon_loss = AverageMeter(args.out_class)

        self.early_stopping = EarlyStopping(patience = 20, verbose = True,save_path=self.logger.log_dir)

    def train_one_epoch(self,epoch,phase,iteration_n):
        print(f"{epoch}/{self.epochs}epochs,IR=>{self.get_lr(self.optimizer)},best_epoch=>{self.best_epoch},phase=>{phase}")
        print(f"==>{self.logger.log_dir}<==")

        self.model.train()
        self.t_loss.reset()
        self.recon_loss.reset()
        self.evaluator.reset()
        phase = 'train'
        for i, batch in enumerate(tqdm(self.Mydataset['train'])):
            
            self._input, self._label = batch[0].to(self.device), Variable(batch[1].to(self.device))
            self.mask_ = Variable(batch[2]).to(self.device)
            self.optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            
            ##### train with source code #####
                
            self.predict,self.prediction_map=self.model(self._input)
            

            
            loss = self.loss_list['mainloss'](self.predict,self._label,self._input,phase)

            if self.scheduler.__class__.__name__ != 'ReduceLROnPlateau':
                self.scheduler.step()
            
            # self.m_predict = self.mask_
            # self.predict = torch.nn.functional.softmax(self.predict,dim=1)  
            # self.m_predict,_ =self.model(self.mask_)
            # loss += torch.mean(torch.pow(torch.abs(self.m_predict-self.predict),2))

            loss += torch.mean(torch.abs(self.m_predict-self.predict))
            if self.args.RECON:
                recon_loss,self.sum_output,self.back_output = self.loss_list['reconloss'](self.predict,self.mask_,self._label,self.args.activation)
                self.recon_loss.update(recon_loss.detach().item(),self._input.size(0))
                loss += recon_loss  
                
            self.t_loss.update(loss.detach().item(),self._input.size(0))
            self.evaluator.add_batch(torch.argmax(self._label,dim=1).cpu().numpy(),torch.argmax(self.predict,dim=1).cpu().numpy())
            result_dicts = self.evaluator.update()
            
            self.t_loss.reset_dict()
            
            total_score = self.t_loss.update_dict(result_dicts)
            # if phase == 'train': 
            

            loss.backward()
                        

            self.optimizer.step()
            self.total_train_iter += 1
            iteration_n = self.total_train_iter
            self.logger.list_summary_scalars(total_score,iteration_n,phase)
            self.logger.summary_scalars({'loss':self.t_loss.avg},iteration_n,'Losses',phase)
            self.logger.summary_scalars({'IR':self.get_lr(self.optimizer)},iteration_n,tag='IR',phase=phase)
            self.logger.summary_scalars({'reconloss':self.recon_loss.avg},iteration_n,'RLosses',phase)
                
        self.deployresult(epoch,phase)
        self.logger.print_value(result_dicts,phase)
        return result_dicts
        
    def valid_one_epoch(self,epoch,phase,iteration_n):
        print(f"{epoch}/{self.epochs}epochs,IR=>{self.get_lr(self.optimizer)},best_epoch=>{self.best_epoch},phase=>{phase}")
        print(f"==>{self.logger.log_dir}<==")

        self.model.eval()  
        self.t_loss.reset()
        self.recon_loss.reset()
        self.evaluator.reset()
        phase = 'valid'
        with torch.no_grad():
            
            for i, batch in enumerate(tqdm(self.Mydataset['valid'])):
                
                self._input, self._label = batch[0].to(self.device), Variable(batch[1].to(self.device))
                self.mask_ = Variable(batch[2]).to(self.device)
                    
                self.predict,self.prediction_map=self.model(self._input)
                
                loss = self.loss_list['mainloss'](self.predict,self._label,self._input,phase)

                # self.m_predict = self.mask_
                # self.m_predict,_ =self.model(self.mask_)
                # loss += torch.mean(torch.pow(torch.abs(self.m_predict-self.predict),2))

                # self.predict = torch.nn.functional.softmax(self.predict,dim=1)  
                # loss += torch.mean(torch.pow(torch.abs(self.maks_predic-self._label),2))

                if self.args.RECON:
                    recon_loss,self.sum_output,self.back_output = self.loss_list['reconloss'](self.predict,self.mask_,self._label,self.args.activation)
                    self.recon_loss.update(recon_loss.detach().item(),self._input.size(0))
                    loss += recon_loss  
                    
                self.t_loss.update(loss.detach().item(),self._input.size(0))
                self.evaluator.add_batch(torch.argmax(self._label,dim=1).cpu().numpy(),torch.argmax(self.predict,dim=1).cpu().numpy())
                result_dicts = self.evaluator.update()
                
                self.t_loss.reset_dict()
                
                total_score = self.t_loss.update_dict(result_dicts)
                    # if phase == 'train': 
                             
        self.logger.list_summary_scalars(total_score,iteration_n,phase)
        self.logger.summary_scalars({'loss':self.t_loss.avg},iteration_n,'Losses',phase)
        self.logger.summary_scalars({'IR':self.get_lr(self.optimizer)},iteration_n,tag='IR',phase=phase)
        self.logger.summary_scalars({'reconloss':self.recon_loss.avg},iteration_n,'RLosses',phase)
        self.total_valid_iter += 1
        iteration_n = self.total_valid_iter
            # self.total_valid_iter += 1
        # self.deployresult(i,phase)
        self.deployresult(epoch,phase)
        self.logger.print_value(result_dicts,phase)
        return result_dicts

    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def deployresult(self,epoch,phase):
        
        print(f"label shape : {self._label.shape},featuer shape:,{self.prediction_map.shape},self.predict shape:{self.predict.shape}")
        save_stack_images = {'_label':self._label.detach().cpu().numpy() * 255. ,
                            '_input':self._input.detach().cpu().numpy() * 65535.,
                            'prediction_map':self.prediction_map.detach().cpu().numpy(),
                            'predict_channe_wise':self.predict.detach().cpu().numpy(),
                            'mask_predict':(self.m_predict.detach().cpu().numpy()*65535.).astype('uint16'),
                            'mask':self.mask_.detach().cpu().numpy().astype('uint8')*255.}

        if self.args.RECON:
            save_stack_images.update({'sum_ouput':self.sum_output.detach().cpu().numpy() * 65535.,
                                        'back_output':self.back_output.detach().cpu().numpy() * 65535.})

        # self._input = cv2.normalize(self._input,  normalizedImg, 0, 65535 , cv2.NORM_MINMAX)
        save_stack_images = self.logger.make_stack_image(save_stack_images)
        self.predict = self.predict.detach().cpu().numpy() 

        pre_body = decode_segmap(np.argmax(self.predict,axis=1))
        pre_body = np.transpose(pre_body,(0,2,3,1))
        
        save_stack_images['_label'] = save_stack_images['_label'].astype('uint8')
        save_stack_images['prediction_map'] = save_stack_images['prediction_map'].astype('uint8')
        save_stack_images.update({'prediction_result':pre_body.astype('uint8')})
        
        self.logger.summary_images(save_stack_images,epoch,phase)
        if phase == 'valid' :
            if epoch % self.args.changestep == 0:
                self.logger.save_images(save_stack_images,epoch)
        elif phase == 'test':
            self.logger.save_images(save_stack_images,epoch)

    def save_model(self,epoch):
        if  self.evaluator.Class_F1score[3] > self.best_axon_recall :
            torch.save({"self.model_model":self.model.state_dict(),
                    "optimizer":self.optimizer.state_dict(),
                    "epochs":epoch},
                    self.logger.log_dir+"bestsave_models{}.pt")
            print('save!!!')
            best_axon = self.evaluator.Class_IOU[3]
            self.best_axon_recall = self.evaluator.Class_F1score[3]
            F1best = self.evaluator.Class_F1score[3]
            self.best_epoch = epoch
        torch.save({"self.model_model":self.model.state_dict(),
                "optimizerG":self.optimizer.state_dict(),
                "epochs":epoch},
                self.logger.log_dir+"lastsave_models{}.pt")

    def testing(self,phase):
        print(f"/{self.epochs}epochs,IR=>{self.get_lr(self.optimizer)},best_epoch=>{self.best_epoch},phase=>{phase}")
        print(f"==>{self.logger.log_dir}<==")
        self.t_loss.reset()
        self.recon_loss.reset()
        self.evaluator.reset()

        for i, batch in enumerate(tqdm(self.Mydataset['valid'])):
            
            self._input, self._label = Variable(batch[0]).to(self.device), Variable(batch[1]).to(self.device)
            self.mask_ = Variable(batch[2]).to(self.device)
            with torch.no_grad():
            ##### train with source code #####
                self.predict,self.prediction_map=self.model(self._input)                    
                # self.t_loss.update(loss.detach().item(),self._input.size(0))
                # self.predict= (1-self.mask_) * self.predict
                self.mas = 1-self.mask_.cpu().numpy()[:,0].astype('int')
                sample_predict = torch.argmax(self.predict,dim=1).cpu().numpy() * self.mas
                print(self.mas.shape,sample_predict.shape,'asdfasdfasd')
                self.evaluator.add_batch(torch.argmax(self._label,dim=1).cpu().numpy(),sample_predict)
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
            phase = 'train'
            
            result_dict = self.train_one_epoch(epoch,phase,self.total_train_iter)
            result_dict = self.valid_one_epoch(epoch,phase,self.total_valid_iter)
            self.save_model(epoch)
            if epoch % 5 == 0:
                self.early_stopping(self.t_loss.avg, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    def test(self): 
        print("start testing")
        self.model.eval()
        phase = 'test'
        result_dict = self.testing(phase)
 