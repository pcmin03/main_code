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

class Trainer():
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
        self.optimizer = optim.Adam(self.model.parameters(),lr=args.start_lr)    
        # evaluator
        self.evaluator = Evaluator(args.out_class)
        # scheuler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_mult=2,T_0=2,eta_min=args.end_lr)

        self.t_loss = AverageMeter(args.out_class)
        self.recon_loss = AverageMeter(args.out_class)

    def train_one_epoch(self,epoch,phase):
        print(f"{epoch}/{self.epochs}epochs,IR=>{self.get_lr(self.optimizer)},best_epoch=>{self.best_epoch},phase=>{phase}")
        print(f"==>{self.logger.log_dir}<==")
        self.t_loss.reset()
        self.recon_loss.reset()

        for i, batch in enumerate(tqdm(self.Mydataset[phase])):
            
            self._input, self._label = Variable(batch[0]).to(self.device), Variable(batch[1]).to(self.device)
            mask_ = Variable(batch[2]).to(self.device)
            self.optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)

            ##### train with source code #####
            with torch.set_grad_enabled(phase == 'train'):

                self.predict,self.prediction_map=self.model(self._input)

                loss = self.loss_list['mainloss'](self.predict,self._label,mask_)
                
                self.t_loss.update(loss.detach().item(),self._input.size(0))          

                if self.args.RECON:
                    recon_loss,self.sum_output,self.back_output = self.loss_list['reconloss'](self.predict,mask_,self._label,self.args.activation)
                    self.recon_loss.update(recon_loss.detach().item(),self._input.size(0))
                    loss += recon_loss  
                    
                # print(recon_loss,CE_loss)
                self.evaluator.add_batch(torch.argmax(self._label,dim=1).cpu().numpy(),torch.argmax(self.predict,dim=1).cpu().numpy())
                result_dicts = self.evaluator.update()
                #update self.logger
                self.t_loss.reset_dict()
                total_score = self.t_loss.update_dict(result_dicts)

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    self.logger.list_summary_scalars(total_score,self.total_train_iter,phase)
                    self.logger.summary_scalars({'loss':loss.detach().item()},self.total_train_iter,'Losses',phase)
                    self.logger.summary_scalars({'IR':self.get_lr(self.optimizer)},self.total_train_iter,tag='IR',phase=phase)
                    self.total_train_iter += 1
                    
                elif phase == 'valid': 
                    
                    self.logger.list_summary_scalars(total_score,self.total_valid_iter,phase)
                    self.logger.summary_scalars({'loss':loss.detach().item()},self.total_valid_iter,'Losses',phase)
                    self.logger.summary_scalars({'IR':self.get_lr(self.optimizer)},self.total_valid_iter,tag='IR',phase=phase)
                    self.total_valid_iter += 1
        
        self.logger.print_value(result_dicts,phase)
        return result_dicts
    
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def deployresult(self,epoch):
        print(f"label shape : {self._label.shape},featuer shape:,{self.prediction_map.shape},self.predict shape:{self.predict.shape}")
        
        save_stack_images = {'_label':self._label * 255. ,'_input':self._input * 65535,
                            'prediction_map':self.prediction_map,
                            'predict_channe_wise':self.predict}

        # self._input = cv2.normalize(self._input,  normalizedImg, 0, 65535 , cv2.NORM_MINMAX)

        save_stack_images = self.logger.make_stack_image(save_stack_images)
        pre_body = decode_segmap(torch.argmax(self.predict,dim=1).detach().cpu().numpy())
        pre_body = np.transpose(pre_body,(0,2,3,1))

        save_stack_images['_label'] = save_stack_images['_label'].astype('uint8')
        save_stack_images['prediction_map'] = save_stack_images['prediction_map'].astype('uint16')
        save_stack_images.update({'prediction_result':pre_body.astype('uint8')})
        
        self.logger.save_images(save_stack_images,epoch)
        self.logger.summary_images(save_stack_images,epoch)
        
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

    def train(self): 

        print("start trainning!!!!")
        for epoch in range(self.epochs):
            self.evaluator.reset()
            phase = 'train'
            self.model.train() 
            result_dict = self.train_one_epoch(epoch,phase)
            
            if epoch % self.args.changestep == 0:
                phase = 'valid'
                self.model.eval()            
                result_dict = self.train_one_epoch(epoch,phase)
                self.deployresult(epoch)
                self.save_model(epoch)

    def test(self): 
        print("start testing")
        self.model.eval()
        phase = 'test'
        result_dict = self.train_one_epochs(epoch,phase)
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