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
from logger import Logger
from metrics import *
import argparse
import torch.autograd as autograd

from HED import HED
from RCF import RCF

#from fusenet import ICNet 

# from DenseCRFLoss import DenseCRFLoss
from custom_transforms import denormalizeimage
from custom_module import *


class config:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Process some integers')
        parser.add_argument('--knum', help='Select Dataset')
        parser.add_argument('--gpu', default='0',help='comma separated list of GPU(s) to use.',type=str)
        parser.add_argument('--weight_decay',default=1e-8,help='set weight_decay',type=float)
        parser.add_argument('--weight',default=100,help='set Adaptive weight',type=float)
        parser.add_argument('--start_lr',default=3e-3, help='set of learning rate', type=float)
        parser.add_argument('--end_lr',default=3e-6,help='set fo end learning rate',type=float)
        parser.add_argument('--paralle',default=False,help='GPU paralle',type=bool)
        parser.add_argument('--scheduler',default='Cosine',help='select schduler method',type=str)
        parser.add_argument('--epochs',default=201,help='epochs',type=int)
        parser.add_argument('--out_class',default=4,help='set of output class',type=int)
        parser.add_argument('--changestep',default=10,help='change train to valid',type=int)
        parser.add_argument('--pretrain',default=False,help='load pretrained',type=bool)

        parser.add_argument('--optimname',default='Adam',help='select schduler method',type=str)
        parser.add_argument('--datatype',default='uint16_wise', type=str)

        parser.add_argument('--mask_trshold',default=0.3,help='set fo end learning rate',type=float)

        parser.add_argument('--labmda',default=0.1,help='set fo end learning rate',type=float)

        #preprocessing 
        parser.add_argument('--use_median', default=False, action='store_true',help='make binary median image')

        parser.add_argument('--patchsize', default=512, help='patch_size',type=int)
        parser.add_argument('--stride', default=60,help='stride',type=int)
        parser.add_argument('--uselabel', default=False, action='store_true',help='make binary median image')
        parser.add_argument('--oversample', default=True, action='store_false',help='oversample')


        parser.add_argument('--use_train', default=False, action='store_true',help='make binary median image')
        parser.add_argument('--partial_recon', default=False, action='store_true',help='make binary median image')
        parser.add_argument('--class_weight', default=False, action='store_true',help='make binary median image')

        #loss
        parser.add_argument('--ADRMSE',default=False, action='store_true',help='set A daptive_RMSE')
        parser.add_argument('--ADCE',default=False,help='set Adaptive_RMSE',type=bool)
        parser.add_argument('--RECON', default=False, action='store_true',help='set reconstruction loss')
        parser.add_argument('--TVLOSS',default=False,help='set total variation',type=bool)
        parser.add_argument('--Gaborloss',default=False,action='store_true',help='set Gaborloss')
        parser.add_argument('--SKloss',default=False,help='set SKloss',type=bool)
        parser.add_argument('--RECONGAU',default=False, action='store_true',help='set reconstruction guaian loss')
        parser.add_argument('--RCE',default=False, action='store_true',help='set ReverseCross entropy')
        parser.add_argument('--NCE',default=False, action='store_true',help='set Normalized Cross entropy')

        parser.add_argument('--BCE',default=False, action='store_true',help='set Normalized Cross entropy')

        parser.add_argument('--NCDICE',default=False, action='store_true',help='set Normalized Cross entropy')

        parser.add_argument('--deleteall',default=False, action='store_true',help='set Adaptive_RMSE')


        parser.add_argument('--back_filter',default=True, action='store_false',help='set reconstruction guaian loss')
        parser.add_argument('--premask',default=False, action='store_true',help='set reconstruction guaian loss')
        parser.add_argument('--clamp',default=False, action='store_true',help='set reconstruction guaian loss')

        parser.add_argument('--Aloss',default='RMSE',help='select ADRMSE loss',type=str)
        parser.add_argument('--Rloss',default='RMSE',help='select reconstruction loss',type=str)
        parser.add_argument('--Gloss',default='RMSE',help='select Garborloss',type=str)
        parser.add_argument('--Sloss',default='RMSE',help='select SKloss',type=str)
        parser.add_argument('--RGloss',default='RMSE',help='select reconsturction guassian loss',type=str)
        parser.add_argument('--NCDICEloss',default=1.5,help='select reconsturction guassian loss',type=float)


        parser.add_argument('--modelname',default='newunet_compare',help='select Garborloss',type=str)
        parser.add_argument('--activename',default='sigmoid',help='select active',type=str)

        parser.add_argument('--DISCRIM',default=False,help='set discriminate',type=bool)

    return args = parser.parse_args()

def main(args): 
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    #set devices
    print(torch.cuda.is_available(),'torch.cuda.is_available()')
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'else')

    #seed seeting
    resetseed(random_seed=2020)

    #inport network
    model = init_model(args,device)
    
    #select dataset
    Dirlist = select_data(args)

    #cross validation
    trainset,validset = divide_kfold(Dirlist,k=args.foldnum,dataname=args.data_name,cross_validation == args.cross_validation)

    #import dataset
    MyDataset = make_dataset(trainset,validset,args)

    #select loss
    loss_list,lossname = select_loss(args)
    
    # logger 
    main_path, valid_path = make_path(args)

    #set log
    logger = Logger(main_path,valid_path+lossname,delete=args.deleteall)

    # continuous training
    if os.path.exists(logger.log_dir+"lastsave_models{}.pth"):
        checkpoint = torch.load(logger.log_dir +"lastsave_models{}.pth")
        gen.load_state_dict(checkpoint['gen_model'])

    #import trainer
    Trainer = Trainer(model, Mydataset,loss_list,optimizer,scheduler,logger,args)
    
    Trainer.train()
    Trainer.valid()
    Trainer.test()


if __name__ == '__main__': 
    args = config
    main(config)


num_workers = 16
learning_rate = args.start_lr
end_rate = args.end_lr
paralle = args.paralle
use_scheduler = args.scheduler
epochs = args.epochs
eval_channel = args.out_class
classnum = args.out_class
knum = args.knum
weight_decay = args.weight_decay
changestep = args.changestep
use_median = args.use_median

#set loss
adrmse = args.ADRMSE
adce = args.ADCE
recon = args.RECON
total_var_loss = args.TVLOSS
discrim = args.DISCRIM
gabor_loss = args.Gaborloss
recon_gau = args.RECONGAU

print(adce,'adce')
print(adrmse,'adrmse')
print(recon_gau,'recon_gau')
print(recon,'recon')
print(gabor_loss,'gabor_loss')
print('==============================')
print(args.RCE,'RCE')
print(args.NCE,'NCE')
print(args.NCDICE,'NCDICE')
print(args.BCE,'BCE')


#set evluaution 
best_epoch, best_axon, best_dend, best_axon_recall, F1best = 0,0,0,0,0
from torchsummary import summary

print(knum,'=============')
foldnum = 10
betavalue = 2
bi_value = args.mask_trshold
tv_value = 0.0001
labmda = args.labmda
labmda2 = args.labmda
use_label = args.uselabel
phase = 'train'
model = args.modelname
data_name = args.datatype
# save_name = 'seg_and_adaptiveRMSE_reconloss_TV2_'
median = MedianPool2d() 
active = args.activename
use_active = args.Aloss
use_active2 = args.Rloss
use_active3 = args.Gloss
partial_recon = args.partial_recon

load_pretrain = args.pretrain
deepsupervision = False
trainning = args.use_train
testing = True

if testing == True:
    #testing
    print("==========testing===============")

    # if os.path.exists(path+"lastsave_models{}.pth"):
    checkpoint = torch.load(path +"lastsave_models{}.pt")
    gen.load_state_dict(checkpoint['gen_model'])
    # print(checkpoint['epochs'],'213232323232323232323232323')
    gen.eval()
    # torch.save({"gen_model":gen.state_dict()},
    # path+"new_last_models{}.pt")

    logger = Logger(path,batch_size=batch_size)

    total_IOU = []
    total_F1 = []
    total_Fbeta = []
    total_recall = []
    total_predict = []
    total_clF1 =[]

    image_test = np.array(natsorted(glob(testDir+'*')))
    label_test = np.array(natsorted(glob(tlabelDir+'*')))

    MyDataset =         {'valid' : DataLoader(mydataset_xray(image_valid,label_valid,False,patchwise=False,
                                phase='valid',preprocessing=False,multiple_scale=multiple_scale,
                                patch_size=args.patchsize,stride=args.stride,oversampling = False,
                                dataname = data_name),
                                1, 
                                shuffle = False,
                                num_workers = num_workers)}
                        #         ,
                        # 'test' :  DataLoader(mydataset_xray(image_test,label_test,False,patchwise=False,
                        #         phase = 'valid', preprocessing=preprocessing_, isDir=False,oversampling = False,
                        #         dataname = data_name),
                        #         1, 
                        #         shuffle = False,
                        #         num_workers = num_workers)
    phase = 'valid'
    logger.changedir(str(phase)+'_resultbest')
    
    total_chanelwise,valid_predictionmap,valid_predicts,total_input,total_label = [],[],[],[],[]
    valid_precision = []
    for i, batch in tqdm.tqdm(enumerate(MyDataset[phase])):
        _input, _label = batch[0].to(device), batch[1].to(device)
        class_label = Variable(batch[2]).to(device)
        with torch.no_grad():
            predict,prediction_map = gen(_input)

            precision = predict
            final_predict = predict
            precision_chanelwise = precision
            total_clF1.append(soft_cldice_loss(predict,_label,k=2))
            predict = torch.argmax(predict,dim=1).cpu().numpy()

            if _label.dim() == 4:
                evaluator.add_batch(torch.argmax(_label,dim=1).cpu().numpy(),predict)
            elif _label.dim() == 3:
                evaluator.add_batch(_label.cpu().numpy(),predict)
            if _label.dim() == 4: 
                precision = precision.unsqueeze(2).cpu().numpy()
                normalizedImg = np.zeros((1024, 1024))
                precision = cv2.normalize(precision,  normalizedImg, 0, 65535 , cv2.NORM_MINMAX)
                
                prediction_map = F.sigmoid(prediction_map.unsqueeze(2)).cpu().numpy()
                print(prediction_map.max(),prediction_map.min())
                prediction_map = cv2.normalize(prediction_map,  normalizedImg, 0, 255 , cv2.NORM_MINMAX)
                print("???????????????????????????????????????????????")
                print(path)

                v_la = decode_segmap(_label.cpu().detach().numpy().astype('uint16'),nc=4,name='full_4')
                #load mask not propagation
                
                
                one_img = torch.zeros_like(_input)
                zero_img = torch.ones_like(_input)
                mask_img = torch.where(_input>bi_value,zero_img,one_img)
                
                if use_median == True:
                    mask_img = median(mask_img).cpu().numpy()
                else : 
                    mask_img = mask_img.cpu().numpy()
                print(mask_img.shape,'mask')
                _inputs = _input
                _input = _input.detach().cpu().numpy()
                save_stack_images = { 'FINAL_input':_input,'mask_input':mask_img,
                                    'precision':precision.astype('uint16'),'prediction_map':prediction_map.astype('uint16')}
                # if active == 'sigmoid':
                precision_chanelwise = precision_chanelwise.unsqueeze(1).cpu().numpy() * 255
                print(precision_chanelwise.shape,'precision_chanelwise.shape',precision_chanelwise.max(),precision_chanelwise.min())
                precision_chanelwise[:,:,1:2],precision_chanelwise[:,:,2:3] = precision_chanelwise[:,:,2:3],precision_chanelwise[:,:,1:2]  
                save_stack_images.update({'precision_channel':precision_chanelwise[:,:,1:4].astype('uint8')})
                # result_crf=np.array(decode_segmap(result_crf,name='full_4'))

                print(predict.shape,'22')
                pre_body=decode_segmap(predict,nc=4,name='full_4')[:,np.newaxis]
                print(pre_body.shape,'33')
                # print(precision[:,2].max())
                # save_stack_images.update({'final_predict_20':precision[:,2].cpu().numpy()*10})
                save_stack_images.update({'final_predict':pre_body})
                save_path=logger.save_images(save_stack_images,i)
            elif _label.dim() == 5: 
                total_label.append(_label.detach().cpu().numpy())
                total_input.append(_input.detach().cpu().numpy())
                valid_predicts.append(predict)
                valid_predictionmap.append(prediction_map.detach().cpu().numpy())
                valid_precision.append(precision.detach().cpu().numpy())
                print(pre_body.shape,'33')

            #select class
            IOU,Class_IOU,wo_back_MIoU = evaluator.Mean_Intersection_over_Union()
            Acc_class,Class_ACC,wo_back_ACC = evaluator.Pixel_Accuracy_Class()
            Class_precision, Class_recall,Class_F1score = evaluator.Class_F1_score()
            _, _,Class_Fbetascore = evaluator.Class_Fbeta_score(beta=betavalue)
            # print(Class_precision.shape,'23232')
            # pre_body=decode_segmap(ch_channel(predict),nc=4,name='full_4')
            # pre_body=decode_segmap(ch_channel(predict),nc=4,name='full_4')
            
            total_IOU.append(Class_IOU)
            total_F1.append(Class_F1score)
            total_Fbeta.append(Class_Fbetascore)
            total_recall.append(Class_recall)
            total_predict.append(Class_precision)
            # if 

    if _label.dim() == 5: 
        
        normalizedImg = np.zeros((1024, 1024))
        total_label = np.array(total_label)
        total_label = cv2.normalize(total_label,  normalizedImg, 0, 255 , cv2.NORM_MINMAX).astype('uint8')
        v_la = np.concatenate((total_label[:,:,0],total_label[:,:,1],total_label[:,:,2],total_label[:,:,3]),axis=-3)
        v_la = np.swapaxes(make_full_image(v_la),1,2)
        print(v_la.shape,'11111111111111111111111111111111111')

        total_input = np.array(total_input)
        _input = make_full_image(total_input)[0]
        print(_input.shape,'_input')
        
        valid_predicts = np.array(valid_predicts)
        predict = make_full_image(valid_predicts)[0]
        print(predict.shape,'np.array(valid_predicts).shape')
        # ful_v_la = decode_segmap(_ful_label.cpu().detach().numpy(),name='full_4')
        pre_body = decode_segmap(predict)

        zero_img = np.zeros_like(_input)
        one_img = np.ones_like(_input)
        # mask_ = torch.where(_input>bi_value,zero_img,one_img)
        print(_input.max(),'_input.max()',_input.min())
        mask_ = np.where(_input>0.1,zero_img,one_img)
        
        # _input = _input>bi_value
        _input = cv2.normalize(_input,  normalizedImg, 0, 255 , cv2.NORM_MINMAX)

        mask_ = np.transpose(mask_,(0,2,1,3,4)).astype('uint8') *255.
        pre_body = np.transpose(pre_body,(0,1,2,3,4))
        _input = np.transpose(_input,(0,2,1,3,4))

        prediction_map = make_full_image(np.array(valid_predictionmap)) 
        precision = make_full_image(np.array(valid_precision))
        print(prediction_map.shape,'np.array(prediction_map).shape')
        print(precision.shape,'np.array(precision).shape')

        prediction_map = np.concatenate((prediction_map[:,:,0],prediction_map[:,:,1],prediction_map[:,:,2],prediction_map[:,:,3]),axis=-3)
        print(prediction_map.shape,'np.array(prediction_map).shape')
        prediction_map = cv2.normalize(prediction_map,  normalizedImg, 0, 65535 , cv2.NORM_MINMAX)[0]
        prediction_map = prediction_map[:,:,np.newaxis]

        precision = np.concatenate((precision[:,:,0],precision[:,:,1],precision[:,:,2],precision[:,:,3]),axis=-3)
        precision = cv2.normalize(precision,  normalizedImg, 0, 255 , cv2.NORM_MINMAX)[0]
        precision = precision[:,:,np.newaxis]
        precision_chanelwise = precision 

        print(prediction_map.shape,'prediction_mal')
        print(precision.shape,'123123123')

        # precision
        # torchvision.utils.save_images(1-results_all,)
        save_stack_images = {'mask_':mask_.astype('uint8'),'v_la':v_la.astype('uint8'),'_input':_input.astype('uint16'),
                            'precision':precision.astype('uint8'),'prediction_map':prediction_map.astype('uint16')}
                            
        save_stack_images.update({'pre_body':pre_body})

        save_path=logger.save_images(save_stack_images,i)

    print(np.array(total_clF1))
    
    
    for samplelabel,samplepredcit in zip(total_label,valid_predicts):
        evaluator.add_batch(np.argmax(samplelabel,axis=1),samplepredcit)
        IOU,Class_IOU,wo_back_MIoU = evaluator.Mean_Intersection_over_Union()
        Acc_class,Class_ACC,wo_back_ACC = evaluator.Pixel_Accuracy_Class()
        Class_precision, Class_recall,Class_F1score = evaluator.Class_F1_score()
        _, _,Class_Fbetascore = evaluator.Class_Fbeta_score(beta=betavalue)

        print(Class_IOU)
        print(Class_F1score)
        print(Class_precision)
        print(Class_recall)
        
        
        
    # logger.make_full_4_image(imagename='final_predict')
    logger.save_csv_file(np.array(total_IOU),name='total_IOU')
    logger.save_csv_file(np.array(total_F1),name='total_F1')
    logger.save_csv_file(np.array(total_Fbeta),name='total_Fbeta')
    logger.save_csv_file(np.array(total_recall),name='valid_total_recall')
    logger.save_csv_file(np.array(total_predict),name='valid_total_precision')
    logger.save_csv_file(np.array(total_clF1),name='valid_total_clF1')
    # logger.make_full_4_image(imagename='post_image')
            # # print(result_crf)
            # result_crf = np.transpose(result_crf,[0,2,1])
            # # print(v_input[0].shape)
            
            # skimage.io.imsave(path+"pre_body"+"_"+str(epoch)+".png",np.transpose(pre_body[num],[1,2,0]))    
            # skimage.io.imsave(path+"labe_"+"_"+str(epoch)+".png",v_la[num])
            # skimage.io.imsave(path+"img"+"_"+str(epoch)+".png",np.transpose(v_input[num].detach().cpu().numpy(),[1,2,0]))
            # skimage.io.imsave(path+"result_crf"+"_"+str(epoch)+".png",result_crf)
        


