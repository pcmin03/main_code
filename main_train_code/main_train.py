import numpy as np
import skimage 
import os ,tqdm, glob , random
import torch
import torch.nn.functional as F
import yaml

from torch import nn, optim
from torchvision import models ,transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

#custom set#
from my_network import *
from neuron_util import *
from neuron_util import channel_wise_segmentation
from my_loss import *
import config
from mydataset import prjection_mydataset,mydataset_2d
from logger import Logger
from metrics import *
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--knum', help='Select Dataset')
parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
parser.add_argument('--weight_decay',default=0.001,help='set weight_decay',type=float)
parser.add_argument('--weight',help='set Adaptive weight',type=int)
# parser.add_argument('--batch',defalut=110,help='set Adaptive weight',type=int)
args = parser.parse_args()


with open('my_config.yml') as f:
    conf = yaml.load(f)


num_workers = 8
learning_rate = 1e-4
end_rate = 1e-6
paralle = False
use_scheduler = 'Cosine'
cross_validation = True
epochs = 201
best_epoch = 0
best_axon = 0
best_dend = 0
betavalue = 2
F1best = 0
best_axon_recall = 0
eval_channel = 4
knum = args.knum
print(knum,'=============')
foldnum = 10
classnum = 4
phase = 'train'
model = 'efficient_nestunet'
# ../unetmodel9Adaptive_weight_decay__0.001/
# name = 'Adaptive_loss_weight_decay'
# name = 'Adaptive_weight_decay_sample'
# name = 'CE_DICE'
# name = 'original'
# name = 'Dice_ignore_index_axon'
# name = 'MT_L1_loss'
# name = 'Adaptive_loss_refine_weight'
# name = 'Adaptive_weight_decay_multiTask'
# name = '_rmse_CE'
# name = 'Dice_ignore_index_axon'
# name = 'CE_ignore_index'
# name = "Adaptive_deepsupervision"
# name = "Ada"

# name = "Adaptive_Full_image_weight"+str(args.weight)

# name = "Adaptive_Full_image_RMSE"
# name = "Adpative_attention_module"+str(args.weight)
name = "Adaptive_Full_image_AdaptiveCE_"+str(args.weight)
# name = "Adaptive_Full_image_AdaptiveCE_distacemap"+str(args.weight)

# name = "Adaptive_Full_image_CustomCE_"+str(args.weight)

# name = "Adaptive_Full_image_CE_Scehduler"+str(args.weight)
# name = "Adaptive_Full_image_custom_CE"+str(args.weight)
# name = "Custom_Adaptive_DistanceMap"+str(args.weight)
# name = "Custom_Adaptive_apply_DistanceMap"+str(args.weight)
# name = "Adaptive_CE"+str(args.weight)
# name = "Adaptive_loss"
# name = "Adaptive_CE+RMSE"
# name = 'Adaptive_loss_Multi_task'
# name = 'Adaptive_weight_decay__'
# name = 'Axon_Adpative_loss'
# name = 'CE_ignore_index_dend'
# unetmodel2Adaptive_loss_weight_decay0.001
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# print(args.weight,'11111111')
changestep = 20
worst = 10
# weight = args.weight_decay
# path = '../nested_CV_nest_net_loss'+str(knum)+'/'
path = '../'+str(model)+'model'+str(knum)+str(name)+'/'
#set devices
device = torch.device('cuda:'+str(args.gpu)if torch.cuda.is_available() else "else")

if model =='unet':
    batch_size = 110
    gen = pretrain_unet(4).to(device)

elif model=='res_unet':
    batch_size = 22
    gen = Segmentataion_resnet101unet().to(device)

elif model == 'efficent_unet':
    batch_size = 40
    gen = pretrain_efficent_unet().to(device)

elif model == 'nest_unet':
    batch_size = 25
    gen = NestedUNet(input_channels=1,out_channels=1,deepsupervision=deepsupervision).to(device)

elif model == 'AttU_Net':
    batch_size = 30
    gen = AttU_Net().to(device)

elif model == 'efficient_nestunet':
    batch_size = 30
    gen = efficient_nestunet(1,4).to(device)

elif model == 'Multi_Unet':
    batch_size = 30
    gen = Multi_UNet().to(device)
    

# path = '../'+str(model)+'model'+str(knum)+str(name)+str(weight)+'/'

# path  = '../res_unet_loss_single_nested_CV_unet'+str(knum)+'/'
# path = '../nest_unetnew_model3/'
# path = '../nest_unetmodel4/'
# path = '../res_unetmode'

# path = '../nested_CV_nest_net_loss'+str(knum)+'/'

# path = '../pre_result_unet_loss_single2_summary_model_crf/'
# path = '../res_unetfull_model3_new/'
# path = '../pre_result_unet_loss_single2_summary_model_NestedUNet/'

# path ='../nest_unetfull_model3_new/'
deepsupervision = True
trainning = True
testing = True
use_postprocessing = True
deleteall = False
load_pretrain = False
patchwise = False
multi = False
multi_ = False
#set data path 
imageDir= '../new_project_original_image/'
labelDir = '../new_project_label_modify_image/'
testDir ='../test_image/'
tlabelDir = '../test_label/'

# if model == 'res_unet':
#     batch_size = 5
   

# if cross_validation == False:
#     # Dataloader
#     MyDataset = {'train': DataLoader(prjection_mydataset(imageDir,labelDir,True,cross_validation),
#                                     batch_size, 
#                                     shuffle = True,
#                                     num_workers = num_workers),
#                 'test' : DataLoader(prjection_mydataset(testDir,tlabelDir,False,cross_validation),
#                                     1, 
#                                     shuffle = True,
#                                     num_workers = num_workers)}


#set paralle


if paralle == True:
    gen = torch.nn.DataParallel(gen, device_ids=[0,1])



optimizerG = optim.Adam(gen.parameters(),lr=learning_rate)

if use_scheduler == 'Cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerG,100,T_mult=1,eta_min=end_rate)

if deleteall==True:
    logger = Logger(path,batch_size=batch_size,delete=deleteall,num=str(knum),name=model+name)

else:
    logger = Logger(path,batch_size=batch_size,delete=False,num=str(knum),name=model+name)
    #training
    



#set loss
# criterion = Custom_WeightedCrossEntropyLossV2().to(device)

# criterion = Custom_Adaptive().to(device)
# criterion = Custom_CE(args.weight).to(device)
criterion = Custom_Adaptive_DistanceMap(args.weight,distanace_map=False).to(device)
# criterion = Custom_Adaptive_DistanceMap(args.weight).to(device)
# criterion = RMSELoss().to(device)

# criterion = BinaryDiceLoss().to(device)
# criterion = Custom_Adaptive_RMSE().to(device)
# criterion = DiceLoss(ignore_index=1).to(device)
# criterion = Custom_Adaptive().to(device)
# criterion = Custom_Adaptive_DistanceMap().to(device)
# criterion = Custom_WeightedCrossEntropyLossV2().to(device)
# criterion = nn.cross_entropy().to(device)
print(1/(1+int(args.weight)),'1111')
# criterion0 = nn.BCELoss(torch.from_numpy(np.array(1/(1+(args.weight))))).to(device)
# criterion1 = nn.BCELoss().to(device)

# criterion2 = nn.BCELoss().to(device)
# criterion3 = nn.BCELoss().to(device)

# criterion = nn.CrossEntropyLoss().to(device)
# criterion = Custom_Adaptive_DistanceMap().to(device)
#set matrix score
evaluator = Evaluator(eval_channel)
inversevaluator = Evaluator(eval_channel)

vevaluator = Evaluator(eval_channel)
inversvevaluator = Evaluator(eval_channel)



image,labels = divide_kfold(imageDir,labelDir,k=foldnum,name='test')
train_num, test_num = 'train'+str(knum), 'test'+str(knum)
# image_valid = image[train_num][-3:]
# label_valid = labels[train_num][-3:]

image_train = image[train_num]
label_train = labels[train_num]
image_valid = image[test_num]
label_valid = labels[test_num]

if trainning == True:

    if cross_validation == True:
        print(f"{image_valid},{label_valid}")
        print(f"{image[train_num]},{labels[train_num]}")
        MyDataset = {'train': DataLoader(mydataset_2d(image_train,label_train,patchwise=patchwise,multi=multi),
                                        batch_size, 
                                        shuffle = True,
                                        num_workers = num_workers),
                    'valid' : DataLoader(mydataset_2d(image_valid,label_valid,False,patchwise=patchwise,phase='test',multi=multi),
                                        1, 
                                        shuffle = False,
                                        num_workers = num_workers)}

    print("start trainning!!!!")
    for epoch in range(epochs):
        
        if epoch %changestep == 0:
            phase = 'valid'
            gen.eval()            
            vevaluator.reset()
            inversvevaluator.reset()
            total_IOU = []
            total_F1 = []
            total_Fbeta = []
            total_recall = []
            total_predict = []

            inversetotal_IOU = []
            inversetotal_F1 = []
            inversetotal_Fbeta = []
            inversetotal_recall = []
            inversetotal_predict = []
        else : 
            phase = 'train'
            gen.train()  
            evaluator.reset()
            inversevaluator.reset()
        print(f"{epoch}/{epochs}epochs,IR=>{get_lr(optimizerG)},best_epoch=>{best_epoch},phase=>{phase}")
        print(f"==>{path}<==")
        for i, batch in enumerate(tqdm.tqdm(MyDataset[phase])):
            
            _input, _label = Variable(batch[0]).to(device), Variable(batch[1]).to(device)
            
            if phase == 'train':
                optimizerG.zero_grad()
                torch.autograd.set_detect_anomaly(True)

                if deepsupervision==True and model == 'nest_unet':   
                    loss = 0
                    back,body,dend,axon = gen(_input)

                    gt=_label
                    # back_label = torch.where(gt==0,torch.ones_like(gt),torch.zeros_like(gt)).view(-1,).float()
                    # body_label = torch.where(gt==1,torch.ones_like(gt),torch.zeros_like(gt)).view(-1,).float()
                    # dend_label = torch.where(gt==2,torch.ones_like(gt),torch.zeros_like(gt)).view(-1,).float()
                    # axon_label = torch.where(gt==3,torch.ones_like(gt),torch.zeros_like(gt)).view(-1,).float()

                    predict = torch.cat((back, body,dend,axon),dim=1).cuda().float()
                    
                    back_loss = criterion(predict,_label)
                    # body_loss = criterion1(torch.sigmoid(body.view(-1,)),body_label)
                    # dend_loss = criterion2(torch.sigmoid(dend.view(-1,)),dend_label)
                    # axon_loss = criterion3(torch.sigmoid(axon.view(-1,)),axon_label)

                    # CE_loss = (back_loss+body_loss+dend_loss+axon_loss)/4
                    CE_loss = back_loss
                    seg_loss_body = CE_loss 
                    seg_loss_body.backward(retain_graph = True)
                    
                    optimizerG.step()
                    
                    evaluator.add_batch(_label.cpu().numpy(),torch.argmax(predict,dim=1).cpu().numpy())
                    IOU,Class_IOU,wo_back_MIoU = evaluator.Mean_Intersection_over_Union()
                    Class_precision, Class_recall,Class_F1score = evaluator.Class_F1_score()
                    _, _,Class_Fbetascore = evaluator.Class_Fbeta_score(beta=betavalue)                    
                        
                else :
                    predict=gen(_input)
                    if classnum == 8:
                        CE_loss = criterion(predict,_label)
                        inversevaluator.add_batch(_label.cpu().numpy(),torch.argmax(predict[:,5:8],dim=1).cpu().numpy())
                        _,inverseClass_IOU,_ = inversevaluator.Mean_Intersection_over_Union()
                        inverseClass_precision, inverseClass_recall,inverseClass_F1score = inversevaluator.Class_F1_score()
                        _, _,inverseClass_Fbetascore = inversevaluator.Class_Fbeta_score(beta=betavalue)
                        CE_loss = criterion(predict,_label)
                        seg_loss_body = CE_loss 
                        seg_loss_body.backward(retain_graph = True)
                        optimizerG.step()
                    
                    else:
                        CE_loss = criterion(predict,_label)
                        # dice_loss = criterion_v2(predict,_label)
                        
                        seg_loss_body = CE_loss 
                        seg_loss_body.backward(retain_graph = True)
                        optimizerG.step()
                    
                        evaluator.add_batch(_label.cpu().numpy(),torch.argmax(predict,dim=1).cpu().numpy())
                        IOU,Class_IOU,wo_back_MIoU = evaluator.Mean_Intersection_over_Union()
                        Class_precision, Class_recall,Class_F1score = evaluator.Class_F1_score()
                        _, _,Class_Fbetascore = evaluator.Class_Fbeta_score(beta=betavalue)                    
                        
            
            else:
                pre_IOU = 0
                pre_ACC = 0
                loss = 0

                with torch.no_grad():
                    
                    
                    if deepsupervision==True and model == 'nest_unet':
                        back,body,dend,axon = gen(_input)
                        gt = _label
                        # back_label = torch.where(gt==0,torch.ones_like(gt),torch.zeros_like(gt)).view(-1,).float()
                        # body_label = torch.where(gt==1,torch.ones_like(gt),torch.zeros_like(gt)).view(-1,).float()
                        # dend_label = torch.where(gt==2,torch.ones_like(gt),torch.zeros_like(gt)).view(-1,).float()
                        # axon_label = torch.where(gt==3,torch.ones_like(gt),torch.zeros_like(gt)).view(-1,).float()

                        predict = torch.cat((back, body,dend,axon),dim=1).cuda().float()
                                                
                        back_loss = criterion(predict,_label)
                        # body_loss = criterion1(torch.sigmoid(body.view(-1,)),body_label)
                        # dend_loss = criterion2(torch.sigmoid(dend.view(-1,)),dend_label)
                        # axon_loss = criterion3(torch.sigmoid(axon.view(-1,)),axon_label)

                        # val_CE_loss = (back_loss+body_loss+dend_loss+axon_loss)/4
                        val_CE_loss = back_loss
                        val_loss = val_CE_loss

                        # vevaluator.add_batch(_label.cpu().numpy(),spre)
                        vevaluator.add_batch(_label.cpu().numpy(),torch.argmax(predict,dim=1).cpu().numpy())
                        IOU,Class_IOU,wo_back_MIoU = vevaluator.Mean_Intersection_over_Union()
                        Acc_class,Class_ACC,wo_back_ACC = vevaluator.Pixel_Accuracy_Class()
                        Class_precision, Class_recall,Class_F1score = vevaluator.Class_F1_score()
                        _, _,Class_Fbetascore = vevaluator.Class_Fbeta_score(beta=betavalue)
                        
                        pre_IOU += Class_IOU
                        pre_ACC += Class_ACC
                        
                        if epoch %changestep ==0:
                            total_IOU.append(Class_IOU)
                            total_F1.append(Class_F1score)
                            total_Fbeta.append(Class_Fbetascore)
                            total_recall.append(Class_recall)
                            total_predict.append(Class_precision)
                        # pre_IOU += Class_IOU
                        # pre_ACC += Class_ACC

                        middle_IOU = Class_IOU
                        middle_ACC = Class_ACC

                        pre_IOU = [class_IOU / 9 for class_IOU in pre_IOU]
                        pre_ACC = [class_IOU / 9 for class_IOU in pre_ACC]
                        # loss /= len(predict)
                        # val_loss = loss
                        # predict = spre

                    else:
                        predict=gen(_input)
                        
                        if classnum == 8:
                            inversevaluator.add_batch(_label.cpu().numpy(),torch.argmax(predict[:,],dim=1).cpu().numpy())
                            _,inverseClass_IOU,_ = inversevaluator.Mean_Intersection_over_Union()
                            inverseClass_precision, inverseClass_recall,inverseClass_F1score = inversevaluator.Class_F1_score()
                            _, _,inverseClass_Fbetascore = inversevaluator.Class_Fbeta_score(beta=betavalue)

                            if epoch %changestep ==0:
                                inversetotal_IOU.append(inverseClass_IOU)
                                inversetotal_F1.append(inverseClass_F1score)
                                inversetotal_Fbeta.append(inverseClass_Fbetascore)
                                inversetotal_recall.append(inverseClass_recall)
                                inversetotal_predict.append(inverseClass_precision)

                            val_CE_loss = criterion(predict,_label)
                        else : 
                            
                            # vevaluator.add_batch(_label.cpu().numpy(),predict)
                            
                            vevaluator.add_batch(_label.cpu().numpy(),torch.argmax(predict[:,0:4],dim=1).cpu().numpy())
                            IOU,Class_IOU,wo_back_MIoU = vevaluator.Mean_Intersection_over_Union()
                            Acc_class,Class_ACC,wo_back_ACC = vevaluator.Pixel_Accuracy_Class()
                            Class_precision, Class_recall,Class_F1score = vevaluator.Class_F1_score()
                            _, _,Class_Fbetascore = vevaluator.Class_Fbeta_score(beta=betavalue)
                            
                            pre_IOU += Class_IOU
                            pre_ACC += Class_ACC
                            if epoch %changestep ==0:
                                total_IOU.append(Class_IOU)
                                total_F1.append(Class_F1score)
                                total_Fbeta.append(Class_Fbetascore)
                                total_recall.append(Class_recall)
                                total_predict.append(Class_precision)

                            val_CE_loss = criterion(predict,_label)
                        # val_dice_loss = criterion_v2(predict,_label)
                        
                            val_loss = val_CE_loss 
                            # val_CE_loss = criterion(predict,_label)
                        
                        
                            middle_IOU = Class_IOU
                            middle_ACC = Class_ACC

                            pre_IOU = [class_IOU / 9 for class_IOU in pre_IOU]
                            pre_ACC = [class_IOU / 9 for class_IOU in pre_ACC]
                
        if  phase == 'train':
            if use_scheduler == 'Cosine':
                scheduler.step(epoch)
            if classnum == 8:
                invesesum_print = {
                                'inverseClass_IOU':inverseClass_IOU,
                                'inverseprecision':inverseClass_precision, 
                                'inverserecall':inverseClass_recall,
                                'inverseF1score':inverseClass_F1score,
                                'inverseFbetascore':inverseClass_Fbetascore}
                logger.print_value(invesesum_print,'train')

            else:
                summary_print = {'seg_loss_body':seg_loss_body,
                                'Class_IOU':Class_IOU,
                                'precision':Class_precision, 
                                'recall':Class_recall,
                                'F1score':Class_F1score,
                                'Fbetascore':Class_Fbetascore}
                logger.print_value(summary_print,'train')
            train_loss = {  'seg_loss_body':seg_loss_body,
                            'CE_loss':CE_loss}
            logger.summary_scalars(train_loss,epoch)
            

        elif phase == 'valid': 

            if classnum == 8:
                inversetest_val = {"inverseClass_IOU":inverseClass_IOU,
                            'inverseprecision':inverseClass_precision, 
                            'inverserecall':inverseClass_recall,
                            'inverseF1score':inverseClass_F1score,
                            'inverseFbetascore':inverseClass_Fbetascore}
                logger.print_value(inversetest_val,'inversetest')
            else :
                test_val = {"pre_IOU":pre_IOU,"pre_ACC":pre_ACC,"Class_IOU":Class_IOU,"Class_ACC":Class_ACC,
                            "wo_back_MIoU":wo_back_MIoU,"wo_back_ACC":wo_back_ACC,
                            'precision':Class_precision, 
                            'recall':Class_recall,
                            'F1score':Class_F1score,
                            "val_loss":val_loss,
                            'Fbetascore':Class_Fbetascore}
                logger.print_value(test_val,'test')

            IOU_scalar = dict()
            precision_scalar = dict()
            recall_scalr = dict()
            F1score_scalar = dict()
            Fbetascore_scalar = dict()
            
            
            inverseIOU_scalar = dict()
            inverseprecision_scalar = dict()
            inverserecall_scalr = dict()
            inverseF1score_scalar = dict()
            inverseFbetascore_scalar = dict()

            for i in range(4):
                IOU_scalar.update({'val_IOU_'+str(i):Class_IOU[i]})
                precision_scalar.update({'val_precision_'+str(i):Class_precision[i]})
                recall_scalr.update({'val_recall_'+str(i):Class_recall[i]})
                F1score_scalar.update({'val_F1_'+str(i):Class_F1score[i]})
                Fbetascore_scalar.update({'val_Fbeta'+str(i):Class_Fbetascore[i]})
                if classnum == 8:
                    inverseIOU_scalar.update({'inverseval_IOU_'+str(i):Class_IOU[i]})
                    inverseprecision_scalar.update({'inverseval_precision_'+str(i):Class_precision[i]})
                    inverserecall_scalr.update({'inverseval_recall_'+str(i):Class_recall[i]})
                    inverseF1score_scalar.update({'inverseval_F1_'+str(i):Class_F1score[i]})
                    inverseFbetascore_scalar.update({'inverseval_Fbeta'+str(i):Class_Fbetascore[i]})
                
            validation_loss = {'val_loss':val_loss,
                                'val_CE_loss':val_CE_loss}

            logger.summary_scalars(IOU_scalar,epoch,'IOU')
            logger.summary_scalars(precision_scalar,epoch,'precision')
            logger.summary_scalars(recall_scalr,epoch,'recall')
            logger.summary_scalars(F1score_scalar,epoch,'F1')
            logger.summary_scalars(Fbetascore_scalar,epoch,'Fbeta')
            logger.summary_scalars(validation_loss,epoch)
            logger.summary_scalars({'IR':get_lr(optimizerG)},epoch,'IR')
            if classnum == 8:
                logger.summary_scalars(inverseIOU_scalar,epoch,'inverseIOU')
                logger.summary_scalars(inverseprecision_scalar,epoch,'inverseprecision')
                logger.summary_scalars(inverserecall_scalr,epoch,'inverserecall')
                logger.summary_scalars(inverseF1score_scalar,epoch,'inverseF1')
                logger.summary_scalars(inverseFbetascore_scalar,epoch,'inverseFbeta')
            

            if  (Class_IOU[3] > best_axon) or (Class_F1score[3] > best_axon_recall) :
                torch.save({"gen_model":gen.state_dict(),
                        "optimizerG":optimizerG.state_dict(),
                        "epochs":epoch},
                        path+"bestsave_models{}.pth")
                print('save!!!')
                best_axon = Class_IOU[3]
                best_axon_recall = Class_F1score[3]
                F1best = Class_F1score[3]
                best_epoch = epoch

                        
            if  epoch %changestep == 0:
                torch.save({"gen_model":gen.state_dict(),
                        "optimizerG":optimizerG.state_dict(),
                        "epochs":epoch},
                        path+"lastsave_models{}.pth")
                # print(.max())
                # if multi_ == True:
                #     pre_body = decode_segmap(torch.argmax(body,dim=1).cpu().numpy(),name='body')
                #     pre_dend = decode_segmap(torch.argmax(dend,dim=1).cpu().numpy(),name='dend')
                #     pre_axon = decode_segmap(torch.argmax(axon,dim=1).cpu().numpy(),name='axon')
                #     pre_body = pre_body + pre_dend +  pre_axon
                #     v_la = decode_segmap(torch.argmax(_label,dim=1).cpu().detach().numpy(),name='full')
                #     _input = _input.detach().cpu().numpy()
                # else : 
                pre_body=decode_segmap(torch.argmax(predict[:,0:4],dim=1).cpu().numpy())
                v_la=decode_segmap(_label.cpu().detach().numpy().astype('uint8'),name='full')
                _input = _input.detach().cpu().numpy()
                    # inversepre_body=decode_segmap(torch.argmax(predict[:,4:8],dim=1).cpu().numpy(),name='inverse')
                    
                save_stack_images = {'pre_body':pre_body,'v_la':v_la,'_input':_input}
                # print(total_IOU)
                logger.save_csv_file(np.array(total_IOU),name='valid_total_IOU')
                logger.save_csv_file(np.array(total_F1),name='valid_total_F1')
                logger.save_csv_file(np.array(total_Fbeta),name='valid_total_Fbeta')
                logger.save_csv_file(np.array(total_recall),name='valid_total_recall')
                logger.save_csv_file(np.array(total_predict),name='valid_total_precision')
                if classnum == 8:
                    logger.save_csv_file(np.array(inversetotal_IOU),name='inversevalid_total_IOU')
                    logger.save_csv_file(np.array(inversetotal_F1),name='inversevalid_total_F1')
                    logger.save_csv_file(np.array(inversetotal_Fbeta),name='inversevalid_total_Fbeta')
                    logger.save_csv_file(np.array(inversetotal_recall),name='inversevalid_total_recall')
                    logger.save_csv_file(np.array(inversetotal_predict),name='inversevalid_total_precision')


                logger.save_images(save_stack_images,epoch)
if load_pretrain == True:
    # if os.path.exists(path+"lastsave_models{}.pth"):
    checkpoint = torch.load(path +"lastsave_models{}.pth")
    gen.load_state_dict(checkpoint['gen_model'])

if testing == True:
    #testing
    print("==========testing===============")

    gen.eval()
    logger = Logger(path,batch_size=batch_size)

    logger.changedir()
    total_IOU = []
    total_F1 = []
    total_Fbeta = []
    total_recall = []
    total_predict = []

    MyDataset = {'test' :   DataLoader(mydataset_2d(image_valid,label_valid,False,patchwise=patchwise,phase='test',multi=multi),
                            1, 
                            shuffle = False,
                            num_workers = num_workers)}
    for i, batch in enumerate(MyDataset['test']):
        _input, _label = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():

            if deepsupervision==True and model == 'nest_unet':
                back,body,dend,axon = gen(_input)
                predict = torch.cat((back, body,dend,axon),dim=1).cuda().float()

            else :
                predict = gen(_input)

            print("???????????????????????????????????????????????")
            print(path)
            # predict = channel_segmap(predict.cpu().numpy())
            # vevaluator.add_batch(_label.cpu().numpy(),predict)
            vevaluator.add_batch(_label.cpu().numpy(),torch.argmax(predict,dim=1).cpu().numpy())
            IOU,Class_IOU,wo_back_MIoU = vevaluator.Mean_Intersection_over_Union()
            Acc_class,Class_ACC,wo_back_ACC = vevaluator.Pixel_Accuracy_Class()
            Class_precision, Class_recall,Class_F1score = vevaluator.Class_F1_score()
            _, _,Class_Fbetascore = vevaluator.Class_Fbeta_score(beta=betavalue)

            # result_crf=np.array(decode_segmap(result_crf,name='full'))
            pre_body=decode_segmap(ch_channel(predict),nc=4,name='full')
            v_la=decode_segmap(_label.cpu().detach().numpy().astype('uint8'),nc=4,name='full')
            _input = _input.detach().cpu().numpy()
            
            total_IOU.append(Class_IOU)
            total_F1.append(Class_F1score)
            total_Fbeta.append(Class_Fbetascore)
            total_recall.append(Class_recall)
            total_predict.append(Class_precision)

            save_stack_images = {'final_predict':pre_body,'final_la':v_la,
                                'FINAL_input':_input}
            save_path=logger.save_images(save_stack_images,i)

    # logger.make_full_image(imagename='final_predict')
    logger.save_csv_file(np.array(total_IOU),name='total_IOU')
    logger.save_csv_file(np.array(total_F1),name='total_F1')
    logger.save_csv_file(np.array(total_Fbeta),name='total_Fbeta')
    logger.save_csv_file(np.array(total_recall),name='valid_total_recall')
    logger.save_csv_file(np.array(total_predict),name='valid_total_precision')
    # logger.make_full_image(imagename='post_image')
            # # print(result_crf)
            # result_crf = np.transpose(result_crf,[0,2,1])
            # # print(v_input[0].shape)
            
            # skimage.io.imsave(path+"pre_body"+"_"+str(epoch)+".png",np.transpose(pre_body[num],[1,2,0]))    
            # skimage.io.imsave(path+"labe_"+"_"+str(epoch)+".png",v_la[num])
            # skimage.io.imsave(path+"img"+"_"+str(epoch)+".png",np.transpose(v_input[num].detach().cpu().numpy(),[1,2,0]))
            # skimage.io.imsave(path+"result_crf"+"_"+str(epoch)+".png",result_crf)
        

