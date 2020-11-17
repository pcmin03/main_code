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
from mydataset import mydataset_2d
from logger import Logger
from metrics import *
import argparse
import torch.autograd as autograd

from HED import HED
from RCF import RCF

from fusenet import ICNet 

# from DenseCRFLoss import DenseCRFLoss
from custom_transforms import denormalizeimage

parser = argparse.ArgumentParser(description='Process some integers')
parser.add_argument('--knum', help='Select Dataset')
parser.add_argument('--gpu', default=0,help='comma separated list of GPU(s) to use.',type=int)
parser.add_argument('--weight_decay',default=0.001,help='set weight_decay',type=float)
parser.add_argument('--weight',help='set Adaptive weight',type=float)
parser.add_argument('--start_lr',default=7e-2, help='set of learning rate', type=float)
parser.add_argument('--end_lr',default=7e-3,help='set fo end learning rate',type=float)
parser.add_argument('--paralle',default=False,help='GPU paralle',type=bool)
parser.add_argument('--scheduler',default='Cosine',help='select schduler method',type=str)
parser.add_argument('--epochs',default=201,help='epochs',type=int)
parser.add_argument('--out_class',default=4,help='set of output class',type=int)
parser.add_argument('--changestep',default=20,help='change train to valid',type=int)
parser.add_argument('--pretrain',default=False,help='load pretrained',type=bool)


# parser.add_argument('--batch',defalut=110,help='set Adaptive weight',type=int)_
args = parser.parse_args()


with open('my_config.yml') as f:
    conf = yaml.load(f)


num_workers = 8
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
#set evluaution 
best_epoch, best_axon, best_dend, best_axon_recall, F1best = 0,0,0,0,0

print(knum,'=============')
foldnum = 10
betavalue = 2
phase = 'train'
model = 'class_unet'
data_name = 'uint16'
save_name = 'seg_'
name = save_name+str(int(args.weight))+'_'+str(data_name)+'_'

# name = 'Adaptive_Full_4_image_scribble1000_scribble_'
# name = "Adaptive_Full_4_image_weight"+str(args.weight)
# name = "Adaptive_Full_4_image_RMSE"
# name = "Adpative_at0tention_module"+str(args.weight)
# name = "neworiginalv7"+str(data_name)
# name = "neworiginalv8_boundary_v3_"+str(data_name)
# name = "Adaptive_Full_4_image_skeletonsize"+str(int(args.weight))+'_'+str(data_name)+'_'

# name = "Adaptive_Full_4_image_scribble"+str(int(args.weight))+'_'+str(data_name)+'_'
# name = "Adaptive_Full_4_image_class_loss"+str(int(args.weight))+'_'+str(data_name)+'_'

# name = "neworiginal"
# neworiginalv3
# unetmodel8neworiginal
# name = Adaptive_Full_4_image_gaussian_AdaptiveCE_V21000.0_uint16_
# name = "Adaptive_Full_4_image_AdaptiveCE_"+str(int(args.weight))+'_'+str(data_name)+'_'
# name = "New_Adaptive_Full_4_image_AdaptiveCE_dend_"+str(int(args.weight))+'_'+str(data_name)+'_'
# name = "New_Adaptive_Full_4_image_AdaptiveCE_full_4_v2"+str(int(args.weight))+'_'+str(data_name)+'_'

# name = "Adaptive_Full_4_image_AdaptiveCE_"+str(int(args.weight))
# name = "Adaptive_Full_4_image_AdaptiveCE_Multiloss"+str(int(args.weight))+'_'+str(data_name)+'_'

# name = "Adaptive_Full_4_image_class_AdaptiveCE_v2"+str(args.weight)
# name  =  "Adaptive_Full_4_image_gaussian_AdaptiveCE_V3"+str(args.weight)+'_'+str(data_name)+'_'
# name  =  "Adaptive_Full_4_image_class_gaussian_AdaptiveCE_V3"+str(args.weight)+'_'+str(data_name)+'_'
# unetmodel/unetmodel8Adaptive_Full_4_image_gaussian_AdaptiveCE_V2_uint161000.0/
# name  =  "Adaptive_Full_4_image_gaussian_AdaptiveCE_V2"+str(args.weight)+'_'+str(data_name)+'_'
# unetmodel8New_Adaptive_Full_4_image_AdaptiveCE_full_4_v21000_uint16_
# name = "Adaptive_Full_4_image_gaussian_AdaptiveCE_V2_"+str(data_name)+str(args.weight)
# unetmodel8New_Adaptive_Full_4_image_AdaptiveCE_full_4_v21000_uint16_/
# name = "New_Adaptive_Full_4_image_AdaptiveCE_full_4_v2"+str(int(args.weight))+'_'+str(data_name)+'_'
# name  =  "Adaptive_Full_4_image_gaussian_AdaptiveCE_V2"'_'+str(data_name)+str(args.weight)
# name = 'Adaptive_Full_4_image_gaussian_AdaptiveCE_V210003.0_uint16_'
# name  =  "Adaptive_Full_4_image_gaussian_AdaptiveCE_V2_up_"+str(args.weight)+'_'+str(data_name)+'_'
# name  =  "Adaptive_Full_4_image_gaussian_AdaptiveCE_V2_uint16"+str(args.weight)

# name = "remove_background_Adaptive_Full_4_image_gaussian_AdaptiveCE_V2"+str(args.weight)
# name = "Adaptive_Full_4_image_class_AdaptiveCE_"+str(args.weight)
# name = "Adaptive_Full_4_image_AdaptiveCE_distacemap"+str(args.weight)

# name = "Adaptive_Full_4_image_CustomCE_"+str(args.weight)

# name = "Adaptive_Full_4_image_CE_Scehduler"+str(args.weight)
# name = "Adaptive_Full_4_image_custom_CE"+str(args.weight)
# name = "Custom_Adaptive_DistanceMap"+str(args.weight)
# name = "Custom_Adaptive_apply_DistanceMap"+str(args.weight)
# name = "Adaptive_CE"+str(args.weight)

# unetmodel2Adaptive_loss_weight_decay0.001

# path = '../nested_CV_nest_net_loss'+str(knum)+'/'

load_pretrain = args.pretrain
deepsupervision = False
trainning = True
testing = True
discrim = False
deleteall = False
patchwise = False
multichannel = False
multi_ = False
preprocessing_ = True
multiple_scale = False

#set data path 
cross_validation = True

if data_name == 'uint8':
    #uint8 train
    imageDir= '../new_project_original_image/'
    labelDir = '../new_project_label_modify_image/'
    #uint8 test
    testDir ='../test_image/'
    tlabelDir = '../test_label/'

elif data_name == 'uint16':
    #uint16 train
    imageDir= '../AIAR_orignal_data/train_project_image/'
    labelDir = '../AIAR_orignal_data/train_project_label/'
    #uint16 test
    testDir ='../AIAR_orignal_data/test_project_image/'
    tlabelDir = '../AIAR_orignal_data/test_project_label/'

elif data_name == 'edge':
    #edge dataset train
    imageDir= '../AIAR_orignal_data/train_project_image/'
    labelDir = '../AIAR_orignal_data/train_boundary_label/'
    #edge dataset test
    testDir= '../AIAR_orignal_data/test_project_image/'
    tlabelDir = '../AIAR_orignal_data/test_boundary_label/'
    
elif data_name == 'scribble':
    #edge dataset train
    # imageDir= '../AIAR_orignal_data/train_project_image/'
    imageDir= '../new_project_original_image/'
    labelDir = '../AIAR_orignal_data/train_scribble_project_label/'
    full_4_labelDir = '../AIAR_orignal_data/train_project_label/'
    #edge dataset test
    # testDir= '../AIAR_orignal_data/test_project_image/'
    testDir ='../test_image/'
    tlabelDir = '../AIAR_orignal_data/test_boundary_label/'


modelsave_dir = '../'+str(model)+'model'
if not os.path.exists(modelsave_dir):
    print('----- Make_save_Dir-------------')
    os.makedirs(modelsave_dir)
    print(modelsave_dir)

path = modelsave_dir+'/'+str(model)+'model'+str(knum)+str(name)+'/'
#set devices
device = torch.device('cuda:'+str(args.gpu)if torch.cuda.is_available() else "else")

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()
from Wnet.models import WNet
#image segmentation 
if model =='unet':
    batch_size = 120
    gen = pretrain_unet(3,5).to(device)

elif model=='res_unet':
    batch_size = 22
    gen = Segmentataion_resnet101unet().to(device)

elif model=='deeplab':
    batch_size = 120
    gen = pretrain_deeplab_unet(plus = False).to(device)

elif model=='deeplabplus':
    batch_size = 100
    gen = pretrain_deeplab_unet(plus = True).to(device)

elif model=='pspnet':
    batch_size = 130
    gen = pretrain_pspnet().to(device)

elif model=='multi_net':
    batch_size = 150
    gen = ICNet().to(device)

elif model=='refinenet':
    batch_size = 50
    gen = refinenet().to(device)

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
    gen = efficient_nestunet(1,5).to(device)

elif model == 'Multi_Unet':
    batch_size = 30
    gen = Multi_UNet().to(device)

elif model == 'DANET':
    batch_size = 35
    gen = DANet().to(device)
#boundray detection
elif model == 'CASENet':
    batch_size = 18
    gen = pretrain_casenet().to(device)

elif model == 'HED':
    batch_size = 40
    gen = HED().to(device).apply(weights_init)
    eval_channel = int(2)
    classnum = int(2)

elif model == 'RCF' :
    batch_size = 60
    gen = RCF().to(device).apply(weights_init)
    eval_channel = int(2)
    classnum = int(2)

elif model == 'WNet':
    batch_size = 15
    gen = WNet().to(device)

elif model == 'class_unet':
    batch_size = 85
    gen = clas_pretrain_unet(1,4).to(device)
#set paralle
if paralle == True:
    gen = torch.nn.DataParallel(gen, device_ids=[0,1,2,3])

if deleteall==True:
    logger = Logger(path,batch_size=batch_size,delete=deleteall,num=str(knum),name=model+name)

else:
    logger = Logger(path,batch_size=batch_size,delete=False,num=str(knum),name=model+name)
    #training

optimizerG = optim.Adam(gen.parameters(),lr=learning_rate)
    #tune lr
if model == 'HED' or model == 'RCF' :
    net_parameters_id = {}
    net = gen
    for pname, p in net.named_parameters():
        if pname in ['conv1_1.weight','conv1_2.weight',
                        'conv2_1.weight','conv2_2.weight',
                        'conv3_1.weight','conv3_2.weight','conv3_3.weight',
                        'conv4_1.weight','conv4_2.weight','conv4_3.weight']:
            print(pname, 'lr:1 de:1')
            if 'conv1-4.weight' not in net_parameters_id:
                net_parameters_id['conv1-4.weight'] = []
            net_parameters_id['conv1-4.weight'].append(p)
        elif pname in ['conv1_1.bias','conv1_2.bias',
                        'conv2_1.bias','conv2_2.bias',
                        'conv3_1.bias','conv3_2.bias','conv3_3.bias',
                        'conv4_1.bias','conv4_2.bias','conv4_3.bias']:
            print(pname, 'lr:2 de:0')
            if 'conv1-4.bias' not in net_parameters_id:
                net_parameters_id['conv1-4.bias'] = []
            net_parameters_id['conv1-4.bias'].append(p)
        elif pname in ['conv5_1.weight','conv5_2.weight','conv5_3.weight']:
            print(pname, 'lr:100 de:1')
            if 'conv5.weight' not in net_parameters_id:
                net_parameters_id['conv5.weight'] = []
            net_parameters_id['conv5.weight'].append(p)
        elif pname in ['conv5_1.bias','conv5_2.bias','conv5_3.bias'] :
            print(pname, 'lr:200 de:0')
            if 'conv5.bias' not in net_parameters_id:
                net_parameters_id['conv5.bias'] = []
            net_parameters_id['conv5.bias'].append(p)

        elif pname in ['score_dsn1.weight','score_dsn2.weight','score_dsn3.weight',
                        'score_dsn4.weight','score_dsn5.weight']:
            print(pname, 'lr:0.01 de:1')
            if 'score_dsn_1-5.weight' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.weight'] = []
            net_parameters_id['score_dsn_1-5.weight'].append(p)
        elif pname in ['score_dsn1.bias','score_dsn2.bias','score_dsn3.bias',
                        'score_dsn4.bias','score_dsn5.bias']:
            print(pname, 'lr:0.02 de:0')
            if 'score_dsn_1-5.bias' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.bias'] = []
            net_parameters_id['score_dsn_1-5.bias'].append(p)
        elif pname in ['score_final.weight']:
            print(pname, 'lr:0.001 de:1')
            if 'score_final.weight' not in net_parameters_id:
                net_parameters_id['score_final.weight'] = []
            net_parameters_id['score_final.weight'].append(p)
        elif pname in ['score_final.bias']:
            print(pname, 'lr:0.002 de:0')
            if 'score_final.bias' not in net_parameters_id:
                net_parameters_id['score_final.bias'] = []
            net_parameters_id['score_final.bias'].append(p)

    optimizer = torch.optim.SGD([
            {'params': net_parameters_id['conv1-4.weight']      , 'lr': args.start_lr*1    , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['conv1-4.bias']        , 'lr': args.start_lr*2    , 'weight_decay': 0.},
            {'params': net_parameters_id['conv5.weight']        , 'lr': args.start_lr*100  , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['conv5.bias']          , 'lr': args.start_lr*200  , 'weight_decay': 0.},
            {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': args.start_lr*0.01 , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': args.start_lr*0.02 , 'weight_decay': 0.},
            {'params': net_parameters_id['score_final.weight']  , 'lr': args.start_lr*0.001, 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['score_final.bias']    , 'lr': args.start_lr*0.002, 'weight_decay': 0.},
        ], lr=args.start_lr, momentum=0.9, weight_decay=2e-4)


if use_scheduler == 'Cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerG,100,T_mult=1,eta_min=end_rate)
    if model == 'HED' or model == 'RCF' :
        scheduler = optim.lr_scheduler.StepLR(optimizerG, step_size=100, gamma=0.1)

if discrim == True:
    batch_size = 50
    logger.changedir('adversarial')
    dis = classification_model().to(device)
    criterion1 =torch.nn.L1Loss().to(device)
    optimizerD = optim.Adam(dis.parameters(),lr=learning_rate)
    schdulerD = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD,100,T_mult=1,eta_min=end_rate)

def compute_gradient_penalty(D, real_samples, fake_samples):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).expand_as(real_samples)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0],1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# densecrflosslayer = DenseCRFLoss(weight=4e-9, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
# print(densecrflosslayer)
    
    

#set loss
# criterion = Custom_CE(int(args.weight),Gaussian=True).to(device)
# criterion = Custom_Adaptive_Gaussian_DistanceMap(int(args.weight),True).to(device)
# criterion = Custom_dend_Adaptive_Gaussian_DistanceMap(int(args.weight),True).to(device)

# criterion = Custom_WeightedCrossEntropyLossV2().to(device)
criterion = nn.CrossEntropyLoss()
classificaiton_loss = nn.BCEWithLogitsLoss()
# from configure import Config

# from DataLoader import DataLoader
# from Wnet.Ncuts import NCutsLoss, cal_weight

# criterion = NCutsLoss()

# criterion = Custom_WeightedCrossEntropyLossV2().to(device)
# criterion = MultiLLFunction().to(device)

# criterion = cross_entropy_loss().to(device)


# MultiLLFunction
print(1/(1+int(args.weight)),'1111')

#set matrix score
evaluator = Evaluator(eval_channel)
inversevaluator = Evaluator(eval_channel)

image,labels = divide_kfold(imageDir,labelDir,k=foldnum,name='test')

train_num, test_num = 'train'+str(knum), 'test'+str(knum)
# image_valid = image[train_num][-3:]
# label_valid = labels[train_num][-3:]

image_train = image[train_num]
label_train = labels[train_num]
image_valid = image[test_num]
label_valid = labels[test_num]

def make_batch(dataset):
    inputs = [dataset for sample in samples]
    labels = [dataset for sample in samples]
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    return {'input': padded_inputs.contiguous(),
            'label': torch.stack(labels).contiguous()}

if data_name == 'scribble':
    image,full_4_labels = divide_kfold(imageDir,full_4_labelDir,k=foldnum,name='test')
    ful_label_train = full_4_labels[train_num]
    ful_label_valid = full_4_labels[test_num]

if trainning == True:
    if load_pretrain == True:
        # if os.path.exists(path+"lastsave_models{}.pth"):
        checkpoint = torch.load(path +"lastsave_models{}.pth")
        gen.load_state_dict(checkpoint['gen_model'])
    else: 
        pass
        
    if cross_validation == True:
        print(f"{image_valid},{label_valid}")

        if data_name == 'scribble':
            MyDataset = {'train' : DataLoader(mydataset_2d_scribble(image_valid,label_valid,ful_label_valid,False,patchwise=patchwise,
                                            phase='train',multichannel=multichannel,preprocessing=preprocessing_,multiple_scale=multiple_scale),
                                            1, 
                                            shuffle = False,
                                            num_workers = num_workers),

                        'valid' : DataLoader(DataLoader(image_valid,label_valid,ful_label_valid,False,patchwise=patchwise,
                                            phase='test',multichannel=multichannel,preprocessing=preprocessing_,multiple_scale=multiple_scale),
                                            1, 
                                            shuffle = False,
                                            num_workers = num_workers)}
        else : 
            MyDataset = {'train': DataLoader(mydataset_2d(image_train,label_train,patchwise=patchwise,phase='train',multichannel=multichannel,preprocessing=True,multiple_scale=multiple_scale),
                                batch_size, 
                                shuffle = True,
                                num_workers = num_workers),
                'valid' : DataLoader(mydataset_2d(image_valid,label_valid,False,patchwise=False,phase='test',multichannel=multichannel,preprocessing=False,multiple_scale=multiple_scale),
                                    1, 
                                    shuffle = False,
                                    num_workers = num_workers)}
    print("start trainning!!!!")
    for epoch in range(epochs):
        evaluator.reset()
        inversevaluator.reset()
        seg_loss = 0
        cls_loss = 0
        t_loss = 0
        avg_IOU =0 
        avg_F1 = 0
        if epoch %changestep == 0:
            phase = 'valid'
            gen.eval()            
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
            if discrim == True:
                dis.train()

        print(f"{epoch}/{epochs}epochs,IR=>{get_lr(optimizerG)},best_epoch=>{best_epoch},phase=>{phase}")
        print(f"==>{path}<==")
        for i, batch in enumerate(tqdm.tqdm(MyDataset[phase])):
            
            _input, _label = Variable(batch[0]).to(device), Variable(batch[1]).to(device)
            class_label = Variable(batch[2]).to(device)
            
            
            optimizerG.zero_grad()
            torch.autograd.set_detect_anomaly(True)

            if deepsupervision==True and model == 'nest_unet':   
                
                loss = 0
                
                back,body,dend,axon = gen(_input)
                
                gt=_label
                predict = torch.cat((back, body,dend,axon),dim=1).cuda().float()
                
                back_loss = criterion(predict,_label.long())

                CE_loss = back_loss
                seg_loss_body = CE_loss 
                seg_loss_body.backward(retain_graph = True)
                
                optimizerG.step()
                
                evaluator.add_batch(_label.cpu().numpy(),torch.argmax(predict,dim=1).cpu().numpy())
                IOU,Class_IOU,wo_back_MIoU = evaluator.Mean_Intersection_over_Union()
                Class_precision, Class_recall,Class_F1score = evaluator.Class_F1_score()
                _, _,Class_Fbetascore = evaluator.Class_Fbeta_score(beta=betavalue)                    
                    
            else :
                ##### train with source code #####
                with torch.set_grad_enabled(phase == 'train'):
                    predict,pred_class=gen(_input,phase)
                    precision =predict
                
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

                    if model == 'DANET' or model == 'multi_net' or model == 'refinenet':
                        CE_loss1 = criterion(predict[0],_label)
                        CE_loss2,_ = criterion(predict[1],_label,upsample=True)
                        CE_loss3,_ = criterion(predict[2],_label,upsample=True)
                        CE_loss4,_ = criterion(predict[3],_label,upsample=True)
                        CE_loss = (CE_loss1 + CE_loss2 + CE_loss3 + CE_loss4)
                        if multiple_scale == True:
                            predict[0] = F.interpolate(predict[0], predict[1].size()[2:])
                            _label = F.interpolate(_label.unsqueeze(1), predict[1].size()[2:])[:,0]
                            predict = predict[0]
                        elif multiple_scale == False:
                            predict = predict[0]
                    else :
                        # CE_loss = criterion(predict[0],_label)
                        if model == 'HED' or model == 'RCF' :
                            CE_loss = torch.zeros(1).cuda()
                            for o in predict:
                                CE_loss += criterion(o,_label)
                            # CE_loss = CE_loss / len(predict)
                            predict = predict[0]

                        elif model == 'WNet':
                            sw = _label.sum(-1).sum(-1)
                            pred,pad_pred = gen(_input)
                            ncuts_loss = criterion(pred,pad_pred,_label,sw)
                            ncuts_loss = ncuts_loss.sum()/batch_size
                            Ave_Ncuts = (Ave_Ncuts * i + ncuts_loss.item())/(i+1)
                            CE_loss = ncuts_loss

                        else :
                            CE_loss = criterion(predict,_label.long())
                                
                            if phase == 'train':
                                class_loss = CE_loss                            
                                # classificaiton_loss(pred_class,class_label)
                                loss = CE_loss + class_loss
                            else : 
                                class_loss = CE_loss
                                loss = CE_loss + class_loss
                            # target_vars = list()
                            # new_class =torch.zeros_like(pred_class)
                            # for i in range(len(_label)):
                            #     new_class[i] = Variable(torch.unique(_label[i]))

                            # print(class_label.shape,pred_class.shape,'11')
                            #  
                            
                            
                            if data_name =='scribble':
                                softmax = nn.Softmax(dim=1)
                                probs = softmax(predict)
                                croppings = (_label!=254).float()
                                denormalized_image = denormalizeimage(_input, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                densecrfloss = densecrflosslayer(denormalized_image,probs,croppings)
                                densecrfloss = densecrfloss.cuda()
                                loss = CE_loss + densecrfloss
                            
                    
                    if discrim == True:
                        optimizerD.zero_grad()
                        
                        real_val = dis(torch.argmax(predict,dim=1).unsqueeze(1).float())
                        fake_val = dis(_label.unsqueeze(1).float())

                        #gradient penalty & WGAN adversarial loss
                        
                        GP = compute_gradient_penalty(dis, _label.unsqueeze(1).float(), torch.argmax(predict,dim=1).unsqueeze(1).float())
                        d_loss = -torch.mean(real_val) + torch.mean(fake_val) + 10 * GP

                        if phase == 'train':
                            d_loss.backward(retain_graph = True)
                            optimizerD.step()

                        #add adversarial loss to generative(segmentation network)
                        other_loss = criterion1(real_val,fake_val)
                        g_loss = -torch.mean(fake_val)
                        CE_loss += g_loss  + other_loss
                        
                    if phase == 'train':
                        seg_loss_body = loss 
                        seg_loss_body.backward(retain_graph = True)
                        optimizerG.step()

                    seg_loss += CE_loss.item() * _input.size(0) 
                    cls_loss += class_loss.item() * _input.size(0)
                    t_loss = seg_loss + cls_loss 

                    predict = torch.argmax(predict,dim=1).cpu().numpy()
                    
                    if model == 'CASENet' or data_name == 'edge':
                        # compare_result = precision[0] > 0.5
                        # mid_result[compare_result == 1] = 1
                        # mid_result[compare_result == 0] = 0
                        # precision[0] = torch.sigmoid(precision[0])                       
                        mid_result = precision
                        compare_back = precision[:,0] > 0.5
                        compare_body = precision[:,1] > 0.5
                        compare_dend = precision[:,2] > 0.5
                        compare_axon = precision[:,3] > 0.5

                        mid_result[:,0][compare_back == 1] = 0
                        mid_result[:,0][compare_back == 0] = 0
                        
                        mid_result[:,1][compare_body == 1] = 1
                        mid_result[:,1][compare_body == 0] = 0
                        
                        mid_result[:,2][compare_dend == 1] = 1
                        mid_result[:,2][compare_dend == 0] = 0
                        
                        mid_result[:,3][compare_axon == 1] = 1
                        mid_result[:,3][compare_axon == 0] = 0

                        # final = mid_result[:,0] + mid_result[:,1] + mid_result[:,2] *2 +mid_result[:,3] *3
                        final =  torch.argmax(mid_result,dim=1)
                        evaluator.add_batch(_label.cpu().numpy(),final.cpu().numpy().astype('uint8'))
                    else :
                        if data_name == 'scribble':
                            evaluator.add_batch(_ful_label.cpu().numpy(),predict)
                        else: 
                            evaluator.add_batch(_label.cpu().numpy(),predict)

                    IOU,Class_IOU,wo_back_MIoU = evaluator.Mean_Intersection_over_Union()
                    Acc_class,Class_ACC,wo_back_ACC = evaluator.Pixel_Accuracy_Class()
                    Class_precision, Class_recall,Class_F1score = evaluator.Class_F1_score()
                    _, _,Class_Fbetascore = evaluator.Class_Fbeta_score(beta=betavalue)
                    if phase == 'valid':
                        avg_IOU += Class_IOU * _input.size(0)
                        avg_F1 += Class_F1score * _input.size(0)
        
        all_loss = [x/len(MyDataset[phase]) for x in [t_loss,seg_loss,cls_loss]]
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
                summary_print = {'t_loss':all_loss[0],
                                'Class_IOU':Class_IOU,
                                'precision':Class_precision, 
                                'recall':Class_recall,
                                'F1score':Class_F1score,
                                'Fbetascore':Class_Fbetascore}
                if discrim == True:
                    summary_print.update({'gan_g_loss':g_loss,'gan_d_loss':d_loss})

                logger.print_value(summary_print,'train')
            train_loss = {  't_loss':all_loss[0],
                            'seg_loss':all_loss[1],
                            'cls_loss': all_loss[2]}
            if discrim == True:
                train_loss.update({'gan_g_loss':g_loss,'gan_d_loss':d_loss})
            logger.summary_scalars(train_loss,epoch)
            

        elif phase == 'valid':

            avg_class_IOU,avg_class_F1 = [x /len(MyDataset[phase]) for x in [avg_IOU,avg_F1]]
            
            if classnum == 8:
                inversetest_val = {"inverseClass_IOU":inverseClass_IOU,
                            'inverseprecision':inverseClass_precision, 
                            'inverserecall':inverseClass_recall,
                            'inverseF1score':inverseClass_F1score,
                            'inverseFbetascore':inverseClass_Fbetascore}
                logger.print_value(inversetest_val,'inversetest')
            else :
                test_val = {"avg_class_IOU":avg_class_IOU,"avg_class_F1":avg_class_F1,
                            "Class_IOU":Class_IOU,"Class_ACC":Class_ACC,
                            "wo_back_MIoU":wo_back_MIoU,"wo_back_ACC":wo_back_ACC,
                            'precision':Class_precision, 
                            'recall':Class_recall,
                            'F1score':Class_F1score,
                            "val_loss":all_loss[0],
                            'seg_loss':all_loss[1],
                            'cls_loss':all_loss[2]}
                logger.print_value(test_val,'test')

            IOU_scalar = dict()
            precision_scalar = dict()
            recall_scalr = dict()
            F1score_scalar = dict()
            Fbetascore_scalar = dict()
            
            for i in range(classnum):
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
                
            validation_loss = {'val_loss':all_loss[1],
                                'val_CE_loss':all_loss[2]}

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
            

            if  (Class_IOU[classnum-1] > best_axon) or (Class_F1score[classnum-1] > best_axon_recall) :
                torch.save({"gen_model":gen.state_dict(),
                        "optimizerG":optimizerG.state_dict(),
                        "epochs":epoch},
                        path+"bestsave_models{}.pth")
                print('save!!!')
                best_axon = Class_IOU[classnum-1]
                best_axon_recall = Class_F1score[classnum-1]
                F1best = Class_F1score[classnum-1]
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
                #     v_la = decode_segmap(torch.argmax(_label,dim=1).cpu().detach().numpy(),name='full_4')
                #     _input = _input.detach().cpu().numpy()
                # else : 
                if model == 'HED' or model == 'RCF'  :
                    v_la = (1-_label.unsqueeze(1).cpu().detach().numpy())*255.0
                elif model == 'CASENet':
                    v_la = decode_segmap(_label.cpu().detach().numpy(),name='full_4')
                else:
                    v_la = decode_segmap(_label.cpu().detach().numpy(),name='full_4')
                    # ful_v_la = decode_segmap(_ful_label.cpu().detach().numpy(),name='full_4')

                _input = _input.detach().cpu().numpy()
                if data_name == 'uint16':
                    normalizedImg = np.zeros((1024, 1024))
                    _input = cv2.normalize(_input,  normalizedImg, 0, 65535 , cv2.NORM_MINMAX)
                if model == 'DANET' or model == 'multi_net' or model == 'refinenet':
                    v_la2=decode_segmap(label2.cpu().detach().numpy(),name='full_4')
                    v_la3=decode_segmap(label3.cpu().detach().numpy(),name='full_4')
                    v_la4=decode_segmap(label4.cpu().detach().numpy(),name='full_4')
                        
                    pre_body1=decode_segmap(torch.argmax(DAnetpredict[0][:,0:4],dim=1).cpu().numpy())
                    pre_body2=decode_segmap(torch.argmax(DAnetpredict[1][:,0:4],dim=1).cpu().numpy())
                    pre_body3=decode_segmap(torch.argmax(DAnetpredict[2][:,0:4],dim=1).cpu().numpy())
                    pre_body4=decode_segmap(torch.argmax(DAnetpredict[3][:,0:4],dim=1).cpu().numpy())

                    save_stack_images = {'pre_1_':pre_body1,'pre_2_':pre_body2,'pre_3_':pre_body3,'pre_4_':pre_body4,'v_la_':v_la,'_input_':_input}
                    save_stack_images.update({'v_la2_':v_la2,'v_la3_':v_la3,'v_la4_':v_la4})
                else:
                    # print(predict.shape)

                    pre_body = decode_segmap(predict)
                    precision = precision.unsqueeze(2).cpu().numpy() * 65535.
                    print(precision.shape,'123123123')

                    # precision
                    # torchvision.utils.save_images(1-results_all,)
                    save_stack_images = {'v_la':v_la,'_input':_input.astype('uint16'),'precision':precision.astype('uint16')}
                    if model == 'CASENet':
                        # precision
                        pre_body = decode_segmap(precision.cpu().numpy(),name='full_4')
                        save_stack_images.update({'pre_body_back':pre_body})
                        # save_stack_images.update({'pre_body_back':pre_body[0],
                        #                     'pre_body_body':pre_body[1],
                        #                     'pre_body_dend':pre_body[2],
                        #                     'pre_body_axon':pre_body[3],
                        #                     'pre_body_full_4':pre_body[0]+pre_body[1]+pre_body[2]+pre_body[3]})
                    
                    # save_stack_images = {'pre_body1':pre_body[1],'v_la':v_la,'_input':_input.astype('uint16')}
                    # precision=score_output(precision)
                    # print(precision.shape)
                    elif model == 'HED' or model =='RCF':
                        result = torch.squeeze(precision[-1].detach()).cpu().numpy()
                        results_all = torch.zeros((len(precision),1,1024,1024))
                        for i in range(len(precision)):
                            results_all[i,0,:,:] = precision[i]
                        results_all = results_all.cpu().detach().numpy()
                        save_stack_images.update({'precision1':(1-results_all[0:1])*255.0,
                        'precision2':(1-results_all[1:2])*255.0,
                        'precision3':(1-results_all[2:3])*255.0,
                        'precision4':(1-results_all[3:4])*255.0})
                    else : 
                        save_stack_images.update({'pre_body':pre_body})
                    # inversepre_body=decode_segmap(torch.argmax(predict[:,4:8],dim=1).cpu().numpy(),name='inverse')

                

                # logger.save_csv_file(np.array(total_IOU),name='valid_total_IOU')
                # logger.save_csv_file(np.array(total_F1),name='valid_total_F1')
                # logger.save_csv_file(np.array(total_Fbeta),name='valid_total_Fbeta')
                # logger.save_csv_file(np.array(total_recall),name='valid_total_recall')
                # logger.save_csv_file(np.array(total_predict),name='valid_total_precision')
                if classnum == 8:
                    logger.save_csv_file(np.array(inversetotal_IOU),name='inversevalid_total_IOU')
                    logger.save_csv_file(np.array(inversetotal_F1),name='inversevalid_total_F1')
                    logger.save_csv_file(np.array(inversetotal_Fbeta),name='inversevalid_total_Fbeta')
                    logger.save_csv_file(np.array(inversetotal_recall),name='inversevalid_total_recall')
                    logger.save_csv_file(np.array(inversetotal_predict),name='inversevalid_total_precision')


                logger.save_images(save_stack_images,epoch)


if testing == True:
    #testing
    print("==========testing===============")

    # if os.path.exists(path+"lastsave_models{}.pth"):
    checkpoint = torch.load(path +"lastsave_models{}.pth")
    gen.load_state_dict(checkpoint['gen_model'])
    gen.eval()

    logger = Logger(path,batch_size=batch_size)

    total_IOU = []
    total_F1 = []
    total_Fbeta = []
    total_recall = []
    total_predict = []
    if data_name == 'scribble':
        MyDataset = {'train' : DataLoader(mydataset_2d_scribble(image_valid,label_valid,ful_label_valid,False,patchwise=patchwise,
                                        phase='train',multichannel=multichannel,preprocessing=preprocessing_,multiple_scale=multiple_scale),
                                        1, 
                                        shuffle = False,
                                        num_workers = num_workers),

                    'valid' : DataLoader(DataLoader(image_valid,label_valid,ful_label_valid,False,patchwise=patchwise,
                                        phase='test',multichannel=multichannel,preprocessing=preprocessing_,multiple_scale=multiple_scale),
                                        1, 
                                        shuffle = False,
                                        num_workers = num_workers)}
    else:
        MyDataset = {'valid' :   DataLoader(mydataset_2d(image_valid,label_valid,ful_label_valid,False,patchwise=patchwise,phase='test',preprocessing=preprocessing_,multichannel=multichannel),
                                1, 
                                shuffle = False,
                                num_workers = num_workers),
                    'test' :   DataLoader(mydataset_2d(testDir,tlabelDir,False,patchwise=patchwise,phase='test',preprocessing=preprocessing_,multichannel=multichannel, isDir=False),
                                1, 
                                shuffle = False,
                                num_workers = num_workers)}
    phase = 'valid'
    logger.changedir(str(phase)+'_result2')
    
    for i, batch in enumerate(MyDataset[phase]):
        _input, _label = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():

            if deepsupervision==True and model == 'nest_unet':
                back,body,dend,axon = gen(_input)
                predict = torch.cat((back, body,dend,axon),dim=1).cuda().float()

            else :
                if model == 'DANET'or model == 'multi_net' or model == 'refinenet' or model == 'RCF':
                    # print(_input.max(),'1111')
                    predict = gen(_input)
                    # print(predict[0].max(),'1111')
                    DAnetpredict = predict
                    precision = predict
                    predict = predict[0]
                else:
                    
                    predict = gen(_input)
            precision = predict
            predict = torch.argmax(predict,dim=1).cpu().numpy()

            print("???????????????????????????????????????????????")
            print(path)
            # predict = channel_segmap(predict.cpu().numpy())
            # vevaluator.add_batch(_label.cpu().numpy(),predict)
            if model =='RCF':
                v_la = (1-_label.unsqueeze(1).cpu().detach().numpy())*255.0
                
            else:
                v_la = decode_segmap(_label.cpu().detach().numpy().astype('uint16'),nc=4,name='full_4')
            _input = _input.detach().cpu().numpy()
            save_stack_images = {'final_la':v_la, 'FINAL_input':_input}

            # result_crf=np.array(decode_segmap(result_crf,name='full_4'))
            if model == 'DANET' or model == 'multi_net'  or model == 'refinenet':
                pre_body1=decode_segmap(torch.argmax(DAnetpredict[0][:,0:4],dim=1).cpu().numpy())
                pre_body2=decode_segmap(torch.argmax(DAnetpredict[1][:,0:4],dim=1).cpu().numpy())
                pre_body3=decode_segmap(torch.argmax(DAnetpredict[2][:,0:4],dim=1).cpu().numpy())
                pre_body4=decode_segmap(torch.argmax(DAnetpredict[3][:,0:4],dim=1).cpu().numpy())
                save_stack_images = {'final_predict1':pre_body1,'final_predict2':pre_body2,'final_predict3':pre_body3,'final_predict4':pre_body4,'final_la':v_la,
                                'FINAL_input':_input}
            
            elif model == 'RCF':    
                result = torch.squeeze(precision[-1].detach()).cpu().numpy()
                results_all = torch.zeros((len(precision),1,1024,1024))
                for i in range(len(precision)):
                    results_all[i,0,:,:] = precision[i]
                results_all = results_all.cpu().detach().numpy()
                save_stack_images.update({'precision1':((1-results_all[0:1])>0.5)*255.0,
                                        'precision2':((1-results_all[1:2])>0.5)*255.0,
                                        'precision3':((1-results_all[2:3])>0.5)*255.0,
                                        'precision4':((1-results_all[3:4])>0.5)*255.0})
                
                vevaluator.add_batch(_label.cpu().numpy(),(1-results_all[0])>0.5)
            # elif model == 'CASENet':
            #     result = 
            else:
                pre_body=decode_segmap(predict,nc=5,name='full_4')
                print(precision[:,2].max())
                save_stack_images.update({'final_predict_20':precision[:,2].cpu().numpy()*10})

                save_stack_images.update({'final_predict':pre_body})

            #select class
            vevaluator.add_batch(_label.cpu().numpy(),predict)
            IOU,Class_IOU,wo_back_MIoU = vevaluator.Mean_Intersection_over_Union()
            Acc_class,Class_ACC,wo_back_ACC = vevaluator.Pixel_Accuracy_Class()
            Class_precision, Class_recall,Class_F1score = vevaluator.Class_F1_score()
            _, _,Class_Fbetascore = vevaluator.Class_Fbeta_score(beta=betavalue)


            
                # pre_body=decode_segmap(ch_channel(predict),nc=4,name='full_4')
            # pre_body=decode_segmap(ch_channel(predict),nc=4,name='full_4')
            
            total_IOU.append(Class_IOU)
            total_F1.append(Class_F1score)
            total_Fbeta.append(Class_Fbetascore)
            total_recall.append(Class_recall)
            total_predict.append(Class_precision)
            # if 
            save_path=logger.save_images(save_stack_images,i)

    # logger.make_full_4_image(imagename='final_predict')
    logger.save_csv_file(np.array(total_IOU),name='total_IOU')
    logger.save_csv_file(np.array(total_F1),name='total_F1')
    logger.save_csv_file(np.array(total_Fbeta),name='total_Fbeta')
    logger.save_csv_file(np.array(total_recall),name='valid_total_recall')
    logger.save_csv_file(np.array(total_predict),name='valid_total_precision')
    # logger.make_full_4_image(imagename='post_image')
            # # print(result_crf)
            # result_crf = np.transpose(result_crf,[0,2,1])
            # # print(v_input[0].shape)
            
            # skimage.io.imsave(path+"pre_body"+"_"+str(epoch)+".png",np.transpose(pre_body[num],[1,2,0]))    
            # skimage.io.imsave(path+"labe_"+"_"+str(epoch)+".png",v_la[num])
            # skimage.io.imsave(path+"img"+"_"+str(epoch)+".png",np.transpose(v_input[num].detach().cpu().numpy(),[1,2,0]))
            # skimage.io.imsave(path+"result_crf"+"_"+str(epoch)+".png",result_crf)
        

