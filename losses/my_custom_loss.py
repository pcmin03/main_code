
from Distance_loss import * 
from information_loss import *
from Garborloss import *

def select_loss(args): 

    lossdict = dict()
    labelname = ""
    #suggest loss
    if recon_gau == True:
        criterion = Custom_Adaptive_gausian_DistanceMap(float(args.weight),distanace_map=args.class_weight,select_MAE=use_active,
                                                        treshold_value=bi_value,back_filter=args.back_filter,premask=args.premask)
        if args.back_filter== True:
            labelname += 'back_filter_'
        labelname += 'seg_gauadaptive_'+str(use_active)+'_'
  
    #compare loss
    elif args.RCE == True:
        criterion = noiseCE(int(args.weight),RCE=args.RCE)
        labelname += 'RCE_'

    elif args.NCE == True:
        criterion = noiseCE(int(args.weight),NCE=args.NCE)
        labelname += 'NCE_'

    elif args.BCE == True:
        criterion = noiseCE(int(args.weight),BCE=args.BCE)
        labelname += 'BCE_'

    elif args.NCDICE == True:
        criterion = NCDICEloss(r=args.NCDICEloss)
        labelname += 'NCDICE_'+str(args.NCDICEloss)+'_'

    elif adce == True:
        criterion = Custom_CE(int(args.weight),Gaussian=False,active=active)
        labelname += 'seg_adaptiveCE_'    
    lossdict.update({'mainloss':criterion})
    
    if recon == True:
        reconstruction_loss = Custom_RMSE_regularize(float(labmda),treshold_value=bi_value,select_MAE=use_active2,
                                                    use_median=use_median,partial = partial_recon,premask=args.premask,clamp=args.clamp)
        
        lossdict.update({'reconloss':reconstruction_loss})
        if partial_recon == True:
            labelname += 'part_reconloss2_'+str(use_active2)+'_' + str(labmda)+'_'
        else : 
            labelname += 'reconloss_'+str(use_active2)+'_' + str(labmda)+'_'
        
    labelname += str(bi_value) +'_'
    elif total_var_loss == True:
        tv_loss = TVLoss(TVLoss_weight=tv_value)
        labelname += 'TVLoss_'+str(tv_value)+'_'
    elif gabor_loss == True:
        labelname += 'garbor3_'+str(use_active3)+'_'+str(labmda2)+'_'
        if use_label == True:
            labelname += 'uselabel_'    
        gabors = dont_train(Custom_Gabor_loss(device=device,weight=float(labmda2),use_median=use_median,use_label=use_label).to(device))
        lossdict.update({'gaborloss':gabors})
        
    return lossdict, labelname+str(int(args.weight))+'_'+str(data_name)+'_' 