import argparse 

def my_config():
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
    parser.add_argument('--Kfold',default=10,help='set fo end learning rate',type=int)
    #preprocessing 
    parser.add_argument('--use_median', default=False, action='store_true',help='make binary median image')

    parser.add_argument('--patchsize', default=512, help='patch_size',type=int)
    parser.add_argument('--stride', default=60,help='stride',type=int)
    parser.add_argument('--batch_size', default=60,help='stride',type=int)
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
    parser.add_argument('--cross_validation',default=True, action='store_false',help='set Normalized Cross entropy')

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
    parser.add_argument('--activation',default='sigmoid',help='select active',type=str)

    parser.add_argument('--DISCRIM',default=False,help='set discriminate',type=bool)

    return parser.parse_args()