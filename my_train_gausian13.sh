for i in 6
do
   python mytrain.py --modelname=unet_test --epochs=701 --batch_size=250 --use_train --knum=$i --weight=1 --gpu=2 --patchsize=128 --stride=40 --RECONGAU --Aloss=SIGRMSE --Rloss=RMSE --labmda=1. --partial_recon  --mask_trshold=0.3 --deleteall 
done

