for i in 4
do
   python mytrain.py --modelname=unet_test --epochs=801 --batch_size=250 --use_train --knum=$i --weight=100 --gpu=3 --patchsize=128 --stride=40 --RECONGAU --RECON --Aloss=SIGRMSE --Rloss=RMSE --labmda=1. --partial_recon  --mask_trshold=0.25 --deleteall  --cross_validation
done

