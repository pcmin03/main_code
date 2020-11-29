for i in 8
do
   python mytrain.py --modelname=unet_test --oversample  --epochs=501 --datatype=uint16_uint --batch_size=300 --use_train --knum=$i --weight=1 --gpu=0 --patchsize=128 --stride=40 --RECONGAU --RECON --Aloss=SIGRMSE --Rloss=RMSE --labmda=1  --mask_trshold=0.25  --deleteall --cross_validation 
done

