for i in 1
do
   python mytrain.py --modelname=unet_test --oversample  --epochs=501 --datatype=uint16_xray --batch_size=250 --use_train --knum=$i --weight=100 --gpu=1 --patchsize=256 --stride=40 --RECONGAU --RECON --Aloss=SIGRMSE --Rloss=RMSE --labmda=1  --mask_trshold=0.3  --deleteall --cross_validation 
done

