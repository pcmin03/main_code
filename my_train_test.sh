for i in 4
do
	python mytrain.py --modelname=unet_test --epochs=801 --activation=sigmoid --batch_size=400  --knum=$i --weight=1 --gpu=0 --patchsize=128 --stride=40 --RECONGAU --Aloss=SIGRMSE --labmda=1. --RECON --partial_recon --mask_trshold=0.25 --cross_validation --use_train
done 
