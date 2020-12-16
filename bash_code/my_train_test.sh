cd ..
for i in 2
do
	python mytrain.py --modelname=unet_test --epochs=801 --activation=sigmoid --batch_size=400  --knum=$i --weight=100 --gpu=0 --patchsize=128 --stride=40 --RECONGAU --Aloss=SIGRMSE --labmda=1. --RECON --mask_trshold=0.3 --cross_validation --pretrain --use_train
done 
