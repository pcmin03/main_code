for i in 8
do
	python mytrain.py --modelname=unet_test --epochs=801 --activation=sigmoid --batch_size=400  --knum=$i --weight=100 --gpu=1 --patchsize=128 --stride=40 --RECONGAU --Aloss=SIGRMSE --RECON --labmda=1. --partial_recon --mask_trshold=0.3 --cross_validation  
done 
