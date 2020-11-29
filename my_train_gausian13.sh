for i in 8
do
	python mytrain.py --modelname=unet_test --epochs=801 --activation=sigmoid --batch_size=400  --knum=$i --weight=1 --gpu=1 --patchsize=128 --stride=40 --RECONGAU --RECON --Aloss=SIGMAE --labmda=1. --partial_recon --mask_trshold=0.2 --cross_validation  
done 
