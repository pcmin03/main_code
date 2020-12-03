for i in 4
do
	python mytrain.py --modelname=unet_test --epochs=801 --activation=sigmoid --batch_size=250  --knum=$i --weight=5 --gpu=0 --patchsize=128 --stride=40 --RECONGAU --RECON --Aloss=SIGRMSE --labmda=1. --RECON  --mask_trshold=0.3 --cross_validation  --deleteall
done 
