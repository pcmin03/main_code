cd ..
for i in 2
do
	python mytrain.py --modelname=unet_plus --epochs=400 --activation=sigmoid --batch_size=100 --knum=$i --weight=100 --gpu=2 --patchsize=128 --stride=40 --RECONGAU --RECON --partial_recon --Aloss=SIGRMSE --labmda=1 --mask_trshold=0.3 --cross_validation 
done 
