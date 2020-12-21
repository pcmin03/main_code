cd ..
for i in 6
do
	python mytrain.py --modelname=unet_sample --epochs=300 --activation=sigmoid --batch_size=150  --knum=$i --weight=100 --gpu=1 --patchsize=128 --stride=80 --RECONGAU --RECON --partial_recon  --Aloss=SIGRMSE --labmda=1  --mask_trshold=0.3 --cross_validation --deleteall  
done 
