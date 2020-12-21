cd ..
for i in 5
do
	python mytrain.py --modelname=unet_sample --epochs=300 --activation=sigmoid --batch_size=300  --knum=$i --weight=100 --gpu=0 --patchsize=128 --stride=40 --RECONGAU --RECON --partial_recon  --Aloss=SIGRMSE --labmda=1  --mask_trshold=0.3 --cross_validation --deleteall  
done 
