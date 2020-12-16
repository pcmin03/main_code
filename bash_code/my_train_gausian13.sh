cd ..
for i in 8
do
	python mytrain.py --modelname=unet_sample --epochs=300 --activation=sigmoid --batch_size=200  --knum=$i --weight=1000 --gpu=2 --patchsize=128 --stride=40 --RECONGAU --RECON --partial_recon  --Aloss=SIGRMSE --labmda=1  --mask_trshold=0.3 --cross_validation --deleteall  
done 
