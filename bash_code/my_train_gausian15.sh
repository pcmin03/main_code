cd ..
for i in 10
do
	python mytrain.py --modelname=unet_test --epochs=150 --activation=sigmoid --batch_size=300  --knum=$i --weight=100 --gpu=2 --patchsize=128 --stride=40 --RECONGAU --RECON  --partial_recon  --Aloss=SIGRMSE --labmda=0.1  --mask_trshold=0.3 --cross_validation --pretrain
done 
