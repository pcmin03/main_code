for i in 8
do
	python mytrain.py  --modelname=unet_test --datatype=uint16_xray --epochs=801 --activation=sigmoid --batch_size=50  --knum=$i --weight=100 --gpu=3 --patchsize=256 --stride=40 --RECONGAU --Aloss=SIGRMSE --labmda=1. --back_filter --partial_recon --mask_trshold=0.2 --cross_validation --start_lr=1e-4 --end_lr=1e-5 
done 
