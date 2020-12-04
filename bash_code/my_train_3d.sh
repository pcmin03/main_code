cd ..
for i in 3
do
	python mytrain.py --modelname=unet_test --modelname=ResidualUNet3D --datatype=uint16_3d_wise --epochs=801 --activation=sigmoid --batch_size=30  --knum=$i --weight=5 --gpu=1 --patchsize=128 --stride=100 --RECONGAU --Aloss=MSE --labmda=1.  --mask_trshold=0.3 --cross_validation  --deleteall
done 
