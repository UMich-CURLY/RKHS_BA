export CUDA_VISIBLE_DEVICES=0
cd build && make -j && cd .. && \
#for i in 05
#for i in 01  04 07 10 05 02 00 03 06 08 09 
for i in 01 04 02 05 07 10  	
do
	echo "new seq $i"
    ./build/bin/cvo_align_gpu_semantic_img /home/v9999/media/seagate_2t/kitti/stereo/$i cvo_params/cvo_semantic_params_img_gpu0.yaml \
                                       cvo_img_semantic_gpu0_$i.txt 0 300000
 	
done
