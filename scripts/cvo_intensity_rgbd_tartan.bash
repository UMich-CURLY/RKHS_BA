#build_dir=build
build_dir=build_debug

export CUDA_VISIBLE_DEVICES=0
cd $build_dir && make -j && cd .. 

date=$1
clear

disk="/home/rayzhang/media/"


    
skylabel=(-1 -1 -1 -1)
seqs=(SH003 SH005 SH006 SH000)
for ind in ${!seqs[@]}
do
    i=${seqs[ind]}
    sky=${skylabel[ind]}
    #pind=${index[ind]}
    folder=tartan_rgbd_test_${i}_${date}
    dataset_folder=${disk}/tartanair/$i/
    echo " Current Seq: ${i} ${difficulty} with sky label ${sky} ${pind}"        
    rm -rf $folder
    mkdir -p $folder
    rm *.pcd
    
    gdb -ex run --args \
    ./$build_dir/bin/cvo_align_gpu_rgbd_tartan $dataset_folder cvo_params/cvo_outdoor_params.yaml \
                                          tartan_rgbd_test_${i}_${date}.txt 0 30000  $sky

    
    mv tartan_rgbd_test_${i}_${date}.txt  $folder/${seq}.txt
    #cp $dataset_folder/poses.txt $folder/gt.txt

done



for difficulty in Easy
do
    break
    echo "new seq $i"

    
    skylabel=(196 196 130)
    seqs=(abandonedfactory seasonsforest soulcity)
    index=(P005 P002 P001)
    for ind in ${!seqs[@]}
    do
        i=${seqs[ind]}
        sky=${skylabel[ind]}
        pind=${index[ind]}
	folder=tartan_rgbd_${difficulty}_${pind}_${i}_${date}
        dataset_folder=${disk}/tartanair/$i/${difficulty}/${pind}/
        echo " Current Seq: ${i} ${difficulty} with sky label ${sky} ${pind}"        
	rm -rf $folder
	mkdir -p $folder
	rm *.pcd
    
    
      ./$build_dir/bin/cvo_align_gpu_rgbd_tartan $dataset_folder cvo_params/cvo_outdoor_params.yaml \
                                            tartan_rgbd_${i}_${pind}_${date}.txt 0 30000  $sky

      
      mv tartan_rgbd_${i}_${pind}_${date}.txt  $folder/${seq}.txt
      cp $dataset_folder/poses.txt $folder/gt.txt

      done
done




for difficulty in Easy #Hard
do
    break

    echo "new seq $i"
    skylabel=(196 112 -- 130  196 146 130)
    seqs=(abandonedfactory gascola hospital seasidetown seasonsforest seasonsforest_winter soulcity)
    for ind in ${!seqs[@]}
    do
        i=${seqs[ind]}
        sky=${skylabel[ind]}
	folder=tartan_rgbd_${difficulty}_${i}_${date}
        dataset_folder=${disk}/tartanair/$i/${difficulty}/P001/
        echo " Current Seq: ${i} ${difficulty} with sky label ${sky}"        
	rm -rf $folder
	mkdir -p $folder
	rm *.pcd
    
    
      ./$build_dir/bin/cvo_align_gpu_rgbd_tartan $dataset_folder cvo_params/cvo_outdoor_params.yaml \
                                            tartan_rgbd_${i}_${date}.txt 0 30000  $sky

      
      mv tartan_rgbd_${i}_${date}.txt  $folder/${seq}.txt
      cp $dataset_folder/poses.txt $folder/gt.txt

      done
done


