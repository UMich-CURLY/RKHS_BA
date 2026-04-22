date=$1
clear

cd build && make -j && cd .. 
cd build_debug && make -j && cd .. 

noise_type=time_ablation
for difficulty in Easy #Hard
do
    skylabel=(196 112 -- 130  196 146 130)
    seqs=(abandonedfactory gascola hospital seasidetown seasonsforest seasonsforest_winter soulcity)
    #noise=(semantic geometric)
    for ind in ${!seqs[@]}
    do
        for init_angle in 0.027 0.054 0.108 0.2 
        do
                
                i=${seqs[ind]}
                sky=${skylabel[ind]}
	        folder=${noise_type}_color/$init_angle/tartan_ablation_${difficulty}_${i}_${date}
                dataset_folder=/home/rayzhang/media/tartanair/$i/${difficulty}/P001/
                tracking_file=tartan_rgbd_${difficulty}_${i}_oct27/cvo.txt
                echo " Current Seq: ${i} ${difficulty} with sky label ${sky}"        
	        rm -rf $folder
	        mkdir -p $folder
	        rm *.pcd
                rm *.png
                

                #gdb  -ex run  --args \
                    ./build/bin/cvo_irls_tartan_ablation $dataset_folder cvo_params/cvo_tartan_color_params.yaml cvo_calib_deep_depth.txt 4 $folder $sky 1.0 ${noise_sigma} 10.0  3 3 100000 0.0 0.0 $tracking_file $init_angle 0.0  # > log_tartan_rgbd_${difficulty}_${i}.txt
                    mv *.pcd err_*.txt pose_iter*.txt *.png $folder/
                    cp ${dataset_folder}/pose_left.txt $folder/

                    #python3 /home/rayzhang/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref ${folder}/groundtruth_kitti.txt ${folder}/tracking_kitti.txt   ${folder}/ba_kitti.txt  -p --plot_mode xyz

                    #mv log_tartan_rgbd_${difficulty}_${i}.txt $folder
                    sleep 3
                    
        done
    done
done



for difficulty in Easy #Hard
do
    skylabel=(196 112 -- 130  196 146 130)
    seqs=(abandonedfactory gascola hospital seasidetown seasonsforest seasonsforest_winter soulcity)
    #noise=(semantic geometric)
    for ind in ${!seqs[@]}
    do
        for angle in 5 10 20 40
        do
                
                i=${seqs[ind]}
                sky=${skylabel[ind]}
	        folder=${noise_type}_semantic/$angle/tartan_ablation_${difficulty}_${i}_${date}
                dataset_folder=/home/rayzhang/media/tartanair/$i/${difficulty}/P001/
                tracking_file=tartan_rgbd_${difficulty}_${i}_oct27/cvo.txt
                echo " Current Seq: ${i} ${difficulty} with sky label ${sky}"        
	        rm -rf $folder
	        mkdir -p $folder
	        rm *.pcd
                rm *.png
                

                #gdb  -ex run  --args \
                    ./build/bin/cvo_irls_tartan_ablation $dataset_folder cvo_params/cvo_tartan_semantic_params.yaml cvo_calib_deep_depth.txt 4 $folder $sky 1.0 ${noise_sigma} 10.0  3 3 100000 0.0 0.0 tracking_file  # > log_tartan_rgbd_${difficulty}_${i}.txt
                    mv *.pcd err_*.txt pose_iter*.txt *.png $folder/
                    cp ${dataset_folder}/pose_left.txt $folder/

                    #python3 /home/rayzhang/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref ${folder}/groundtruth_kitti.txt ${folder}/tracking_kitti.txt   ${folder}/ba_kitti.txt  -p --plot_mode xyz

                    #mv log_tartan_rgbd_${difficulty}_${i}.txt $folder
                    sleep 3
                    
        done
    done
done
