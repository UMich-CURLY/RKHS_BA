build_dir=build
#build_dir=build_debug

cd $build_dir
make -j
cd ..

date=$1
clear


disk="/home/rayzhang/media/"


for difficulty in Easy #Hard
do
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
       
	#folder=tartan_rgbd_${difficulty}_${i}_${}_${date}
        #dataset_folder=${disk}/tartanair/$i/${difficulty}/P001/
        echo " Current Seq: ${i} ${difficulty} with sky label ${sky}"        
	#rm -rf $folder
	mkdir -p $folder
	rm *.pcd
        
	gdb -ex run --args \
        ./$build_dir/bin/cvo_irls_tartan_ba_loop \
            --data_type tartan_rgbd \
            --data_path $dataset_folder \
            --sky_label $sky \
            --cvo_param_file cvo_params/cvo_outdoor_params.yaml \
            --num_neighbors_per_node 2 \
            --tracking_traj_file $folder/tracking_input.txt \
            --loop_closure_input_file $folder/lc.txt \
            --BA_traj_file $folder/ba.txt \
            --is_edge_only 1 \
            --start_ind 0 \
            --max_last_ind 10000 \
            --cov_scale_t 1.0 \
            --cov_scale_r 0.05 \
            --num_merging_sequential_frames 0 \
            --is_doing_pgo 1 \
            --is_read_loop_closure_poses_from_file 0 \
            --is_store_pcd_each_frame 0 \
            --is_global_registration 1 \
            --is_doing_ba 0
        
        #gdb -ex run --args \
        #./build/bin/cvo_irls_tartan_odom $dataset_folder cvo_params/cvo_outdoor_params.yaml cvo_calib_deep_depth.txt 4 tracking.txt ba.txt 0 0 100000 $sky # > log_tartan_rgbd_${difficulty}_${i}.txt
        mv *.pcd $folder/
        mv tracking.txt err_wrt_iters_*.txt groundtruth.txt $folder/
        cp ${dataset_folder}/pose_left.txt $folder/

        # convert traj to kitti format
        python3 scripts/xyzq2kitti.py ${folder}/groundtruth.txt  ${folder}/groundtruth_kitti.txt
        python3 scripts/xyzq2kitti.py ${folder}/tracking.txt  ${folder}/tracking_kitti.txt
        python3 scripts/xyzq2kitti.py ${folder}/ba.txt  ${folder}/ba_kitti.txt
        #python3 /home/rayzhang/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref ${folder}/groundtruth_kitti.txt ${folder}/tracking_kitti.txt   ${folder}/ba_kitti.txt  -p --plot_mode xyz

        #mv log_tartan_rgbd_${difficulty}_${i}.txt $folder
        sleep 3
        break
    done
done





cd build
make -j
cd ..

date=$1
clear

    #skylabel=(196 112 -- 130  196 146 130)
    seqs=( 03 )
    for ind in ${!seqs[@]}
    do
        break
        i=${seqs[ind]}
        #sky=${skylabel[ind]}
	folder=kitti_lidar_${i}_${date}
        dataset_folder=/home/rayzhang/media/Samsung_T5/kitti_lidar/dataset/sequences/$i/
        echo " Current Seq: ${i}"        
	rm -rf $folder
	mkdir -p $folder
	rm *.pcd

        #gdb -ex run --args \
        ./build/bin/cvo_irls_lidar_ba $dataset_folder cvo_params/cvo_irls_kitti_ba_params.yaml 4 tracking.txt ba.txt 0 720 725 # > log_tartan_rgbd_${difficulty}_${i}.txt
        mv *.pcd $folder/
        mv tracking.txt ba.txt err_wrt_iters_*.txt groundtruth.txt $folder/
        cp ${dataset_folder}/poses.txt $folder/

        # convert traj to kitti format
        python3 scripts/xyzq2kitti.py ${folder}/groundtruth.txt  ${folder}/groundtruth_kitti.txt
        python3 scripts/xyzq2kitti.py ${folder}/tracking.txt  ${folder}/tracking_kitti.txt
        python3 scripts/xyzq2kitti.py ${folder}/ba.txt  ${folder}/ba_kitti.txt
        #python3 /home/rayzhang/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref ${folder}/groundtruth_kitti.txt ${folder}/tracking_kitti.txt   ${folder}/ba_kitti.txt  -p --plot_mode xyz

        #mv log_tartan_rgbd_${difficulty}_${i}.txt $folder
        sleep 3

    done

