mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j6
cd ..
mkdir -p build_debug
cd build_debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j6
cd ..
export CUDA_VISIBLE_DEVICES=0
dtype=$1
date=$2
clear

    #skylabel=(196 112 -- 130  196 146 130)
    #seqs=( 05 00 07 06 09 )
    
    #seqs=( 07 05 00 02 06 08 09  )
    #seqs=( 07 09 05 06 00 02 08 )
    seqs=(   07 05 09 00 08 02 )
    for ind in ${!seqs[@]}
    do
        i=${seqs[ind]}
        echo " Current Seq: ${i}"        
        #sky=${skylabel[ind]}
	folder=${dtype}_${i}_${date}
	#track_folder=/home/rzh/slam_eval/result_floam/${i}/
	#track_folder=/home/rzh/slam_eval/result_floam/${i}/
        #dataset_folder=/home/`whoami`/media/Samsung_T5/${dtype}/dataset/sequences/${i}/
	dataset_folder=/home/rzh//media/sdg1/rzh/${dtype}/dataset/sequences/${i}/
        #lc_file=/home/`whoami`/unified_cvo/demo_data/kitti_loop_closure/kitti_${i}.txt
        lc_file=/home/rayzhang/unified_cvo/demo_data/kitti_loop_closure/kitti_${i}_loop_closure.g2o

	rm -rf $folder
	mkdir -p $folder
	#cp results/dso/${i}.txt $folder/tracking_full.txt 
	#cp results/cvo_geometric_img_gpu0_mar21/${i}.txt $folder/tracking_full.txt        
	#cp results/cvo_intensity_lidar_jan23/${i}.txt $folder/tracking_full.txt        
	#cp results/mulls_no_loop/${i}/${i}.txt $folder/tracking_full.txt        
        #lc_file=results/mulls_with_loop/${i}/loop_pairs.txt 
	#cp results/mulls_no_loop/${i}/gt.txt $folder/tracking_full.txt        
	#cp ground_truth/${i}.txt $folder/tracking_full.txt        
	#cp results/cvo_intensity_lidar_jun09/${i}.txt $folder/tracking_full.txt        

	cp results/mulls_with_loop/${i}/${i}_lidar.txt $folder/tracking_full.txt       
        mkdir -p $folder/${i}	
	#cp results/mulls_with_loop/${i}/gt.txt $folder/${i}/gt.txt        
	#cp results/cvo_intensity_lidar_jan23/${i}_gt.txt $folder/${i}/gt.txt        
	cp ground_truth/kitti/lidar/${i}.txt ${folder}/${i}/gt.txt
	#cp results/mulls/${i}.txt $folder/tracking_full.txt        
	#cp cvo_align_lidar_jun05/${i}.txt $folder/tracking_full.txt        
	#cp ${track_folder}/odom${i}kittiAlign.txt $folder/tracking_full.txt        
	last_index=`cat $folder/tracking_full.txt | wc -l`
	last_index="$((${last_index}-1))"
	echo "last index is $last_index"
	rm *.pcd
	mkdir -p ${folder}/pcds
        ### run global BA
        #gdb -ex run --args \
        ./build/bin/cvo_irls_lidar_loop ${dtype} $dataset_folder cvo_params/cvo_irls_kitti_ba_params.yaml 2 $folder/tracking_full.txt $lc_file  ba.txt 0 0 $last_index 1 0.1  0 0 2 0 0 1 #> log_kitti_loop_${i}.txt
	
	
	mv [0-9]*.pcd ${folder}/pcds/
        mv *.pcd $folder/
        mv pgo.txt pgo.g2o global.txt loop_closures.g2o tracking.txt ba.txt pose_iter*.txt err_wrt_*.txt log_kitti*.txt groundtruth.txt $folder/
        cp ${dataset_folder}/poses.txt $folder/
	
	

        # convert traj to kitti format
        #python3 scripts/xyzq2kitti.py ${folder}/groundtruth.txt  ${folder}/groundtruth_kitti.txt #--is-change-of-basis
        #python3 scripts/xyzq2kitti.py ${folder}/tracking.txt  ${folder}/tracking_kitti.txt #--is-change-of-basis
        #python3 scripts/xyzq2kitti.py ${folder}/ba.txt  ${folder}/ba_kitti.txt #--is-change-of-basis
        #python3 scripts/xyzq2kitti.py ${folder}/pgo.txt  ${folder}/pgo_kitti.txt# --is-change-of-basis
        #python3 /home/rayzhang/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref ${folder}/groundtruth_kitti.txt ${folder}/tracking_kitti.txt   ${folder}/ba_kitti.txt  -p --plot_mode xyz

        #mv log_tartan_rgbd_${difficulty}_${i}.txt $folder
        sleep 3
    done

