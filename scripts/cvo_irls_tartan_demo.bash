clear

cd build && make -j && cd .. 

for difficulty in Easy
do
    skylabel=(196 112 -- 130  196 146 130)
    #seqs=(abandonedfactory gascola hospital seasidetown seasonsforest seasonsforest_winter soulcity)
    seqs=(abandonedfactory)
    for ind in ${!seqs[@]}
    do
        for angle in 30
        do
                
                i=${seqs[ind]}
                sky=${skylabel[ind]}
	        folder=${noise_type}_semantic/$angle/tartan_ablation_${difficulty}_${i}_${date}
                dataset_folder=/home/rayzhang/media/tartanair/$i/${difficulty}/P001/
                echo " Current Seq: ${i} ${difficulty} with sky label ${sky}"        
	        rm -rf $folder
	        mkdir -p $folder
	        rm *.pcd
                rm *.png
                

                    ./build/bin/cvo_irls_tartan_ablation $dataset_folder cvo_params/cvo_tartan_semantic_params.yaml cvo_calib_deep_depth.txt 4 $folder $sky 1.0 0.1 10.0  3 3 100000 0.0 0.0 
""

                    mv *.pcd err_*.txt pose_iter*.txt *.png $folder/
                    cp demo_data/tartan_demo/$i.txt $folder/

                    
        done
    done
done
