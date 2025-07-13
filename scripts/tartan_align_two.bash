cd build
make -j6
cd ..

disk="/home/rayzhang/media/"

#sky=130
#dataset_folder=${disk}/tartanair/soulcity/Easy/P001/

sky=196
dataset_folder=${disk}/tartanair/abandonedfactory/Easy/P006/
source_inds=(198)
target_inds=(199)

for ind in ${!source_inds[@]}
do
#	break
	./build/bin/cvo_align_gpu_two_tartan --data_type tartan_rgbd \
                                     --dataset_path $dataset_folder \
                                     --sky_index $sky \
                                     --cvo_param_file cvo_params/cvo_outdoor_params.yaml \
                                     --source_index ${source_inds[ind]} \
                                     --target_index ${target_inds[ind]} \
                                     --is_depth_filtering 1 \
                                     --depth_normal_ell 0.01 \
                                     --depth_dir_ell 0.25
mv before_align.pcd before_align_SH003_${source_inds[ind]}_${target_inds[ind]}.pcd
mv after_align.pcd after_align_SH003_${source_inds[ind]}_${target_inds[ind]}.pcd
done


sky=-1
dataset_folder=${disk}/tartanair/carwelding/Easy/P004
source_inds=(146 148)
target_inds=(300 308)
for ind in ${!source_inds[@]}
do
    break
	./build/bin/cvo_align_gpu_two_tartan --data_type tartan_rgbd \
                                     --dataset_path $dataset_folder \
                                     --sky_index $sky \
                                     --cvo_param_file cvo_params/cvo_outdoor_params.yaml \
                                     --source_index ${source_inds[ind]} \
                                     --target_index ${target_inds[ind]}
mv before_align.pcd before_align_carwelding_${source_inds[ind]}_${target_inds[ind]}.pcd
mv after_align.pcd after_align_carwelding_${source_inds[ind]}_${target_inds[ind]}.pcd
done

sky=-1
dataset_folder=${disk}/tartanair/SH003/
source_inds=(183 383 380 619 673)
target_inds=(256 544 93 675 0)
for ind in ${!source_inds[@]}
do
	break
	./build/bin/cvo_align_gpu_two_tartan --data_type tartan_rgbd \
                                     --dataset_path $dataset_folder \
                                     --sky_index $sky \
                                     --cvo_param_file cvo_params/cvo_outdoor_params.yaml \
                                     --source_index ${source_inds[ind]} \
                                     --target_index ${target_inds[ind]}
mv before_align.pcd before_align_SH003_${source_inds[ind]}_${target_inds[ind]}.pcd
mv after_align.pcd after_align_SH003_${source_inds[ind]}_${target_inds[ind]}.pcd
done



sky=196
dataset_folder=${disk}/tartanair/abandonedfactory/Easy/P006/

source_inds=(518 242 641)
target_inds=(180 332 681)
for ind in ${!source_inds[@]}
do
	break
./build/bin/cvo_align_gpu_two_tartan --data_type tartan_rgbd \
                                     --dataset_path $dataset_folder \
                                     --sky_index $sky \
                                     --cvo_param_file cvo_params/cvo_outdoor_params.yaml \
                                     --source_index ${source_inds[ind]} \
                                     --target_index ${target_inds[ind]}
mv before_align.pcd before_align_abandonedfactory_${source_inds[ind]}_${target_inds[ind]}.pcd
mv after_align.pcd after_align_abandonedfactory_${source_inds[ind]}_${target_inds[ind]}.pcd

done



