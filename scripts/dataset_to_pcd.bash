cd build_debug
make -j12
cd ..


dataset_name=tartanair
dataset_path=/home/rayzhang/media/tartanair/abandonedfactory/Easy/P001/
dtype=2  # 0: LIDAR, 1: STEREO,  2: RGBD

gdb -ex run --args \
./build_debug/bin/pcd_gen $dataset_name $dataset_path $dtype "" 0 3 4 $dataset_path/cvo_calib_deep_depth.txt 
