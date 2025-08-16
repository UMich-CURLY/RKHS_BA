cd build
make -j12
cd ..


dataset_name=$1 #tartanair
dataset_path=$2
dtype=2

./build/bin/pcd_gen $dataset_name $dataset_path $dtype "" 0 3 4 $dataset_path/cvo_calib_deep_depth.txt 
