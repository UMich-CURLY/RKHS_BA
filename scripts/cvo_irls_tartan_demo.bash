cd build_debug
make -j8
cd ..

cp demo_data/tartan_demo/*.pcd ./

gdb -ex run --args \
./build_debug/bin/cvo_irls_pcd cvo_params/cvo_tartan_demo_params.yaml demo_data/tartan_demo/pose_graph_50.txt #demo_data/tartan_demo/init_pose.txt 

rm 50.pcd 53.pcd 56.pcd 59.pcd
