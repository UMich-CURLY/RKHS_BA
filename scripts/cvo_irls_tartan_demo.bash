cd build_debug
make -j8
cd ..

cp demo_data/tartan_demo/*.pcd ./

gdb -ex run --args \
./build_debug/bin/cvo_irls_pcd cvo_params/cvo_tartan_demo_params.yaml demo_data/tartan_demo/pose_graph_simple.txt demo_data/tartan_demo/init_pose.txt 

rm 0.pcd 3.pcd 6.pcd 9.pcd
