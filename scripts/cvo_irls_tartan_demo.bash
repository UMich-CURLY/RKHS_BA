cd build
make -j8
cd ..


./build/bin/cvo_irls_pcd cvo_params/cvo_outdoor_params.yaml demo_data/tartan_demo/pose_graph.txt demo_data/tartan_demo/init_pose.txt 
