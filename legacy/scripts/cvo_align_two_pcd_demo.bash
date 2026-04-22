cd build_debug
make -j8
cd ..

gdb -ex run --args \
./build_debug/bin/cvo_align_color_pcd_shared_mem  demo_data/source.pcd  demo_data/target.pcd  cvo_params/cvo_outdoor_params.yaml 
