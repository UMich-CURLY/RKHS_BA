#include <utility>



namespace cvo {
      std::unordered_map<int, std::tuple<uint8_t, uint8_t, uint8_t>> label2color;
    label2color[0]  =std::make_tuple(128, 64,128 ); // road
    label2color[1]  =std::make_tuple(244, 35,232 ); // sidewalk
    label2color[2]  =std::make_tuple(70, 70, 70 ); // sidewalk
    label2color[3]  =std::make_tuple(102,102,156   ); // building
    label2color[4] =std::make_tuple(190,153,153 ); // pole
    label2color[5] =std::make_tuple(153,153,153  ); // sign
    label2color[6]  =std::make_tuple(250,170, 30   ); // vegetation
    label2color[7]  =std::make_tuple(220,220,  0   ); // terrain
    label2color[8] =std::make_tuple(107,142, 35 ); // sky
    label2color[9]  =std::make_tuple(152,251,152 ); // water
    label2color[10]  =std::make_tuple(70,130,180  ); // person
    label2color[11]  =std::make_tuple( 220, 20, 60   ); // car
    label2color[12]  =std::make_tuple(255,  0,  0  ); // bike
    label2color[13] =std::make_tuple( 0,  0,142 ); // stair
    label2color[14]  =std::make_tuple(0,  0, 70 ); // background
    label2color[15]  =std::make_tuple(0, 60,100 ); // background
    label2color[16]  =std::make_tuple(0, 80,100 ); // background
    label2color[17]  =std::make_tuple( 0,  0,230 ); // background
    label2color[18]  =std::make_tuple(119, 11, 32 ); // background


}
