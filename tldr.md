
clear
cd ..
rm -r build

mkdir build && cd build
cmake ..
make -j

./PCBSegment