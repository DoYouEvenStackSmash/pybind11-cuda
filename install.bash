git submodule init
git submodule update
cd build
cmake ..
make -j4
export PYTHONPATH="$PWD:$PYTHONPATH"
cd ..
