# pybind11-cuda
Starting point for GPU accelerated python libraries 

Prerequisites -
Cuda installed in /usr/local/cuda
Python 3.6 or greater
Cmake 3.6 or greater

To build - 
```source install.bash```
Test it with
```python3 test_mul.py``` 
 
# Features demonstrated
Compiles out of the box with cmake
Numpy integration
C++ Templating for composable kernels with generic data types

Originally based on https://github.com/torstem/demo-cuda-pybind11
