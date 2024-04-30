#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// #include <stdio.h>
// #include <stdlib.h>
#include <cuda_runtime.h>
// // Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
namespace py = pybind11;

#include "camera_kernels.cuh"
#include "scalar_kernels.cuh"
#include "filter_kernels.cuh"

void coharmonic_mean_wrap(uint64_t g, uint64_t f, int x, int y, double Q, int width, int height, int buflen) {
  coharmonic_mean<<<width, height>>>((double *)g, (double *)f, x, y, Q, buflen);
}


void find_min_wrap(uint64_t ptr1, uint64_t ptr2, uint64_t back_ptr, double thres, int lb, int ub, int width, int height) {
  find_min<<<width, height>>>((double*)ptr1, (double*)ptr2, (double*)back_ptr, thres, lb, ub, width, height, width*height);
}


void prewarp_wrap(uint64_t ptr1, uint64_t ptr2, double min, double mid, double max_px, double theta_max, double cam_height, int buflen) {
  prewarp<<<20,32>>>((double*)ptr1, (double*)ptr2, min, mid, max_px, theta_max, cam_height, buflen);
}

void conv_wrap(double* A,double *B, double *H, int hlen, int ylen,int width, int height,int lb,int span) {
  int threads = min(1024,(height + height%32));
  int off_t = 0;
  if (threads == 1024) {
    off_t = height - threads;
  }
  // int blocks = ceil(ylen / min(height, threads))
  conv<<<(width+off_t+width%32),min(1024,(height + height%32))>>>(A,B,H, hlen, ylen,width,height,lb,span);
}


void invert_wrap(uint64_t ptr,double val, double width, double height) {
  invert<<<width, height>>>((double*)ptr, val, width*height);
}

void scalar_divide_wrap(uint64_t ptr, double val, double width, double height) {
  scalar_divide<<<width, height>>>((double*)ptr, val, width*height);
}

uint64_t move_to_gpu(py::array_t<double> arr1) {
  auto buf1 = arr1.request();
  double *ptr1 = (double *)buf1.ptr;
  double *buf1_gpu;

  int buflen = buf1.shape[0];
  
  cudaMalloc((void**)&buf1_gpu, buflen * sizeof(double));
  cudaMemcpy(buf1_gpu, ptr1, buflen * sizeof(double),cudaMemcpyHostToDevice);
  
  return (uint64_t)buf1_gpu;
}

void move_to_gpu_addr(uint64_t ptr, py::array_t<double> arr1,int sz) {
  auto buf1 = arr1.request();
  double *ptr1 = (double *)buf1.ptr;
  double *buf1_gpu = (double*)ptr;

  int buflen = sz;
  
  // cudaMalloc((void**)&buf1_gpu, buflen * sizeof(double));
  cudaMemcpy(buf1_gpu, ptr1, buflen * sizeof(double),cudaMemcpyHostToDevice);
}

void move_to_cpu(uint64_t ptr, py::array_t<double> arr1) {
  auto buf1 = arr1.request();
  double *ptr1 = (double *)buf1.ptr;
  double *buf1_gpu = (double*)ptr;
  int buflen = buf1.shape[0];
  //memcpy((void**)&(buf1.ptr), (void**)&buf1_gpu, sizeof(double*));
  cudaMemcpy(ptr1, buf1_gpu,buflen * sizeof(double),cudaMemcpyDeviceToHost);
  // return 
}

void cuda_conv(py::array_t<double> arr1, py::array_t<double> arr2, py::array_t<double> filter) {

  // Get pointers to the data
  auto buf1 = arr1.request(), buf2 = arr2.request(), buf3 = filter.request();
  double *ptr1 = (double *)buf1.ptr, *ptr2 = (double *)buf2.ptr, *ptr3 = (double *)buf3.ptr;
  
  conv_wrap(ptr1, ptr2, ptr3, buf3.shape[0], buf1.shape[0]*buf1.shape[1], buf1.shape[0], buf1.shape[1], 0, 0);
}

void direct_conv_wrap(uint64_t ptr1, uint64_t ptr2, uint64_t filter_ptr, int hlen,int width, int height, int buflen) {
  conv_wrap((double*)ptr1, (double*)ptr2, (double*)filter_ptr, hlen, buflen, width, height, 0, 0);
}

void c_add_wrap(uint64_t ptr1, uint64_t ptr2, int width, int height, int buflen) {
  c_add<<<height, width>>>((double*)ptr1, (double*)ptr2, width+height, buflen);
}

void age_out_wrap(uint64_t vals, uint64_t val_ages, uint64_t inc_vals, int max_age, int height, int width, int buflen) {
  age_out<<<height, width>>>((double*)vals, (double*)val_ages, (double*)inc_vals, max_age,buflen);
}

void minimize_wrap(double* vals, double* val_ages, double* new_vals, int lifetime, int width, int height, int buflen){
  int threads = 1024;
  int off_t = 0;
  if (threads == 1024) {
    off_t = buflen/1024;
  }
  // int blocks = ceil(ylen / min(height, threads))
  minimize<<<width,height>>>(vals,val_ages,new_vals, lifetime, width*height);
}

void direct_minimize_wrap(uint64_t vals,uint64_t ages,uint64_t new_vals, int lifetime,int width, int height, int buflen) {
  minimize_wrap((double*)vals, (double*)ages, (double*)new_vals, lifetime, width, height, buflen);
}

uint64_t warmup(int size) {
  int devID = gpuGetMaxGflopsDeviceId();
  checkCudaErrors(cudaSetDevice(devID));
  int major = 0, minor = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
  checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
         devID, _ConvertSMVer2ArchName(major, minor), major, minor);
  double *buf1_gpu;
  cudaMalloc((void**)&buf1_gpu, size * sizeof(double));
  return (uint64_t)buf1_gpu;
}

void free_gpu(uint64_t ptr) {
  cudaFree((double*)ptr);
}

uint64_t a_number() {
  return rand();
}

// Binding code
PYBIND11_MODULE(numpy_multiply, m) {
    m.def("a_number", a_number, "return a number");
    m.def("warmup", warmup, "warmup the gpu");
    m.def("move_to_gpu", move_to_gpu, "move an array to the gpu");
    m.def("move_to_cpu", move_to_cpu, "move an array to the cpu");
    m.def("free_gpu", free_gpu, "free gpu memory");
    m.def("direct_conv_wrap",direct_conv_wrap,"direct_conv wrapper");
    m.def("move_to_gpu_addr", move_to_gpu_addr, "moves to an address on gpu");
    m.def("minimize_wrap", minimize_wrap, "wrapper for element wise minimization");
    m.def("direct_minimize_wrap", direct_minimize_wrap, "direct wrapper for minimize_wrap");
    m.def("scalar_divide_wrap", scalar_divide_wrap,"wrapper for scalar division");
    m.def("find_min_wrap", find_min_wrap, "wrapper for finding obstacle min");
    m.def("invert_wrap", invert_wrap, "inverted division");
    m.def("prewarp_wrap", prewarp_wrap, "prewarp wrapper function");
    m.def("coharmonic_mean_wrap",coharmonic_mean_wrap,"coharmonic mean filter");
    m.def("c_add_wrap",c_add_wrap,"addition in parallel");
    m.def("age_out_wrap", age_out_wrap, "age out wrapper function");
}
