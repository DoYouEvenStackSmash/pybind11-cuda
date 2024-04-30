__global__ void c_add(double* buf1, double* buf2,int bufcount, int buflen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < buflen) {
    buf1[col] = buf1[col] + buf2[col];
  }
}

__global__ void scalar_divide(double* a, double val, int buflen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < buflen) {
    a[col] = a[col] / val;
  }
}

