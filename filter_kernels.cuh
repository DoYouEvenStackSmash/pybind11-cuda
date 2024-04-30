__global__ void coharmonic_mean(double *g, double *f, int x, int y, double Q, int buflen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col+y < buflen && col - x > 0) {
    double sum = 0.0;
    for (int s = col-x; s < col + y ; s++) {
      sum = sum + g[s];
    }
    f[col] = pow(sum, Q+1) / pow(sum, Q);
  }
}

__global__ void conv(double* A,double* B, double* H, int hlen, int ylen,int width, int height, int lb, int span) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // int row = blockIdx.y * blockDim.y + threadIdx.y;
  int i = col;
  if (col < ylen && col >= lb * width) {
    double sum = 0.0f;
    for (int j = 0; j < hlen; j++) {
      if (i-j*width < 0)
        break;
      sum = sum + (double) H[j] * A[i - j * width];
    }
    B[i] = sum;
  }
}