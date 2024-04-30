__global__ void find_min(double *A, double *vB, double *Aprime, double thres, double lb, double ub, int width, int height, int buflen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int offt = threadIdx.x;
  if ((int)(col/width)== (height - lb)) {
    
    int counter = (col) / width - ub;
    for (int j = 0; j < counter; j++) {
      if (col - j * width < 0)break;
      if (A[col - j * width] > thres) {
        vB[col % width] = Aprime[col  - (j-4) * width];
        break;
      }
    }
  }
}

__global__ void age_out(double* vals, double* val_ages,double* inc_vals, int max_age, int ylen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < ylen) {
    if (val_ages[col] >= max_age) {
      val_ages[col] = 0;
      vals[col] = inc_vals[col];
    }
    val_ages[col]++;
  }
}

__global__ void minimize(double* vals, double* vals_ages, double* inc_vals, int max_age,int ylen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < ylen) {
    if (vals_ages[col] >= max_age || vals[col] >= inc_vals[col]) {
      vals[col] = (double)inc_vals[col];
      vals_ages[col] = 0;
    }
    // vals[col] = vals[col] <= inc_vals[col] ? vals[col] : (double)inc_vals[col];
    vals_ages[col]++;
    
  }
}

__global__ void prewarp(double* A, double* B, double min, double mid, double max_px, double theta_max, double cam_height, int buflen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (min < col && col < max_px) {
    double theta_oc = (col - mid) / max_px * theta_max;
    double range_oc = cam_height / (tan(asin(cam_height / A[col])));
    B[col] = range_oc / cos(theta_oc);
  }
}

__global__ void invert(double *a, double val, int buflen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < buflen) {
    a[col] = val / (a[col]+1);
  }
}
