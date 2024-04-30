import numpy as np
import numpy_multiply as cuda_processing
import time

# Create two NumPy arrays
# cpu arrays
def test1():
  arr1 = np.random.rand(640*400)*100
  arr2 = np.random.rand(100,640*400)*100
  arr3 = np.zeros(640*400)
  # arr1 = np.array([1.0, 2.0,3.0, 4.0,5.0, 6.0,7.0, 8.0])
  # arr2 = np.array([1.0, 2.0,3.0, 2,5.0, 6.0,7.0, 8.0])+1
  # arr3 = np.array([0,0,0,0,0,0,0,0])+1
  s1 = time.perf_counter()
  for i in range(100):
    mask = arr3 > 5
    arr1[mask] = arr2[i][mask]
    arr3[mask] = 0
    arr1 = np.minimum(arr1, arr2[i])
    arr3+=1
  s2 = time.perf_counter()
  arr1 = np.random.rand(640*400)*100
  arr2 = np.random.rand(100,640*400)*100
  arr3 = np.zeros(640*400)
  print(s2 - s1)
  arr4 = np.zeros(8)
  val0 = cuda_processing.warmup(1)
  vals = cuda_processing.move_to_gpu(arr1)
  new_vals = cuda_processing.move_to_gpu(arr2[0])
  print(arr3)
  print(arr1.shape)
  ages = cuda_processing.move_to_gpu(arr3)
  print(arr4)
  s1x = time.perf_counter()
  for i in range(99):
    cuda_processing.direct_minimize_wrap(vals, ages, new_vals, 4,2, 8)
    cuda_processing.move_to_gpu_addr(new_vals, arr2[i+1])
  s2x = time.perf_counter()
  print(s2x - s1x)
  # cuda_processing.move_to_cpu(ages, arr4)
  # print(arr4)
# # move to gpu
# # ptr1 = move_to_gpu(arr1)
def arrprint(arr1,cols, rows):
  for i in range(rows):
    for j in range(cols):
      # arr1[i * cols + j] = i * cols + j
      print(arr1[i * cols + j],end='\t')
    print("")
# # ptr 1 is no longer the same as arr1, operations should use that
# # any time i want to see the contents of ptr1 i have to move it back to cpu
def test2():
  numiter = 10
  conv = lambda x, h: np.apply_along_axis(
      lambda x: np.convolve(x, h.flatten(), mode="full"), axis=0, arr=x
  )
  def get_vec_filter(f):
      return f[:, np.newaxis]
  deriv_f = get_vec_filter(np.array([0.000133831,0.00443186,0.0539911,0.241971,0.398943,0.241971,0.0539911,0.00443186,0.000133831]))



  flt = np.array([0.000133831,0.00443186,0.0539911,0.241971,0.398943,0.241971,0.0539911,0.00443186,0.000133831])
  cols = 640
  rows = 400
  arr1 = np.zeros(rows*cols)

  arr2 = np.zeros(rows*cols)
  arr3 = np.zeros(rows * cols)
  # print(arr1)
  for i in range(rows):
    for j in range(cols):
      arr1[i * cols + j] = i * cols + j
  val0 = cuda_processing.warmup(1)
  # cuda_processing.move_to_gpu_addr(val1, arr1)
  val1 = cuda_processing.move_to_gpu(arr1)
  val2 = cuda_processing.move_to_gpu(arr2)
  val3 = cuda_processing.move_to_gpu(arr3)
  val4 = cuda_processing.move_to_gpu(flt)
  idxarr = [val1, val2, val3]
  c = 0
  idr = lambda x: x+1 if x+1 < 3 else 0
  s1x = time.perf_counter()
  for i in range(numiter):

    cuda_processing.direct_conv_wrap(idxarr[c],idxarr[idr(c)],val4, flt.shape[0], cols, rows,arr3.shape[0])
    c = idr(c)

  s2x = time.perf_counter()
  print(f"GPU CONV: {s2x - s1x}")
  arr1x = arr1.reshape(rows, cols)
  #print(arr1)
  #print(arr1x)
  # time.start()
  # s1 = time.perf_counter()
  s1 = time.perf_counter()
  for i in range(numiter):
    arrr1x = conv(arr1x, deriv_f)
    
  s2 = time.perf_counter()
  print(f"CPU CONV: {s2 - s1}")
  # print(val)
      # arr1[i * rows + j] = i*rows+j
  # for i in rang

  print((s2 - s1)/(s2x - s1x))
  #cuda_processing.move_to_cpu(val2, arr3)
  #arrprint(arr3, cols, rows)

def test3():
  sz = 10
  arr1 = np.random.rand(sz*sz)*5
  
  arr2 = np.zeros(sz)
  val0 = cuda_processing.warmup(1)
  # cuda_processing.move_to_gpu_addr(val1, arr1)
  val1 = cuda_processing.move_to_gpu(arr1)
  val2 = cuda_processing.move_to_gpu(arr2)
  cuda_processing.find_min_wrap(val1, val2, 1, 3 ,sz, sz)
  arrprint(arr1, sz, sz)
  cuda_processing.move_to_cpu(val2, arr2)
  print(arr2)
  # arrprint(arr1)
  
def main():
  test3()

if __name__ == '__main__':
  main()
# print(arr3)
# cuda_processing.free_gpu(val1)
# cuda_processing.free_gpu(val2)
# cuda_processing.free_gpu(val4)

# flt = np.array([1,1,1,])
# for (int i = 0; i < row; i++) {
#   for (int j = 0; j < col; j++) {
#     printf("%.2f\t", A[i*col+j]);
#   }
#   printf("\n");
# }
# print(arr1)
# cuda_conv(arr1, arr2, flt)
# print(arr1)
# # arr2 = np.array([5.0, 6.0,7.0, 8.0])

# # # Set the length of the array
# # length = 20000

# # # Generate a random NumPy array of specified length
# # arr1 = np.random.rand(length, length)

# # arr2 = np.random.rand(length)
# # Call the C++ function to perform array multiplication

# print(val)

# s1 = time.perf_counter()
# print(arr1)

# cuda_processing.array_multiply(arr1, arr2)
# s2 = time.perf_counter()
# # print(s2 - s1)
# # result = arr1 * arr2
# # s3 = time.perf_counter()
# print(cuda_processing.a_number())
# print(f"time to run first {s2 - s1}")#\ntime to run secn {s3 - s2}")
# print(arr1)
# # result2

# # Print the result
# # print("Result of array multiplication:")
# # print(result)
