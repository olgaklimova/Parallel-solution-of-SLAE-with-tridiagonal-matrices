import numpy as np

array1 = []
array2 = []
array3 = []
array4 = []
array5 = []

with open("expected_val.txt") as f1:
 size = f1.readline()
 array1.append([x for x in f1.readline().split()])
 
with open("C_MPI_val.txt") as f2:
 size = f2.readline()
 array2.append([x for x in f2.readline().split()])
if np.array_equal(array1, array2):
 print("C_MPI_TEST COMPLETED")
else:
 print("C_MPI_TEST FAILED")
 
with open("Python_MPI_val.txt") as f3:
 size = f3.readline()
 array3.append([x for x in f3.readline().split()])
if np.array_equal(array1, array3):
 print("Python_MPI_TEST COMPLETED")
else:
 print("Python_MPI_TEST FAILED")

with open("C_OpenMP_val.txt") as f4:
 size = f4.readline()
 array4.append([x for x in f4.readline().split()])
if np.array_equal(array1, array4):
 print("C_OpenMP_TEST COMPLETED")
else:
 print("C_OpenMP_TEST FAILED")
 
with open("C_Pthreads_val.txt") as f5:
 size = f5.readline()
 array5.append([x for x in f5.readline().split()])
if np.array_equal(array1, array5):
 print("C_Pthreads_TEST COMPLETED")
else:
 print("C_Pthreads_TEST FAILED")
