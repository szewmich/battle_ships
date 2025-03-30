import numpy as np
import os

k = '0074707270007470727000747077700074707077007770007207777700727072270077007777000000000000000000707000'
matrix = np.zeros(100)
for n in range(0,100):
    matrix[n] = str(k[n])
matrix = np.array(matrix.reshape(10,10), dtype='int16')
print(matrix[0][0])

print(np.array2string(matrix, separator=', '))