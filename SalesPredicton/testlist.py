import numpy as np

arr = np.array([1, 3, 2, 4, 5])

l=list(arr.argsort()[-3:][::-1])
print(arr[l])