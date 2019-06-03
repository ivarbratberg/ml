import numpy as np 
import test as ts
samples=np.load("./samples.npz")
X = samples['data']
ts.train_EM(X,3)