import numpy as np
from numpy import linalg as LA

a = np.arange(81)
print(a)
print(LA.norm(a))

b = a.reshape(9,9)
print(b)
print(LA.norm(b))