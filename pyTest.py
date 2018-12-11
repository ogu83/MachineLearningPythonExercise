import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

X = [[i/1000,(i/1000)**0.5] for i in range(1000,4000,1)]
Xs = [X[i][0] for i in range(len(X))]
Ys = [X[i][1] for i in range(len(X))]

plt.scatter(Xs,Ys,s=1)
plt.show()