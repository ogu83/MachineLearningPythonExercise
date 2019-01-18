def surroundings():
    size = 3
    
    surroundings = []

    for y in range(-1 * size, size+1):
        row=[]
        for x in range(-1 * size, size+1):
            row.append([x,y])

        surroundings.append(row)

    for r in surroundings:
        print(r)

#surroundings()

# import numpy as np
# n = np.load("training_data/4644-1547814600246.npy")
# print(n[0])

#look for distribution of the training data
import os
import matplotlib.pyplot as plt
from statistics import mean

all_files = os.listdir('training_data')

halite_amounts=[]

for f in all_files:
    halite_amount = int(f.split("-")[0])
    halite_amounts.append(halite_amount)

plt.hist(halite_amounts)
plt.show()

print("Count: ", len(halite_amounts))
print("Mean: ", mean(halite_amounts))