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
import numpy as np

n = np.load("training_data/4644-1547814600246.npy")
print(n[0])