# look for distribution of the training data
import os
import matplotlib.pyplot as plt
from statistics import mean

all_files = os.listdir('/media/oguz/ParrotExt/HaliteTrainingData1')

halite_amounts=[]

for f in all_files:
    halite_amount = int(f.split("-")[0])
    halite_amounts.append(halite_amount)

plt.hist(halite_amounts)
plt.show()

print("Count: ", len(halite_amounts))
print("Mean: ", mean(halite_amounts))