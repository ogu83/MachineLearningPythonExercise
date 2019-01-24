import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATA_PATH = "/media/oguz/ParrotExt/LANLEarthquakeData"
TRAIN_DATA_PATH = f"{DATA_PATH}/train.csv"

traindata = pd.read_csv(TRAIN_DATA_PATH)

print(traindata.head())

# plt.plot(X)
# plt.ylabel('acustic')
# plt.show()