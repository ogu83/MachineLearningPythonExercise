import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATA_PATH = "D:\\LANLEarthquakeData"
TRAIN_DATA_PATH = f"{DATA_PATH}\\train.csv"

traindata = pd.read_csv(TRAIN_DATA_PATH, nrows = 1000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

acustic_data = traindata.iloc[:,0]
time_to_failure = traindata.iloc[:,1]

for t in range(1, len(time_to_failure)):
    if round(time_to_failure[t-1], ndigits=5) != round(time_to_failure[t], ndigits=5):
        print(t, time_to_failure[t])

fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Trends of acoustic_data and time_to_failure. 2% of data (sampled)")
plt.plot(acustic_data, color='b')
ax1.set_ylabel('acoustic_data', color='b')
#plt.legend(['acoustic_data'])
ax2 = ax1.twinx()
plt.plot(time_to_failure, color='g')
ax2.set_ylabel('time_to_failure', color='g')
#plt.legend(['time_to_failure'], loc=(0.875, 0.9))
plt.grid(False)
plt.show()

del acustic_data
del time_to_failure

#fig, ax1 = plt.subplots()
#color = 'tab:gray'
#ax1.set_ylabel('acustic_data', color = color)
#ax1.plot(acustic_data, color = color)
#ax1.tick_params(axis='y', labelcolor = color)

#ax2 = ax1.twinx()

#color = 'tab:blue'
#ax2.set_ylabel('time_to_failure', color = color)
#ax2.plot(time_to_failure, color = color)
#ax2.tick_params(axis='y', labelcolor = color)

#fig.tight_layout()

#plt.show()
