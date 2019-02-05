import numpy as np 
import pandas as pd
import os
from tqdm import tqdm

DATA_PATH = "D:\\LANLEarthquakeData"
TRAIN_DATA_PATH = f"{DATA_PATH}\\train.csv"
TEST_DATA_PATH = f"{DATA_PATH}\\test"
SUBMISSON_PATH = f"{DATA_PATH}\\sample_submission.csv"
NP_DATA_PATH = f"{DATA_PATH}\\np"
ACOUSTICDATA_NPY_PATH = NP_DATA_PATH + "\\acoustic_data.npy"
TIME_TO_FAILURE_NPY_PATH = NP_DATA_PATH + "\\time_to_failure.npy"
TRAIN_NP_PATH = NP_DATA_PATH + "\\train2d.npy"
PREFIX_NP_PATH = NP_DATA_PATH + "\\prefix2d.npy"
ERROR_REPORT_PATH = f"{DATA_PATH}\\train_yvz_kernel_error_report.pickle"
TIMEP_PATH = f"{NP_DATA_PATH}\\timep.npy"
ERRORS_PATH = f"{NP_DATA_PATH}\\errors.npy"


AcousticData = np.array([], dtype=np.int16)
TimeToFailure = np.array([], dtype=np.float32)

if os.path.exists(TRAIN_NP_PATH):
    train_data = np.load(TRAIN_NP_PATH);
    print("Traing 2d Np Files Loaded")
else:
    if os.path.exists(ACOUSTICDATA_NPY_PATH) and os.path.exists(TIME_TO_FAILURE_NPY_PATH):
        AcousticData = np.load(ACOUSTICDATA_NPY_PATH)
        TimeToFailure = np.load(TIME_TO_FAILURE_NPY_PATH)    
        print("Acoustic and Time To Fail Np Files Loaded")
    else:
        print("Build Acustic and Time To Fail")
        chunks = pd.read_csv(TRAIN_DATA_PATH, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}, chunksize= 10 ** 6)    
        for chunk in tqdm(chunks):        
            AcousticData = np.append(AcousticData, np.array(chunk['acoustic_data']))
            TimeToFailure = np.append(TimeToFailure, np.array(chunk['time_to_failure']))

        np.save(ACOUSTICDATA_NPY_PATH, AcousticData)
        np.save(TIME_TO_FAILURE_NPY_PATH, TimeToFailure)
        print("Acoustic and Time To Fail Np Files Saved")

    print("Build 2d Train Data")
    train_data = np.column_stack((AcousticData, TimeToFailure))
    np.save(TRAIN_NP_PATH, train_data)
    print("Train 2d Np Files Saved")

#print(train_data)

prefix_accs = []
if os.path.exists(PREFIX_NP_PATH):
    prefix_accs = np.load(PREFIX_NP_PATH)
    print("prefix accs loaded")
else:
    unique_accs = np.unique(train_data[:,0])
    #print(len(unique_accs))
    #print(unique_accs)

    for acc in tqdm(unique_accs):
        filtered = np.where(train_data[:,0] == acc)
        values = (train_data[filtered])[:,1]
        avg = np.average(values)
        prefix_accs.append([acc, avg])
    prefix_accs = np.array(prefix_accs)
    np.save(PREFIX_NP_PATH, prefix_accs)
    print("prefix accs saved")

print("Predict train data")

time_to_failure_predictions = np.array([], np.float32)
errors = np.array([], np.float32)

if os.path.exists(TIMEP_PATH):
    time_to_failure_predictions = np.load(TIMEP_PATH)

if os.path.exists(ERRORS_PATH):
    errors = np.load(ERRORS_PATH)

dx = 0
for data in tqdm(train_data):
    if dx >= len(errors):       
        acc = data[0]
        time = data[1]
        timeP = np.average((prefix_accs[np.where(prefix_accs[:,0] == acc)])[:,1])    
        np.append(time_to_failure_predictions, timeP)
        error = abs(timeP - time) / time
        np.append(errors, error);
    dx += 1
    if (dx % 1_000_000 == 0):
        np.save(TIMEP_PATH, time_to_failure_predictions)
        np.save(ERRORS_PATH, errors)

train_data_errors = pd.DataFrame(columns={'acoustic_data': np.int16, 'time_to_failure': np.float32, 'time_to_failure_predictions': np.float32, 'error': np.float32 })
train_data_errors['acoustic_data'] = train_data[:,0]
train_data_errors['time_to_failure'] = train_data[:,1]
train_data_errors['time_to_failure_predictions'] = time_to_failure_predictions
train_data_errors['error'] = errors

print(train_data_errors.describe())
print(train_data_errors.head())

train_data_errors.to_pickle(ERROR_REPORT_PATH)