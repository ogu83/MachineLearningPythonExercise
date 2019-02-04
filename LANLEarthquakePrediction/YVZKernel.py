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


submission = pd.read_csv(SUBMISSON_PATH, index_col='seg_id')
X_test = pd.DataFrame(columns={"acoustic_data", "time_to_failure"}, dtype=np.float64, index=submission.index)
test_predictions = []

for i, seg_id in enumerate(tqdm(X_test.index)):
    seg = pd.read_csv(TEST_DATA_PATH + '\\' + seg_id + '.csv')
    seg_acustic_data = seg['acoustic_data']
    
    seg_time_sum = float()
    seg_time_count = int()
    seg_time_avg = float()

    for seg_acc in tqdm(seg_acustic_data):
        seg_time_sum += np.average((prefix_accs[np.where(prefix_accs[:,0] == seg_acc)])[:,1])
        seg_time_count += 1
        seg_time_avg = seg_time_sum / seg_time_count

    test_predictions.append(seg_time_avg)
    print(seg_id, seg_time_avg)   
    
#MAKE SUBMISSION FILE
submission = pd.read_csv(SUBMISSON_PATH, index_col='seg_id')
submission['time_to_failure'] = test_predictions
print(submission.head())
submission.to_csv(f'{DATA_PATH}\\submission.csv') 