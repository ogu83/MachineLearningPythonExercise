import numpy as np
np.set_printoptions(precision=2)
import pandas as pd
import os

import tensorflow.python.estimator
import tensorflow.python.estimator.canned.dnn

from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.metrics import mean_absolute_error

# Import Stuff
import tensorflow.contrib.learn as skflow
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)
logging.info('Tensorflow %s' % tf.__version__) # 1.4.1

from tqdm import tqdm

# This is the magic function which the Deep Neural Network
# has to 'learn' (see http://neuralnetworksanddeeplearning.com/chap4.html)
#f = lambda x: 0.2+0.4*x**2+0.3*x*np.sin(15*x)+0.05*np.cos(50*x)

DATA_PATH = "D:\\LANLEarthquakeData"
TRAIN_DATA_PATH = f"{DATA_PATH}\\train.csv"
TEST_DATA_PATH = f"{DATA_PATH}\\test"
SUBMISSON_PATH = f"{DATA_PATH}\\sample_submission.csv"
TRAINING_DERIVED_ROW_COUNT = 150_000
READ_WHOLE_TRAIN_DATA = True
READ_WHOLE_TEST_DATA = True
NP_DATA_PATH = f"{DATA_PATH}\\np"
PICKLE_PATH = f"{DATA_PATH}\\pickle"

Y_TRAIN_PICKLE = f"{PICKLE_PATH}\\y_train.pickle"
X_TRAIN_PICKLE = f"{PICKLE_PATH}\\x_train.pickle"
X_TEST_PICKLE = f"{PICKLE_PATH}\\x_test.pickle"

X_train_scaled = pd.read_pickle(X_TRAIN_PICKLE)
y_tr = pd.read_pickle(Y_TRAIN_PICKLE)
print("Train Data Loaded")
print(X_train_scaled.head())
print(y_tr.head())
print("")

X_test_scaled = pd.read_pickle(X_TEST_PICKLE)
print("X Test Data Loaded")
print(X_test_scaled.head())
print("")

X_train, X_test, y_train, y_test = model_selection.train_test_split(np.array(X_train_scaled), np.array(y_tr)[:,0], test_size = 0.2)

# Defining the Tensorflow input functions
# for training
def training_input_fn(batch_size=1):
	return tf.estimator.inputs.numpy_input_fn(
					x={'X': X_train.astype(np.float32)},
					y=y_train.astype(np.float32),
					batch_size=batch_size,
					num_epochs=None,
					shuffle=True)
# for test
def test_input_fn():
	return tf.estimator.inputs.numpy_input_fn(
				  	x={'X': X_test.astype(np.float32)},
				  	y=y_test.astype(np.float32),
				  	num_epochs=1,
				  	shuffle=False)

# Network Design
# --------------
feature_columns = [tf.feature_column.numeric_column('X', shape=(len(X_train_scaled.columns),))]

TRAINING = False
WITHPLOT = False
STEPS_PER_EPOCH = 1000
EPOCHS = 30
BATCH_SIZE = 200

hidden_layers = [1024, 512, 256]
dropout = 0.17

MODEL_PATH='./DNNRegressors/'
for hl in hidden_layers:
	MODEL_PATH += '%s_' % hl
MODEL_PATH += 'D0%s' % (int(dropout*100))
logging.info('Saving to %s' % MODEL_PATH)

# Validation and Test Configuration
#validation_metrics = {"MSE": tf.contrib.metrics.streaming_mean_squared_error}
validation_metrics = {"MAE": tf.contrib.metrics.streaming_mean_absolute_error}
test_config = skflow.RunConfig(save_checkpoints_steps=100, save_checkpoints_secs=None)

# Building the Network
#regressor = skflow.DNNRegressor(feature_columns=feature_columns,
#				label_dimension=1,
#				hidden_units=hidden_layers,
#				model_dir=MODEL_PATH,
#				dropout=dropout,
#				config=test_config,
#                optimizer=tf.train.ProximalAdagradOptimizer(
#                   learning_rate=0.1,
#                   l1_regularization_strength=0.001)
#                )

regressor = skflow.DNNRegressor(feature_columns=feature_columns,
				label_dimension=1,
				hidden_units=hidden_layers,
				model_dir=MODEL_PATH,
				dropout=dropout,
				config=test_config,
                optimizer=tf.train.AdamOptimizer(learning_rate=0.0005)
                #optimizer=tf.train.AdadeltaOptimizer()
                #optimizer=tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
            )

#regressor = sskflow. BoostedTreesRegressor(
#    feature_columns=feature_columns,
#    n_batches_per_layer=100,
#    n_trees=100,    
#)

#regressor = skflow.LinearRegressor(feature_columns = feature_columns, model_dir=MODEL_PATH)

# Train it
if TRAINING:
	logging.info('Train the DNN Regressor...\n')
	MSEs = []	# for plotting
	STEPS = []	# for plotting

	for epoch in tqdm(range(EPOCHS+1)):
		# Fit the DNNRegressor (This is where the magic happens!!!)
		regressor.fit(input_fn=training_input_fn(batch_size=BATCH_SIZE), steps=STEPS_PER_EPOCH)
		# Thats it -----------------------------
		# Start Tensorboard in Terminal:
		# 	tensorboard --logdir='./DNNRegressors/'
		# Now open Browser and visit localhost:6006\

		
		# This is just for fun and educational purpose:
		# Evaluate the DNNRegressor every 10th epoch
		if epoch%(EPOCHS//100+1)==0:
			eval_dict = regressor.evaluate(input_fn=test_input_fn(), metrics=validation_metrics)				
			print('Epoch %i: %.5f MAE' % (epoch+1, eval_dict['MAE']))

			if WITHPLOT:
				# Generate a plot for this epoch to see the Network learning
				y_pred = regressor.predict(x={'X': X}, as_iterable=False)

				E = (y.reshape((1,-1))-y_pred)
				MSE = np.mean(E**2.0)
				step = (epoch+1) * STEPS_PER_EPOCH
				title_string = '%s DNNRegressor after %06d steps (MSE=%.5f)' % \
								(MODEL_PATH.split('/')[-1], step, MSE)
				
				MSEs.append(MSE)
				STEPS.append(step)

				fig = plt.figure(figsize=(9,4))
				ax1 = fig.add_subplot(1, 4, (1, 3))
				ax1.plot(X, y, label='function to predict')
				ax1.plot(X, y_pred, label='DNNRegressor prediction')
				ax1.legend(loc=2)
				ax1.set_title(title_string)
				ax1.set_ylim([0, 1])

				ax2 = fig.add_subplot(1, 4, 4)
				ax2.plot(STEPS, MSEs)
				ax2.set_xlabel('Step')
				ax2.set_xlim([0, EPOCHS*STEPS_PER_EPOCH])
				ax2.set_ylabel('Mean Square Error')
				ax2.set_ylim([0, 0.01])

				plt.tight_layout()
				plt.savefig(MODEL_PATH + '_%05d.png' % (epoch+1), dpi=72)
				logging.info('Saved %s' % MODEL_PATH + '_%05d.png' % (epoch+1))

				plt.close()

	# Now it's trained. We can try to predict some values.
else:
    logging.info('Starting Prediction')
    #try:
	# Prediction
    X_pred = np.array(X_test_scaled)
    y_pred = regressor.predict(x={'X': X_pred}, as_iterable=False)            

	# Get trained values out of the Network
    for variable_name in regressor.get_variable_names():
        if str(variable_name).startswith('dnn/hiddenlayer') and (str(variable_name).endswith('weights') or str(variable_name).endswith('biases')):
            print('\n%s:' % variable_name)
            weights = regressor.get_variable_value(variable_name)
            print(weights)
            print('size: %i' % weights.size)
    
    print("")
    print("Predictions")
    print(y_pred)

    submission = pd.read_csv(SUBMISSON_PATH, index_col='seg_id')
    submission['time_to_failure'] = y_pred
    print(submission.head())
    submission.to_csv(f'{DATA_PATH}\\submission.csv')

	# Final Plot
    if WITHPLOT:
        plt.plot(X, y, label='function to predict')
        plt.plot(X, regressor.predict(x={'X': X}, as_iterable=False), label='DNNRegressor prediction')
        plt.legend(loc=2)
        plt.ylim([0, 1])
        plt.title('%s DNNRegressor' % MODEL_PATH.split('/')[-1])
        plt.tight_layout()
        plt.savefig(MODEL_PATH + '.png', dpi=72)
        plt.close()
    #except:
    #    logging.Error('Prediction failed! Train a model')