'''
Based on the code used by the Camformer team for the DREAM Challenge 2022.
This script takes a one-hot-encoded file with 110 nt sequences and expression values.
It implements a CNN to predict expression.
'''

import pickle
import bz2
import math, re, os
import time
import warnings
warnings.simplefilter("ignore")
from optparse import OptionParser

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.metrics import RSquare as r_square
from tensorflow.python.client import device_lib
from tensorflow.keras.utils import Sequence
import tensorflow_addons as tfa

import joblib
from matplotlib import pyplot as plt

#Trying to fix "Python interpreter state not initiatlised" error 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Interrupt if no GPU is available
assert 'GPU' in str(device_lib.list_local_devices())

# Parse command line options
# There are referred to as options.inputFile, options.bestModelPath, options.saveFolder
parser = OptionParser()
parser.add_option('--inputFile', '--inputFile', default=None, type=str, dest='inputFile', help='Input filename')
parser.add_option('--bestModelPath', '--bestModelPath', default="tmp", type=str, dest='bestModelPath', help='Best model path')
parser.add_option('--saveFolder', '--saveFolder', default=None, type=str, dest='saveFolder', help='Name of the file with saved predictions')
(options, args) = parser.parse_args()

# Check whether the specified path exists; create otherwise
if not os.path.exists(options.bestModelPath):
  os.makedirs(options.bestModelPath)
if not os.path.exists(options.saveFolder):
  os.makedirs(options.saveFolder)

# Create timestamp to append in the filenames of the results
timestr = time.strftime('%Y%m%d-%H%M%S')

# Create a MirroredStrategy for multithreading and running on a GPU
tf.config.list_physical_devices('GPU')
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Read compressed one-hot-encoded data
with open(options.inputFile, 'rb') as f:
    X = pickle.load(f)
    y = pickle.load(f)

###################################
####  Define model parameters  ####
batch_size = 32
epochs = 300 # Max number of epochs
learn_rate = 1e-3
weight_decay = 1e-5 # Only used with AdamW optimiser
patience = 5 # Number of epochs without improvement before interrupting
threshold = 1e-3 # Difference in the evaluation metric to be considered an improvement
submission_mode = False # True or False; will determine if test set split is used or not; also changes size of validation set.
###################################

# Split the data in train, test, and validation sets.

if submission_mode==False:
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1,random_state=123)
else:
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.08,random_state=123) # Keep validation size the same: (1-0.2)*0.1=0.08
	X_test, y_test = X_val, y_val # "Ugly" hack to keep rest of the code intact

# Convert to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)
X_val, y_val = np.array(X_val), np.array(y_val)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_val.shape, y_val.shape)

# Shuffle a and b with the same order
def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

# The dataset may be too large for the memory, and then we use a "DataGenerator"
# that returns a "batch_size" number of objects at a time.
class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = unison_shuffled_copies(x_set, y_set)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

# Define datasets using the DataGenerator
train_dataset = DataGenerator(X_train, y_train, batch_size)
test_dataset = DataGenerator(X_test, y_test, batch_size)
val_dataset = DataGenerator(X_val, y_val, batch_size)

#########################################
####     Run on a GPU for speed      ####
with strategy.scope():
    # Paste parameters here
    out_channels=[32, 480, 416, 384, 32, 32]
    kernel_size=[6, 2, 8, 8, 6, 10]
    pool_size=[1, 2, 1, 1, 1, 1]
    strides=[2, 1, 2, 2, 2, 2]
    dropout=[0, 0, 0, 0, 0, 0.5]
    linear_layers= 1 #Change this to add linear layers 
    linear_size=80
    linear_dropouts=[0.5]

    # Define the network according to the parameters above
    model=keras.models.Sequential() # Create a model

    # Convolutional layers
    for i in range(len(out_channels)):
        if i==0: # Need to specify input shape for first layer
            model.add(keras.layers.Conv1D(out_channels[i], kernel_size=kernel_size[i], strides=1, activation='gelu', padding='same', input_shape=(80,4)))
        else:
            model.add(keras.layers.Conv1D(out_channels[i], kernel_size=kernel_size[i], strides=1, activation='gelu', padding='same'))
        #model.add(keras.layers.BatchNormalization())
        if pool_size[i]>1:
            model.add(keras.layers.MaxPooling1D(pool_size=pool_size[i], strides=strides[i], padding='same'))
        if dropout[i]>0:
            model.add(keras.layers.Dropout(dropout[i]))

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate=0.5))
    # Flatten to allow for dense layers
    model.add(keras.layers.Flatten())

    # Dense layers
    #for i in range(linear_layers):
        #model.add(keras.layers.Dense(linear_size, activation='linear'))
        #model.add(keras.layers.BatchNormalization())
        #if linear_dropouts[i] > 0:
            #model.add(keras.layers.Dropout(rate=linear_dropouts[i]))

    # Final output layer
    model.add(keras.layers.Dense(80, activation='softmax'))

    # Compile model
    model.compile(tf.keras.optimizers.Adam(learning_rate=learn_rate), loss='categorical_crossentropy', metrics=[tf.keras.metrics.KLDivergence()], steps_per_execution=1) 
    model.summary() # Print model summary

#########################################
# Run the remaining script on a CPU

# Define some callback functions that will be run at each epoch
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, min_delta=threshold, mode='min', restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(options.bestModelPath, monitor='val_kullback_leibler_divergence', verbose=False, save_best_only=True, mode='min')
callback_list = [early_stop, checkpoint]

# Train the model; this is the most time-consuming part
history=model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callback_list)

n_epochs = len(history.history['loss']) # Retrieve the total number of training epochs that was used

# Perform predictions on the test set
test_pred=model.predict(X_test)
print(test_pred)

joblib.dump(model, 'Trained_model_0305_PS.total.ox.rep1.joblib')

#test_res=pd.DataFrame(np.vstack([np.array_str(test_pred), np.array_str(y_test)]).T, columns=['y_pred', 'y_true'])
test_res=pd.DataFrame(np.vstack([np.array_str(test_pred.T), np.array_str(y_test)]).T, columns=['y_pred', 'y_true'])

test_pred2 = pd.DataFrame(np.array(test_pred))
y_test2 = pd.DataFrame(np.array(y_test))

test_res.to_csv(options.saveFolder+'/predictions_'+timestr+'.txt', sep='\t')
test_pred2.to_csv(options.saveFolder+'/predictions_only.'+timestr+'.txt', sep = '\t')
y_test2.to_csv(options.saveFolder+'/real_only.'+timestr+'.txt', sep = '\t')

y_true=test_res['y_true']
y_pred=test_res['y_pred']

plt.figure(figsize=(10,10))
#fig, axs = plt.subplots(2)
plt.plot(history.history['kullback_leibler_divergence'])
plt.plot(history.history['val_kullback_leibler_divergence'])
plt.title('model KLD')
plt.ylabel('KLD')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('val_loss'+timestr+'.png')


# Calculate and print performance  metrics
if submission_mode == False:
    print("Results on the test set: ")
else:
    print("Results on the validation set: ")
print("MAE:\t%s" % round(mean_absolute_error(y_true, y_pred), 4))
print("MSE:\t%s" % round(mean_squared_error(y_true, y_pred), 4))
print("r:\t%s" % round(stats.pearsonr(y_true, y_pred)[0], 4))
print("r2:\t%s" % round(stats.pearsonr(y_true, y_pred)[0]**2, 4))
print("rho:\t%s" % round(stats.spearmanr(y_true, y_pred)[0], 4))

# Save results and information
test_res.to_csv(options.saveFolder+'/predictions_'+timestr+'.txt', sep='\t')

pd.DataFrame(np.array([['MAE', str(round(mean_absolute_error(y_true, y_pred), 4))],
                       ['MSE', str(round(mean_squared_error(y_true, y_pred), 4))],
                       ['r', str(round(stats.pearsonr(y_true, y_pred)[0], 4))],
                       ['r2', str(round(stats.pearsonr(y_true, y_pred)[0]**2, 4))],
                       ['rho', str(round(stats.spearmanr(y_true, y_pred)[0], 4))]])).to_csv(options.saveFolder+'/summary_'+timestr+'.txt', sep='\t')

pd.DataFrame(np.array([['Data', str(options.inputFile)],
                       ['epochs', str(epochs)],
                       ['n_epochs', str(n_epochs-patience)],
                       ['batch_size', str(batch_size)],
                       ['learn_rate', str(learn_rate)],
                       ['weight_decay', str(weight_decay)],
                       ['min_delta/threshold', str(threshold)],
                       ['patience', str(patience)]])).to_csv(options.saveFolder+'/parameters_'+timestr+'.txt', sep='\t')
