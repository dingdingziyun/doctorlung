import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, ProgbarLogger
from keras.utils import np_utils
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import itertools
from sklearn.model_selection import train_test_split

def data_split(x_data, y_data):
    seed = 1000
    # split data into Train, Validation and Test
    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        train_size=0.8,
                                                        random_state=seed,
                                                        shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      train_size=0.8,
                                                      random_state=seed,
                                                      shuffle=True)

    
    return x_train, x_test, x_val, y_train, y_test, y_val

keras.backend.clear_session()
print('Build LSTM RNN model ...')
model = Sequential()
model.add(
    LSTM(units=64,
         dropout=0.05,
         recurrent_dropout=0.20,
         return_sequences=True,
         input_shape=(50, 245)))
# model.add(
#     LSTM(units=32,
#          dropout=0.05,
#          recurrent_dropout=0.20,
#          return_sequences=True))
model.add(
    LSTM(units=32,
         dropout=0.05,
         recurrent_dropout=0.20,
         return_sequences=False))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='Adamax',
              metrics=['acc', 'mse', 'mae', 'mape', 'cosine'])
model.summary()


%%time
# saved model checkpoint file
best_model_file = "./best_model_melspec_1lstmlayer32_weighted.hdf5"
#train_model_file=file_path+"/checkpoints/weights.best_{epoch:02d}-{loss:.2f}.hdf5"
MAX_PATIENT = 12
MAX_EPOCHS = 250
MAX_BATCH = 32

# callbacks
# removed EarlyStopping(patience=MAX_PATIENT)
callback = [
    ReduceLROnPlateau(patience=MAX_PATIENT, verbose=1),
    ModelCheckpoint(filepath=best_model_file,
                    monitor='loss',
                    verbose=1,
                    save_best_only=True)
]

print("training started..... please wait.")
# training
history = model.fit(x_train,
                    y_train,
                    
                    class_weight= class_weights,
                    batch_size=MAX_BATCH,
                    epochs=MAX_EPOCHS,
                    verbose=0,
                    validation_data=(x_val, y_val),
                    callbacks=callback)

print("training finised!")