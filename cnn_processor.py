import os
import pickle

import numpy as np
from keras.callbacks import EarlyStopping
from keras.engine.saving import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from analyst.viewhandler import view_cnn_train_hist, view_common_predict


# construct model
def cnn_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(13, 32, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1, init="normal"))
    model.compile(loss='mean_squared_error', optimizer=Adam())
    return model


def cnn_train(preprocpath, cnnworkpath, train_range):
    cnnmodelpath = cnnworkpath + '\\model\\'
    cnnhistpath = cnnworkpath + '\\hist\\'
    files = []
    wfiles = os.listdir(preprocpath)
    for file in wfiles:
        files.append(os.path.join(preprocpath, file))
    feature = None
    label = None
    for i in train_range:
        print('train:',files[i])
        data = np.load(files[i])
        eachfeature = data["feature"]
        eachlabel = data["label"]
        if i == train_range[0]:
            feature = eachfeature
            label = eachlabel
        else:
            feature = np.append(feature, eachfeature, axis=0)
            label = np.append(label, eachlabel, axis=0)
    # training
    feature = feature.reshape(feature.shape[0], 13, 32, 1)
    for j in range(1, 4):
        # 1,north
        # 2,east
        # 3,up
        modelname = str(train_range[0]) + '_' + str(train_range[-1]) + '_' + str(j)
        if os.path.exists(cnnmodelpath + modelname + ".h5"):
            print('skip:'+cnnmodelpath + modelname + ".h5")
            continue
        model = cnn_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
        history = model.fit(feature, label[:, j]*1000, epochs=300, batch_size=32, validation_split=0.125, verbose=2,
                            shuffle=True, callbacks=[early_stopping])
        # Save trained model and history
        with open(cnnhistpath + modelname + ".pkl", 'wb') as hist_file:
            pickle.dump(history.history, hist_file)
        model.save(cnnmodelpath + modelname + ".h5")
        view_cnn_train_hist(cnnworkpath, modelname)


def cnn_predict(preprocpath, cnnworkpath, train_range, test_range, show):
    cnnmodelpath = cnnworkpath + '\\model\\'
    cnnoutpath = cnnworkpath + '\\out\\'
    # load model
    files = []
    wfiles = os.listdir(preprocpath)
    for file in wfiles:
        files.append(os.path.join(preprocpath, file))
    for i in test_range:
        data = np.load(files[i])
        test_feature = data["feature"]
        test_feature = test_feature.reshape(test_feature.shape[0], 13, 32, 1)
        for j in range(1, 4):
            # predict
            modelname = str(train_range[0]) + '_' + str(train_range[-1]) + '_' + str(j)
            predictname = str(train_range[0]) + '_' + str(train_range[-1]) + '_' + str(i) + '_' + str(j)
            model = load_model(cnnmodelpath + modelname + ".h5")
            predict = model.predict(test_feature)           
