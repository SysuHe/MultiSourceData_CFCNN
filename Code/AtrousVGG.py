from keras.layers.convolutional import Conv1D
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def AtronsVGG(input_shape, n_class):
    input_x = Input(shape=input_shape)
    x = Conv1D(32, 3, padding="valid")(input_x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.25)(x)

    x = Conv1D(64, 3, padding="valid")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(64, 3, padding="valid", dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Conv1D(128, 3, padding="valid")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(128, 3, padding="valid", dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(n_class)(x)
    x = Activation("softmax")(x)

    return Model(input_x, x)

def trainModel(X, y, nb_class, val_size, epochs, batch_size=64, lr=0.001, model_path="multilabel_vgg.h5", fig_path="loss_acc.png"):

    # split training and validation samples
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=val_size, random_state=2018)
    print(X.shape)
    print(X.shape[1:])

    # fitting model
    model = AtronsVGG(input_shape=train_x.shape[1:], n_class=nb_class)
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(lr=lr, decay=lr/epochs), metrics=["accuracy"])
    checkpoint = ModelCheckpoint(model_path, monitor="val_acc", mode="max", save_best_only=True, verbose=1)
    reducelr = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=5, factor=0.5, min_lr=1e-4, verbose=1)
    time_start = time.time()
    H = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y), verbose=2, callbacks=[checkpoint, reducelr])
    time_end = time.time()
    print(str(time_end - time_start) + "s")

    # save model
    model.save(model_path)
    del model

    # save acc and loss
    loss_df = pd.DataFrame(H.history["loss"], dtype=float).T
    loss_df = loss_df.append(pd.DataFrame(H.history["val_loss"], dtype=float).T)
    loss_df = loss_df.append(pd.DataFrame(H.history["acc"], dtype=float).T)
    loss_df = loss_df.append(pd.DataFrame(H.history["val_acc"], dtype=float).T)
    loss_df.to_csv("../AccandLoss/accandloss_Atrous.csv", header=False, index=False, mode="w")

    # plot training figures
    plt.style.use("ggplot")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(H.epoch, H.history["loss"], label="train_loss")
    plt.plot(H.epoch, H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.subplot(1, 2, 2)
    plt.plot(H.epoch, H.history["acc"], label="train_acc")
    plt.plot(H.epoch, H.history["val_acc"], label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.savefig(fig_path)
    plt.show()

if __name__ == "__main__":
    train = pd.read_csv('../Data/RTUD_Feature_Training.csv', header=None)
    scaler = StandardScaler().fit(train)
    scaler_train = scaler.fit_transform(train)
    x_train = np.expand_dims(scaler_train, -1)
    label = pd.read_csv('../Data/Class_TrainingSample.csv', header=None)
    trainModel(x_train, label, nb_class=7, val_size=0.2, epochs=100, batch_size=16, lr=0.0005, model_path="../Model/AtrousVGG.h5", fig_path="../AccandLoss/loss_acc_Atrous.png")
