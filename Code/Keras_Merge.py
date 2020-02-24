from keras.layers import Concatenate, Add, Input
from keras.layers.convolutional import Conv1D, Conv2D, ZeroPadding2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gdal
from gdalconst import *
import time

def World2Pixel(Xworld, Yworld, geotransform):
    dTemp = geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4]
    Xpixel = int((geotransform[5] * (Xworld - geotransform[0]) - geotransform[2] * (Yworld - geotransform[3]))
                 / dTemp + 0.5)
    Ypixel = int((geotransform[1] * (Yworld - geotransform[3]) - geotransform[4] * (Xworld - geotransform[0]))
                 / dTemp + 0.5)
    return Xpixel, Ypixel

def Pixel2World(Xpixel, Ypixel, geotransform):
    Xworld = geotransform[0] + Xpixel * geotransform[1] + Ypixel * geotransform[2]
    Yworld = geotransform[3] + Xpixel * geotransform[4] + Ypixel * geotransform[5]
    return Xworld, Yworld

def display(image, display_min, display_max):
    image.clip(display_min, display_max, out=image)
    image -= display_min
    image = image // ((display_max - display_min + 1) / 256.)
    return image.astype(np.uint8)

def lut_display(image):
    lut = np.arange(2**16, dtype='uint16')
    display_min = image.min()
    display_max = image.max()
    lut = display(lut, display_min, display_max)
    return np.take(lut, image)

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

def residual_block(inputs, filters, kernelsize, stride=1, stage=1, block="", skip_connection_type="sum"):
    conv_name_base = "res" + str(stage) + str(block) + "_branch"
    bn_name_base = "bn" + str(stage) + str(block) + "_branch"

    x = inputs
    x_shortcut = inputs
    x = Conv2D(filters[0], (1, 1), strides=(stride, stride), padding="valid", name=conv_name_base + "2a",
               kernel_initializer=glorot_uniform(0))(x)
    x = BatchNormalization(name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(filters[1], (kernelsize, kernelsize), strides=(1, 1), padding="same", name=conv_name_base + "2b",
               kernel_initializer=glorot_uniform(0))(x)
    x = BatchNormalization(name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(filters[2], (1, 1), strides=(1, 1), padding="valid", name=conv_name_base + "2c",
               kernel_initializer=glorot_uniform(0))(x)
    x = BatchNormalization(name=bn_name_base + "2c")(x)

    if skip_connection_type == "conv":
        x_shortcut = Conv2D(filters[2], (1, 1), strides=(stride, stride), name=conv_name_base + "1",
                            padding="valid", kernel_initializer=glorot_uniform(0))(x_shortcut)
        x_shortcut = BatchNormalization(name=bn_name_base + "1")(x_shortcut)
        x = Add()([x_shortcut, x])
    elif skip_connection_type == "sum":
        x = Add()([x_shortcut, x])
    else:
        x = x
    x = Activation("relu")(x)

    return x

def Simplified_ResNet(input_shape, n_class):
    input_x = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(input_x)
    x = Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(x)
    x = BatchNormalization(name="bn_conv1")(x)
    x = Activation("relu")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = residual_block(x, filters=[64, 64, 256], kernelsize=3, stage=1, block="ar1", stride=1, skip_connection_type="conv")
    x = residual_block(x, filters=[64, 64, 256], kernelsize=3, stage=1, block="as_1", stride=1, skip_connection_type="sum")
    x = residual_block(x, filters=[64, 64, 256], kernelsize=3, stage=1, block="as_2", stride=1, skip_connection_type="sum")
    x = Dropout(0.5)(x)

    x = residual_block(x, filters=[256, 256, 1024], kernelsize=3, stage=1, block="br1", stride=2, skip_connection_type="conv")
    x = residual_block(x, filters=[256, 256, 1024], kernelsize=3, stage=1, block="bs_1", stride=1, skip_connection_type="sum")
    x = residual_block(x, filters=[256, 256, 1024], kernelsize=3, stage=1, block="bs_2", stride=1, skip_connection_type="sum")
    x = Dropout(0.5)(x)

    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_class)(x)
    x = Activation("softmax")(x)

    return Model(input_x, x)

def trainModel(XImg, YImg, XBdata, YBdata, nb_class, val_size, epochs, batch_size=64, imagesize=256, lr=0.001, model_path="multilabel_vgg.h5", fig_path="loss_acc.png"):
    # split training and validation samples
    train_XImg, val_XImg, train_YImg, val_YImg = train_test_split(XImg, YImg, test_size=val_size, random_state=2018)
    train_XBdata, val_XBdata, train_YBdata, val_YBdata = train_test_split(XBdata, YBdata, test_size=val_size, random_state=2018)

    # CF-CNN model
    checkpoint = ModelCheckpoint(model_path, monitor="val_acc", mode="max", save_best_only=True, verbose=1)
    reducelr = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=5, factor=0.5, min_lr=1e-4, verbose=1)
    model1 = Simplified_ResNet(input_shape=(imagesize, imagesize, 3), n_class=nb_class)
    model1.summary()
    model2 = AtronsVGG(input_shape=train_XBdata.shape[1:], n_class=nb_class)
    model2.summary()
    output1 = model1.get_layer("global_average_pooling2d_1").output
    output2 = model2.get_layer("batch_normalization_7").output
    t = Concatenate()([output1, output2])
    t = Dropout(0.5)(t)
    top_model = Dense(7, activation="softmax")(t)

    # fitting model
    model = Model(inputs=[model1.input, model2.input], outputs=top_model)
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(lr=lr), metrics=["accuracy"])
    time_start = time.time()
    H = model.fit([train_XImg, train_XBdata], train_YImg, batch_size=batch_size, epochs=epochs,
                  validation_data=([val_XImg, val_XBdata], val_YImg), verbose=2, callbacks=[checkpoint, reducelr])
    time_end =time.time()
    print(str(time_end - time_start) + "s")

    # save model
    model.save(model_path)
    del model

    # save acc and loss
    loss_df = pd.DataFrame(H.history["loss"], dtype=float).T
    loss_df = loss_df.append(pd.DataFrame(H.history["val_loss"], dtype=float).T)
    loss_df = loss_df.append(pd.DataFrame(H.history["acc"], dtype=float).T)
    loss_df = loss_df.append(pd.DataFrame(H.history["val_acc"], dtype=float).T)
    loss_df.to_csv("../AccandLoss/accandloss_CFCNN.csv", header=False, index=False, mode="w")

    # plot training figures
    plt.style.use("ggplot")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(H.epoch, H.history["loss"], label="train_loss")
    plt.plot(H.epoch, H.history["val_loss"], label="val_loss")
    print(H.history["val_loss"])
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

if __name__ == '__main__':
    Point_Samfile = "../Data/TrainingSample.csv"
    PointArray = pd.read_csv(Point_Samfile)

    gdal.AllRegister()
    RSimagefilename = "../Data/High_Spatial_Resolution_Image/Guangzhou_Studyarea.tif"
    RSimageDataset = gdal.Open(RSimagefilename, GA_ReadOnly)
    RSimageArray = RSimageDataset.ReadAsArray()
    RSimagetransform = RSimageDataset.GetGeoTransform()

    train = pd.read_csv('../Data/RTUD_feature_Training.csv', header=None)
    scaler = StandardScaler().fit(train)
    scaler_train = scaler.fit_transform(train)
    x_train = np.expand_dims(scaler_train, -1)

    wdSize = 128
    wdStep = 64
    nSampleSize = 2100

    x_Imgdata = np.zeros((nSampleSize, wdSize, wdSize, 3))
    nLabel = [0 for n in range(nSampleSize)]

    for i in range(len(PointArray)):
        xPT = PointArray["WGS_1984_UTM_49N_x"][i]
        yPT = PointArray["WGS_1984_UTM_49N_y"][i]
        nLabelnum = int(PointArray["Class"][i])
        xPixelinRS, yPixelinRS = World2Pixel(xPT, yPT, RSimagetransform)

        if xPixelinRS < wdStep and yPixelinRS < wdStep:
            xPixelinRS = wdStep
            yPixelinRS = wdStep
        elif xPixelinRS < wdStep and yPixelinRS > wdStep:
            xPixelinRS = wdStep
        elif xPixelinRS > wdStep and yPixelinRS < wdStep:
            yPixelinRS = wdStep

        nDataarray = RSimageArray[:, (yPixelinRS - wdStep):(yPixelinRS + wdStep), (xPixelinRS - wdStep):(xPixelinRS + wdStep)]
        nProarr = nDataarray[:3, :, :]
        permutation = np.argsort([3, 2, 1])
        nProarr = nProarr[permutation, :, :]
        image = lut_display(np.rollaxis(nProarr, 0, 3))
        x_Imgdata[i] = image / 255.0
        nLabel[i] = nLabelnum

    trainModel(x_Imgdata, nLabel, x_train, nLabel, nb_class=7, val_size=0.2, epochs=100, batch_size=16, imagesize=wdSize, lr=0.005,
               model_path="../Model/CFCNN.h5", fig_path="../AccandLoss/loss_acc_CFCNN.png")
