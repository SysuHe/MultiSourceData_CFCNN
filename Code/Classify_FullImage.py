import pandas as pd
import numpy as np
import gdal
from gdalconst import *
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

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

if __name__ == '__main__':
    gdal.AllRegister()
    train = pd.read_csv('../Data/Exam_ClassificationResult/RTUD_Feature_Exam.csv', header=None)
    scaler = StandardScaler().fit(train)
    scaler_train = scaler.fit_transform(train)
    x_train = np.expand_dims(scaler_train, -1)

    gdal.AllRegister()
    RSimagefilename = "../Data/Exam_ClassificationResult/Exam_HSR.tif"
    RSimageDataset = gdal.Open(RSimagefilename, GA_ReadOnly)
    RSimageArray = RSimageDataset.ReadAsArray()
    RSimagetransform = RSimageDataset.GetGeoTransform()
    RSimageBands = RSimageDataset.RasterCount
    RSimageProj = RSimageDataset.GetProjection()
    nXSize = RSimageDataset.RasterXSize
    nYSize = RSimageDataset.RasterYSize

    wdSize = 128
    wdStep = 64
    nCount = 0
    nImage = np.zeros((1, 128, 128, 3))

    model_twoCNN = load_model("../Model/CFCNN.h5")
    model_twoCNN.summary()

    for i in range(wdStep, nXSize, wdStep):
        if (i + wdStep) > nXSize:
            i = nXSize - wdStep
        for j in range(wdStep, nYSize, wdStep):
            if (j + wdStep) > nYSize:
                j = nYSize - wdStep

            nDataarray = RSimageArray[:, (j - wdStep):(j + wdStep), (i - wdStep):(i + wdStep)]
            if np.sum(nDataarray == 256) > int(nDataarray.size / 3):
                continue
            else:
                nDataarray[nDataarray == 256] = 0

            permutation = np.argsort([3, 2, 1])
            nProarr = nDataarray[permutation, :, :]
            image = lut_display(np.rollaxis(nProarr, 0, 3))
            nImage[0] = image / 255.0
            x_train_pro = np.expand_dims(x_train[nCount], 0)
            nCount += 1

            nClass_CNN = model_twoCNN.predict([nImage, x_train_pro])
            nClass_CNN_data = pd.DataFrame(np.argmax(nClass_CNN, axis=1), dtype=int)
            nClass_CNN_data.to_csv("../ClassificationResult/CFCNN_Exam.csv", sep=",", header=False, index=True, mode="a")