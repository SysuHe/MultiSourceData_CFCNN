from keras.models import load_model
import gdal
from gdalconst import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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

if __name__ == "__main__":
    gdal.AllRegister()
    RSimagefilename = "../Data/High_Spatial_Resolution_Image/Guangzhou_Studyarea.tif"
    RSimageDataset = gdal.Open(RSimagefilename, GA_ReadOnly)
    RSimageArray = RSimageDataset.ReadAsArray()
    RSimagetransform = RSimageDataset.GetGeoTransform()
    rXSize = RSimageDataset.RasterXSize
    rYSize = RSimageDataset.RasterYSize

    wdSize = 128
    wdStep = 64
    nSampleSize = 350
    Point_Samfile = "../Data/TestSample.csv"
    PointArray = pd.read_csv(Point_Samfile)
    nImage = np.zeros((nSampleSize, wdSize, wdSize, 3))

    train = pd.read_csv('../Data/RTUD_Feature_Test.csv', header=None)
    scaler = StandardScaler().fit(train)
    scaler_train = scaler.fit_transform(train)
    x_train = np.expand_dims(scaler_train, -1)

    model_twostream = load_model("../Model/CFCNN.h5")
    model_twostream.summary()

    for i in range(len(PointArray)):
        xPT = PointArray["WGS_1984_UTM_49N_x"][i]
        yPT = PointArray["WGS_1984_UTM_49N_y"][i]
        xPixelinRS, yPixelinRS = World2Pixel(xPT, yPT, RSimagetransform)

        if (xPixelinRS - wdStep) < 0 and (yPixelinRS - wdStep) < 0:
            xPixelinRS = wdStep
            yPixelinRS = wdStep
        elif (xPixelinRS - wdStep) < 0 and (yPixelinRS - wdStep) > 0:
            xPixelinRS = wdStep
        elif (xPixelinRS - wdStep) > 0 and (yPixelinRS - wdStep) < 0:
            yPixelinRS = wdStep

        nDataarray = RSimageArray[:, (yPixelinRS - wdStep):(yPixelinRS + wdStep),
                     (xPixelinRS - wdStep):(xPixelinRS + wdStep)]
        nDataarray[nDataarray == 256] = 0

        nProarr = nDataarray[:3, :, :]
        permutation = np.argsort([3, 2, 1])
        nProarr = nProarr[permutation, :, :]
        image = lut_display(np.rollaxis(nProarr, 0, 3))
        nImage[i] = image / 255.0

    nClass_CNN = model_twostream.predict([nImage, x_train])
    nClass_CNN_data = pd.DataFrame(np.argmax(nClass_CNN, axis=1), dtype=float)
    nClass_CNN_data.to_csv("../ClassificationResult/CFCNN_Test.csv", sep=",", header=None, index=True, mode="w")

