# MultiSourceData_CFCNN

Keras implementation of land use classification by CNNs

## Code
+ Simpilified Residential Network: `./Code/Simplified_ResNet.py`
+ VGG-like Network: `./Code/AtrousVGG.py`
+ Two-stream convolutional neural network for combining features (CFCNN): `./Code/Keras_Merge.py`
+ Classify test samples by single data-based model: `./Code/Classify_TestSample_SingledataCNN.py`
+ Classify test samples by multi-source data-based model: `./Code/Classify_TestSample_CFCNN.py`
+ Classify the whole research area by trained models: `./Code/Classify_FullImage.py`

## Operating environment
The source code is compiled on the Windows 10 platform using Python 3.6. The dependencies include:
> tensorflow-gpu: 1.9, backend </br>
> Keras: 2.2.4, framework </br>
> pandas: used for csv I/O </br>
> numpy: used for array operations </br>
> matplotlib: used to visualize training accuracy curves </br>
> GDAL: used for remote sensing image I/O </br>
> scikit-learn: used for data preprocessing</br>

## Dataset
We provide training data and test data for estimating the performance of models.
Training data and test data can be found in `./Data`, stored as sample points.

The original high spatial resolution image and population density data can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1qIaLTqkQ7byazxO5vgolZw). The extracted code is `fo7v`.

## Example
We provide a sample of data in `./Data/Exam_ClassificationResult`. The folder contains processed high-resolution images and population density data. We can test the feasibility of the code in `./Code/Classify_FullImage.py`.
