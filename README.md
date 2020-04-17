# IDC_breast_cancer_detection
Detection of breast cancer from histopathology images through a specific purpose CNN built with Pytorch. 
Dataset available [here](https://www.kaggle.com/paultimothymooney/breast-histopathology-images/)

Image samples for positive and negative cases:

![Image Sample positive](https://github.com/Sujit27/IDC_breast_cancer_detection/blob/master/8863_idx5_x1001_y801_class1.png)
![Image Sample negative](https://github.com/Sujit27/IDC_breast_cancer_detection/blob/master/8863_idx5_x201_y1351_class0.png)

## Getting Started
Download the dataset and save it in a new directory named `IDC_breast_cancer_detection/dataset`

Run the create annotation.py file to create annotation.txt, which is a list of all files in the dataset

`python3 create_annotation.py`

Create a new directory `IDC_breast_cancer_detection/saved_model` for storing trained model later

### Prerequisites

1.PIL

2.sklearn

3.Pytorch

### Installing
`pip install Pillow`

`pip install numpy scipy scikit-learn`

`pip install torch torchvision` [see here](https://pytorch.org/)

### Training
Check the default parameters for training and run  

`python train.py -h`

trained model will be saved as **breast_cancer_detector.pth** in `IDC_breast_cancer_detection/saved_model`
