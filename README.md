# CattleSurveillanceDeepLearning


##### Original Paper data can be found and downloaded here along with thier codes: https://zenodo.org/record/3981400#.Y1mceYLMKDU


## Dataset curation and preperation: 

## 


## Baseline CNN:
The baseline CNN codes can be found in the BaselineCNN directory. These codes will do some preprocessing, and running of the baseline CNN. The code for testing the weak supervision model and creating the confusion matricies can also be found here. 

## Weak Supervision:

Found in the weak supervision directory. This code, given all the coordinates of bounding boxes for training, test, and unlabeled data, reproduce the results as shown in paper. 

## MMpose : 

This folder contains the necessary jupyter notebooks to generate the mmpose coordinates for each cow. 

## data_preprocessing:

Contains code for preprocessing the data, and curating and generating training, validation, test, and unlabeled sets (both superimposed and cropped)

