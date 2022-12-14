### AP10K
Please create a conda environment with the dependencies in the requirements.txt.

To get key points for our cow frames of interest, we use the MMPose framework with models specfically trained on the AP10K data set. The github repo for the data set is here: https://github.com/AlexTheBad/AP-10K

Please go to the link and clone the repository. In the README, there's a section with Dataset Preparation. You don't need to download the actual data, but please make the folders that mimic the hierarchy. Also, download the pre-trained HRNet-w48 and put it in the AP-10k (outer) folder.

![image](https://user-images.githubusercontent.com/77544183/208780267-84166ead-739d-45cc-95dd-ecbee8f9f672.png)

In the AP-10k/data/ap10k/data folder, put all the superimposed frames of interest. To access the data that we use:

Unlabeled set in /unlabeled/ : https://drive.google.com/drive/folders/11ZcCVpo9gjVISKYDPMz66ET5IKk5Y7cU?usp=sharing 

We also want to download the coordinates.csv in that folder.

Then, we need to create our own ap10k-test-split1.json file in the annotatons folder. 

To do this, first clone this repository and then use the get_annotation_file.ipynb file. In the second cell, replace 'coordinates.csv' with the coordinates file of your coosing. It should have the filename, left, top, width, and height. Running this file will create annotations/ap10k-test-split1.json. 

### Getting Predictions

After having an annotation file and the frames in the correct locations, you can run inference after by running this command: 

```
python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/hrnet_w48_ap10k_256x256.py 
hrnet_w48_ap10k_256x256-d95ab412_20211029.pth
```

After getting predictions in 'preds.json', open get_prediction_csv.ipynb and run through the notebook. In the coords_full.to_csv command, that is where you output the csv with the coordinates csv and the key point prediction csvs merged into one. This directly feeds into the weak supervision framework.

In Visualization, you can replace p with an image of your choice to visualize the keypoints predicted on the cow.

