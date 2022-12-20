### AP10K
Please create a conda environment with the dependencies in the requirements.txt.

To get key points for our cow frames of interest, we use the MMPose framework with models specfically trained on the AP10K data set. The github repo for the data set is here: https://github.com/AlexTheBad/AP-10K

Please go to the link and clone the repository. In the README, there's a section with Dataset Preparation. You don't need to download the actual data, but please make the folders that mimic the hierarchy. Also, download the pre-trained HRNet-w48 and put it in the AP10k folder.

![image](https://user-images.githubusercontent.com/77544183/208780267-84166ead-739d-45cc-95dd-ecbee8f9f672.png)

In the AP-10k/data/ap10k/data folder, put all the superimposed frames of interest. Then, we need to create our own ap10k-test-split1.json file in the annotatons folder. 

To do this, use the get_annotation_file.ipynb file. In the second cell, replace 'coordinates.csv' with the coordinates file of your coosing. It should have the filename, left, top, width, and height. Running this file will create annotations/ap10k-test-split1.json. 

### Getting Predictions

After having an annotation file and the frames in the correct locations, you can run inference after by running this command: 

```
python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/hrnet_w48_ap10k_256x256.py hrnet_w48_ap10k_256x256-d95ab412_20211029.pth
```



