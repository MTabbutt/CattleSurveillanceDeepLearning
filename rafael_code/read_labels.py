# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 17:00:33 2022

@author: repfe
"""

import os
import json
import numpy as np

root_dir = r'D:\UW\OneDrive - UW-Madison\UW\Courses\Fall 2022\CS 762\Project\d20_files'
labels_dir = os.path.join(root_dir, 'labels')
videos_dir = os.path.join(root_dir, 'video data')
for video_file in os.listdir(videos_dir):
    label_file = video_file[:-4] + '.json'
    if not os.path.isfile(os.path.join(labels_dir, label_file)):
        print(f"{label_file} does not exist!")
        continue
    with open(os.path.join(labels_dir, label_file)) as f:
        labels = np.array(json.load(f))
    percentage_night = np.mean(labels == 8)
    print(video_file, percentage_night)
    if percentage_night < 0.8:
        os.rename(os.path.join(videos_dir, video_file), os.path.join(videos_dir, 'good', video_file))
    else:
        os.rename(os.path.join(videos_dir, video_file), os.path.join(videos_dir, 'bad', video_file))