# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:19:42 2022

@author: repfe
"""

import os
import csv
import cv2
import numpy as np

os.chdir(r'D:\UW\OneDrive - UW-Madison\UW\Courses\Fall 2022\CS 762\Project')
csv_file = 'valid_labeled_frames_random_v1.csv'
output_dir = 'valid_random_frames_v1'
frames_list = {}
with open(csv_file) as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
        split = row[0]
        video = row[1]
        frame_id = int(row[2])
        label_id = int(row[3])
        if video not in frames_list:
            frames_list[video] = (split, [])
        frames_list[video][1].append([frame_id, label_id])
for key, value in frames_list.items():
    frames_list[key] = (value[0], np.stack(value[1]))

for key, value in frames_list.items():
     cap = cv2.VideoCapture(os.path.join('d20_files', 'video data', 'good', key + '.mp4'))
     ret, frame = cap.read()
     frame_id = 0
     while ret:
         frame_ids_list = list(value[1][:,0])
         if frame_id in frame_ids_list:
             label_id = value[1][frame_ids_list.index(frame_id), 1]
             frame_name = f'{key}_{frame_id:05d}_{label_id}.jpg'
             frame_file = os.path.join('data', output_dir, value[0], frame_name)
             cv2.imwrite(frame_file, frame)
         ret, frame = cap.read()
         frame_id += 1
     cap.release()

