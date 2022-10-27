# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 03:04:53 2022

@author: repfe
"""

import os
import cv2

def contains_point(rectangle, point):
    if point[0] < rectangle[0]:
        return False
    if point[1] < rectangle[1]:
        return False
    if point[0] > rectangle[0] + rectangle[2]:
        return False
    if point[1] > rectangle[1] + rectangle[3]:
        return False
    return True

os.chdir(r'D:\UW\OneDrive - UW-Madison\UW\Courses\Fall 2022\CS 762\Project')
videos_dir = os.path.join('d20_files', 'video data', 'good')
bboxes_dir = os.path.join('results')
change_id = 1572166740
roi_1 = [(1087, 62, 564, 822),
         (378, 99, 1273, 785)]
roi_2 = [(1089, 62, 478, 815),
         (387, 155, 1180, 746)]
video = '1570557540.mp4'
starting_frame = 113

video_key = video[:-4]
bboxes_file = f'pred_{video_key}.txt'
bboxes_dict = {}
with open(os.path.join(bboxes_dir, bboxes_file)) as f:
    lines = f.readlines()
for line in lines:
    splits = line.split(' ')
    frame_id = int(splits[0])
    left = int(splits[2])
    top = int(splits[3])
    width = int(splits[4])
    height = int(splits[5])
    if frame_id not in bboxes_dict:
        bboxes_dict[frame_id] = []
    bboxes_dict[frame_id].append((left, top, width, height))
roi = roi_1 if int(video_key) < change_id else roi_2
cap = cv2.VideoCapture(os.path.join(videos_dir, video))
ret, frame = cap.read()
frame_id = 0
while ret:
    if frame_id >= starting_frame:
        bboxes = bboxes_dict.get(frame_id, [])
        for bbox in bboxes:
            center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
            is_contained = False
            for rect in roi:
                if contains_point(rect, center):
                    is_contained = True
                    break
            color = (44, 160, 44) if is_contained else (40, 39, 214)
            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 5)
        frame = cv2.resize(frame, (960, 540))
        is_quit = False
        while True:
            cv2.imshow('labeler', frame)
            k = cv2.waitKey(0)
            if k == 27: # Esc key
                is_quit = True
            else:
                break
        if is_quit:
            break
    ret, frame = cap.read()
    frame_id += 1
cap.release()