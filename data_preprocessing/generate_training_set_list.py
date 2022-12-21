# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:15:42 2022

@author: repfe
"""

#%% Extract labeled frames into videos
import os
import cv2
import json
import numpy as np

os.chdir(r'D:\UW\OneDrive - UW-Madison\UW\Courses\Fall 2022\CS 762\Project')
videos_dir = os.path.join('d20_files', 'video data', 'good')
labels_dir = os.path.join('d20_files', 'labels')
output_dir = os.path.join('data', 'labeled_frames')
labels_of_interest = [1, 2, 3, 5, 6, 7]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = 1920
height = 1080
fps = 25.0

for video in os.listdir(videos_dir):
    cap = cv2.VideoCapture(os.path.join(videos_dir, video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        with open(os.path.join(labels_dir, video[:-4] + '.json')) as f:
            labels = json.load(f)
        labels = np.array(labels)
        n_labeled_frames = labels.shape[0]
    except:
        n_labeled_frames = 0
    if n_frames != n_labeled_frames:
        # print(video, n_frames, n_labeled_frames)
        cap.release()
        continue
    ret, frame = cap.read()
    frame_id = 0
    output_video = None
    current_label_id = -1
    while ret:
        label_id = labels[frame_id]
        if label_id not in labels_of_interest:
            label_id = -1
        if label_id != -1:
            if label_id != current_label_id:
                if output_video is not None:
                    output_video.release()
                current_label_id = label_id
                output_video_name = f'{video[:-4]}_{frame_id:05d}_{label_id}.mp4'
                output_video = cv2.VideoWriter(os.path.join(output_dir, output_video_name), fourcc, fps, (width, height))
            output_video.write(frame)
        else:
            current_label_id = -1
        ret, frame = cap.read()
        frame_id += 1
    cap.release()
    if output_video is not None:
        output_video.release()


#%% Generate valid csv
import os
import cv2
import csv
import json
import numpy as np

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
labels_dir = os.path.join('d20_files', 'labels')
bboxes_dir = os.path.join('results')
good_videos_dir = os.path.join('d20_files', 'video data', 'good')
change_id = 1572166740
roi_1 = [(1087, 62, 564, 822),
         (378, 99, 1273, 785)]
roi_2 = [(1089, 62, 478, 815),
         (387, 155, 1180, 746)]
labels_of_interest = {1, 2, 3, 5, 7}
interaction_labels = {1, 2, 3}
invalid_videos = {'1571227140', '1572386700'}
label_x_separators = [1051, 972]


def get_bboxes_dict(bboxes_file):
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
    return bboxes_dict


def label_makes_sense(label_id, bbox_x, label_x_separator):
    if label_id == 5 and bbox_x < label_x_separator: # Drinking
        return False
    if label_id == 7 and bbox_x > label_x_separator: # Waiting in line
        return False
    return True


def overlap(bounding_boxes):
    bbox1 = bounding_boxes[0]
    bbox2 = bounding_boxes[1]
    if bbox1[0] > bbox2[0] + bbox2[2]: # 1 is completely to the right of 2
        return False
    if bbox1[0] + bbox1[2] < bbox2[0]: # 1 is completely to the left of 2
        return False
    if bbox1[1] > bbox2[1] + bbox2[3]: # 1 is completely below 2
        return False
    if bbox1[1] + bbox1[3] < bbox2[1]: # 1 is completely above 2
        return False
    return True


with open('valid_labeled_frames_v2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['video', 'label_id', 'frame_id', 'left', 'top', 'width', 'height'])
    for label_file in os.listdir(labels_dir):
        print(label_file)
        video_key = label_file[:-5]
        if video_key in invalid_videos:
            continue
        if not os.path.isfile(os.path.join(good_videos_dir, video_key + '.mp4')):
            continue
        with open(os.path.join(labels_dir, label_file)) as f:
            labels = json.load(f)
        bboxes_dict = get_bboxes_dict(f'pred_{video_key}.txt')
        roi = roi_1 if int(video_key) < change_id else roi_2
        label_x_separator = label_x_separators[0] if int(video_key) < change_id else label_x_separators[1]
        for frame_id, label_id in enumerate(labels):
            if label_id not in labels_of_interest:
                continue
            bboxes = bboxes_dict.get(frame_id, [])
            valid_bboxes = []
            for bbox in bboxes:
                center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                is_contained = False
                for rect in roi:
                    if contains_point(rect, center):
                        is_contained = True
                        break
                if is_contained and label_makes_sense(label_id, center[0], label_x_separator):
                    valid_bboxes.append(bbox)
            if len(valid_bboxes) > 0:
                valid_bboxes = np.stack(valid_bboxes)
            if (label_id in interaction_labels and len(valid_bboxes) == 2) or (label_id not in interaction_labels and len(valid_bboxes) == 1):
                # Check if overlapping and interaction
                if label_id in interaction_labels and not overlap(valid_bboxes):
                    continue
                left = valid_bboxes[:,0].min()
                top = valid_bboxes[:,1].min()
                right = (valid_bboxes[:,0] + valid_bboxes[:,2]).max()
                bottom = (valid_bboxes[:,1] + valid_bboxes[:,3]).max()
                writer.writerow([video_key, label_id, frame_id, left, top, right - left, bottom - top])


#%% Generate random valid frames
import os
import csv
import random
random.seed(2022)

os.chdir(r'D:\UW\OneDrive - UW-Madison\UW\Courses\Fall 2022\CS 762\Project')
videos_path = os.path.join('d20_files', 'video data', 'good')
videos = os.listdir(videos_path)
n = len(videos)
change_id = videos.index('1572166740.mp4')
train_videos = videos[:int(change_id*0.8)]
test_videos = videos[int(change_id*0.8):change_id]
train_videos.extend(videos[change_id:change_id+int((n-change_id)*0.8)])
test_videos.extend(videos[change_id+int((n-change_id)*0.8):])
interaction_labels = {1, 2, 3}

instances_per_class = 1000
labels_dict = {}
with open('valid_labeled_frames_v2.csv') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
        video_key = row[0]
        label_id = int(row[1])
        frame_id = int(row[2])
        left = int(row[3])
        top = int(row[4])
        width = int(row[5])
        height = int(row[6])
        if label_id in interaction_labels:
            label_id = 1
        if label_id not in labels_dict:
            labels_dict[label_id] = []
        labels_dict[label_id].append((video_key, frame_id, left, top, width, height))
with open('valid_labeled_frames_random_v2.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['split', 'video', 'frame_id', 'label_id', 'left', 'top', 'width', 'height'])
    for key, value in labels_dict.items():
        random.shuffle(value)
        value = value[:instances_per_class]
        for frame in value:
            video_key = frame[0]
            frame_id = frame[1]
            label_id = key
            split = 'test' if video_key + '.mp4' in test_videos else 'train'
            left = int(frame[2])
            top = int(frame[3])
            width = int(frame[4])
            height = int(frame[5])
            writer.writerow([split, video_key, frame_id, label_id, left, top, width, height])
            





