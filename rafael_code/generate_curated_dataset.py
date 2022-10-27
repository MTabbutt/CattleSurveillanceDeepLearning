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
labels_of_interest = {1, 2, 3, 5, 6, 7}
interaction_labels = {1, 2, 3}
invalid_videos = {'1571227140', '1572386700'}


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


with open('valid_labeled_frames.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['video', 'label_id', 'frame_id'])
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
        for frame_id, label_id in enumerate(labels):
            if label_id not in labels_of_interest:
                continue
            num_bboxes = 0
            bboxes = bboxes_dict.get(frame_id, [])
            for bbox in bboxes:
                center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                is_contained = False
                for rect in roi:
                    if contains_point(rect, center):
                        is_contained = True
                        break
                if is_contained:
                    num_bboxes += 1
            if (label_id in interaction_labels and num_bboxes == 2) or (label_id not in interaction_labels and num_bboxes == 1):
                writer.writerow([video_key, label_id, frame_id])


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

instances_per_class = 1000
labels_dict = {}
with open('valid_labeled_frames.csv') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
        video_key = row[0]
        label_id = int(row[1])
        frame_id = int(row[2])
        if label_id not in labels_dict:
            labels_dict[label_id] = []
        labels_dict[label_id].append((video_key, frame_id))
with open('valid_labeled_frames_random_v1.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['split', 'video', 'frame_id', 'label_id'])
    for key, value in labels_dict.items():
        random.shuffle(value)
        value = value[:instances_per_class]
        for frame in value:
            video_key = frame[0]
            frame_id = frame[1]
            label_id = key
            split = 'test' if video_key + '.mp4' in test_videos else 'train'
            writer.writerow([split, video_key, frame_id, label_id])
            

#%% Labeling UI 
import os
import cv2
import time

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
videos_dir = os.path.join('data', 'labeled_frames')
bboxes_dir = os.path.join('results')
change_id = 1572166740
roi_1 = [(1087, 62, 564, 822),
         (378, 99, 1273, 785)]
roi_2 = [(1089, 62, 478, 815),
         (387, 155, 1180, 746)]

def open_labeler(status_file):
    print(video)
    video_key = video.split('_')[0]
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
    frame_id = int(video.split('_')[1])
    # label_id = int(video.split('_')[2][:-4])
    while ret:
        if frame_id in bboxes_dict:
            bboxes = bboxes_dict[frame_id]
        else:
            bboxes = []
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
        while True:
            cv2.imshow('labeler', frame)
            k = cv2.waitKey(0)
            if k == 27: # Esc key
                cap.release()
                return -1
            elif k == -1:
                continue
            elif k == 100:
                # print('good ', frame_id)
                status_file.write('0')
                break
            elif k == 115:
                # print('not detected ', frame_id)
                status_file.write('1')
                break
            elif k == 97:
                # print('bad ', frame_id)
                status_file.write('2')
                break
            else:
                print('none ', frame_id)
        
        ret, frame = cap.read()
        frame_id += 1
    cap.release()
    print(video, ' complete!')
    time.sleep(5)
    return 0

for video in os.listdir(videos_dir):
    status_file = os.path.join('data', 'labeled_frames_status', f'{video[:-4]}.txt')
    if os.path.isfile(status_file):
        continue
    with open(status_file, 'w') as f:
        return_code = open_labeler(f)
    if return_code == -1:
        break
cv2.destroyAllWindows()



