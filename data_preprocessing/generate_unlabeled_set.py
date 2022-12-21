# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 06:39:44 2022

@author: repfe
"""

import os
import json
import cv2
import csv
import numpy as np

root_dir = r'D:\UW\OneDrive - UW-Madison\UW\Courses\Fall 2022\CS 762\Project'
superimposed_dir = os.path.join(root_dir, 'data', 'unlabeled_frames_v2_superimposed_pipeline')
cropped_dir = os.path.join(root_dir, 'data', 'unlabeled_frames_v2_cropped_pipeline')
background_dir = os.path.join(root_dir, 'background_frames')
labels_dir = os.path.join(root_dir, 'd20_files', 'labels')
bboxes_dir = os.path.join(root_dir, 'results')
roi_1 = [(1087, 37, 564, 847),
         (78, 84, 1573, 800)]
roi_2 = [(1089, 62, 478, 815),
         (87, 135, 1480, 766)]
change_video = '1572166740'
background_frames = []
for i in range(4):
    background_frames.append(cv2.imread(os.path.join(background_dir, f'bg{i}.jpg')))

os.makedirs(os.path.join(superimposed_dir, 'unlabeled'), exist_ok=True)
os.makedirs(os.path.join(cropped_dir, 'unlabeled'), exist_ok=True)


videos_path = os.path.join(root_dir, 'd20_files', 'video data', 'good')
videos = os.listdir(videos_path)
n = len(videos)
change_id = videos.index(change_video + '.mp4')
train_videos = videos[:int(change_id*0.8)]
test_videos = videos[int(change_id*0.8):change_id]
train_videos.extend(videos[change_id:change_id+int((n-change_id)*0.8)])
test_videos.extend(videos[change_id+int((n-change_id)*0.8):])


def get_background_frame(video_key, bw):
    camera_angle = 0 if int(video_key) < int(change_video) else 1
    bg_id = 2 * camera_angle + bw
    return background_frames[bg_id]


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


def get_bboxes_dict():
    bboxes_dict = {}
    for bboxes_file in os.listdir(bboxes_dir):
        video_key = bboxes_file.split('_')[1][:-4]
        bboxes_dict[video_key] = {}
        with open(os.path.join(bboxes_dir, bboxes_file)) as f:
            lines = f.readlines()
        for line in lines:
            splits = line.split(' ')
            frame_id = int(splits[0])
            left = int(splits[2])
            top = int(splits[3])
            width = int(splits[4])
            height = int(splits[5])
            if frame_id not in bboxes_dict[video_key]:
                bboxes_dict[video_key][frame_id] = []
            bboxes_dict[video_key][frame_id].append((left, top, width, height))
    return bboxes_dict


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


def render_frame(video_key, bw, bbox, superimposed_filename, cropped_filename, frame):
    background_frame = get_background_frame(video_key, bw).copy()
    left = bbox[0]
    top = bbox[1]
    width = bbox[2]
    height = bbox[3]
    cropped_frame = frame[top:top+height, left:left+width]
    background_frame[top:top+height, left:left+width] = cropped_frame
    cv2.imwrite(os.path.join(superimposed_dir, 'unlabeled', superimposed_filename), background_frame)
    cv2.imwrite(os.path.join(cropped_dir, 'unlabeled', cropped_filename), cropped_frame)


bboxes_dict = get_bboxes_dict()
coords_csv_rows = []
filenames = set()


all_frames = []
for video in train_videos:
    cap = cv2.VideoCapture(os.path.join(videos_path, video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    try:
        with open(os.path.join(labels_dir, video[:-4] + '.json')) as f:
            data = json.load(f)
        data = np.array(data)
        n_labeled_frames = data.shape[0]
    except:
        # print(video[:-4], ' label not found!')
        n_labeled_frames = 0
    # print(video, n_frames, n_labeled_frames)
    if n_frames != n_labeled_frames:
        continue
    all_frames.extend(zip([int(video[:-4])] * n_frames, range(n_frames), data))
all_frames = np.array(all_frames)
training_frames = set()
for label in ['1', '5', '7']:
    training_frames.update(os.listdir(os.path.join(root_dir, 'data', 'valid_random_frames_v2_curated_cropped', 'train', label)))
np.random.seed(2023)
np.random.shuffle(all_frames)
frames_per_label = 1000
count_labels = {}
random_frames = []
for frame in all_frames:
    label_id = frame[2]
    if label_id not in [1,2,3,5,7]:
        continue
    if label_id in [1,2,3]:
        label_id = 1
    training_filename = f'{frame[0]}_{frame[1]:05d}_{label_id}_cropped.jpg'
    if training_filename in training_frames:
        continue
    current_count = count_labels.get(label_id, 0)
    if current_count < frames_per_label:
        random_frames.append(frame)
        count_labels[label_id] = count_labels.get(label_id, 0) + 1
random_frames = np.array(random_frames)
random_frames = random_frames[random_frames[:, 0].argsort()]
videos = np.unique(random_frames[:, 0])
for video in videos:
    video_key = str(video)
    cap = cv2.VideoCapture(os.path.join(videos_path, f'{video}.mp4'))
    video_frames = random_frames[random_frames[:, 0] == video]
    for frame_id in video_frames[:, 1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        bw = int(np.mean(np.std(frame, axis=2)) < 1)
        roi = roi_1 if int(video_key) < int(change_video) else roi_2
        bboxes = bboxes_dict[video_key].get(frame_id, [])
        valid_bboxes = []
        for bbox in bboxes:
            center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
            is_contained = False
            for rect in roi:
                if contains_point(rect, center):
                    is_contained = True
                    break
            if is_contained:
                valid_bboxes.append(bbox)
        for i, bbox in enumerate(valid_bboxes):
            # render individual bbox
            superimposed_filename = f'{video_key}_{frame_id:05d}_{i}_superimposed.jpg'
            cropped_filename = f'{video_key}_{frame_id:05d}_{i}_cropped.jpg'
            render_frame(video_key, bw, bbox, superimposed_filename, cropped_filename, frame)
            coords_csv_rows.append([cropped_filename, bbox[0], bbox[1], bbox[2], bbox[3]])
            # render overlapping combinations
            for j in range(i + 1, len(valid_bboxes)):
                other_bbox = valid_bboxes[j]
                if overlap([bbox, other_bbox]):
                    # print('overlap', filename, i, j)
                    ssleft = min(bbox[0], other_bbox[0])
                    sstop = min(bbox[1], other_bbox[1])
                    ssright = max(bbox[0] + bbox[2], other_bbox[0] + other_bbox[2])
                    ssbottom = max(bbox[1] + bbox[3], other_bbox[1] + other_bbox[3])
                    sswidth = ssright - ssleft
                    ssheight = ssbottom - sstop
                    superimposed_filename = f'{video_key}_{frame_id:05d}_{i}-{j}_superimposed.jpg'
                    cropped_filename = f'{video_key}_{frame_id:05d}_{i}-{j}_cropped.jpg'
                    render_frame(video_key, bw, [ssleft, sstop, sswidth, ssheight], superimposed_filename, cropped_filename, frame)
                    coords_csv_rows.append([cropped_filename, ssleft, sstop, sswidth, ssheight])
    cap.release()


with open(os.path.join(cropped_dir, 'coordinates.csv'), 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['filename', 'left', 'top', 'width', 'height'])
    for row in coords_csv_rows:
        writer.writerow(row)
        
