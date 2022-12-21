# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 05:10:31 2022

@author: repfe
"""

import os
import cv2
import csv
import numpy as np

os.chdir(r'D:\UW\OneDrive - UW-Madison\UW\Courses\Fall 2022\CS 762\Project')
csv_file = 'valid_labeled_frames_random_v2.csv'
bw_csv_file = 'videos_bw.csv'
background_dir = 'background_frames'
videos_dir = os.path.join('d20_files', 'video data', 'good')
src_img_dir = os.path.join('data', 'valid_random_frames_v2_curated')
superimposed_dir = os.path.join('data', 'valid_random_frames_v2_curated_superimposed')
cropped_dir = os.path.join('data', 'valid_random_frames_v2_curated_cropped')
change_video = '1572166740'
background_frames = []
for i in range(4):
    background_frames.append(cv2.imread(os.path.join(background_dir, f'bg{i}.jpg')))

# videos = set()
# with open(csv_file) as f:
#     reader = csv.reader(f, delimiter=',')
#     next(reader)
#     for row in reader:
#         video = row[1]
#         videos.add(video)
# bw_dict = {}
# for video in videos:
#     print(video)
#     cap = cv2.VideoCapture(os.path.join(videos_dir, video + '.mp4'))
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
#     ret, frame = cap.read()
#     bw_score = frame.std(axis=2).mean()
#     bw_dict[video] = bw_score
#     cap.release()

bw_dict = {}
with open(bw_csv_file) as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
        video = row[0]
        bw = int(row[2])
        bw_dict[video] = bw


def get_background_frame(video_key):
    camera_angle = 0 if int(video_key) < int(change_video) else 1
    bg_id = 2 * camera_angle + bw_dict[video_key]
    return background_frames[bg_id]


frames_list = {}
with open(csv_file) as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
        split = row[0]
        video = row[1]
        frame_id = int(row[2])
        label_id = int(row[3])
        left = int(row[4])
        top = int(row[5])
        width = int(row[6])
        height = int(row[7])
        if video not in frames_list:
            frames_list[video] = (split, [])
        frames_list[video][1].append([frame_id, label_id, left, top, width, height])
for key, value in frames_list.items():
    frames_list[key] = (value[0], np.stack(value[1]))

# Superimpose all frames
# for key, value in frames_list.items():
#      cap = cv2.VideoCapture(os.path.join('d20_files', 'video data', 'good', key + '.mp4'))
#      frame_ids_list = list(value[1][:,0])
#      for i, frame_id in enumerate(frame_ids_list):
#          cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
#          ret, frame = cap.read()
#          label_id = value[1][i, 1]
#          left = value[1][i, 2]
#          top = value[1][i, 3]
#          width = value[1][i, 4]
#          height = value[1][i, 5]
#          # find corresponding background frame
#          background_frame = get_background_frame(key).copy()
#          # superimpose
#          background_frame[top:top+height, left:left+width] = frame[top:top+height, left:left+width]
#          frame_name = f'{key}_{frame_id:05d}_{label_id}.jpg'
#          cv2.imwrite(os.path.join(output_dir, value[0], frame_name), background_frame)
#      cap.release()

# Superimposed just curated frames (training set only)
coords_csv_rows = []
for folder in ['incomplete', 'correct']:
    frames_dir = os.path.join(src_img_dir, folder, 'train')
    for file in os.listdir(frames_dir):
        splits = file.split('_')
        video_key = splits[0]
        frame_id = int(splits[1])
        label_id = splits[2][:-4]
        coords_array = frames_list[video_key][1]
        coords_found = False
        for coords in coords_array:
            if coords[0] == frame_id:
                left = coords[2]
                top = coords[3]
                width = coords[4]
                height = coords[5]
                coords_found = True
                break
        if coords_found:
            frame = cv2.imread(os.path.join(frames_dir, file))
            cropped_frame = frame[top:top+height, left:left+width]
            background_frame = get_background_frame(video_key).copy()
            background_frame[top:top+height, left:left+width] = cropped_frame
            output_sp_dir = os.path.join(superimposed_dir, 'train', label_id)
            os.makedirs(output_sp_dir, exist_ok=True)
            superimposed_filename = file[:-4] + '_superimposed.jpg'
            cv2.imwrite(os.path.join(output_sp_dir, superimposed_filename), background_frame)
            output_cropped_dir = os.path.join(cropped_dir, 'train', label_id)
            os.makedirs(output_cropped_dir, exist_ok=True)
            cropped_filename = file[:-4] + '_cropped.jpg'
            cv2.imwrite(os.path.join(output_cropped_dir, cropped_filename), cropped_frame)
            coords_csv_rows.append([cropped_filename, left, top, width, height])

with open(os.path.join(cropped_dir, 'coordinates.csv'), 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['filename', 'left', 'top', 'width', 'height'])
    for row in coords_csv_rows:
        writer.writerow(row)



