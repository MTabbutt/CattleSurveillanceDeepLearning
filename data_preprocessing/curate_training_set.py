# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 03:01:21 2022

@author: repfe
"""

import os
import cv2
import csv

os.chdir(r'D:\UW\OneDrive - UW-Madison\UW\Courses\Fall 2022\CS 762\Project')
csv_file = 'valid_labeled_frames_random_v2.csv'
output_dir = os.path.join('data', 'valid_random_frames_v2_curated')
correct_dir = os.path.join(output_dir, 'correct')
incomplete_dir = os.path.join(output_dir, 'incomplete')
incorrect_dir = os.path.join(output_dir, 'incorrect')
videos_dir = os.path.join('d20_files', 'video data', 'good')
colors = {1:(40, 39, 214), 5:(180, 119, 31), 7:(44, 160, 44)}
num_adjacent_frames = 0
last_frame = None


def get_adjacent_frames(video_key, frame_id):
    cap = cv2.VideoCapture(os.path.join(videos_dir, video_key + '.mp4'))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame_id = max(frame_id - int(num_adjacent_frames / 2), 0)
    end_frame_id = min(frame_id + int(num_adjacent_frames / 2), total_frames - 1)
    frames = []
    main_frame = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
    current_frame_id = start_frame_id
    ret, frame = cap.read()
    while ret:
        if current_frame_id <= end_frame_id:
            frames.append(frame)
            if current_frame_id == frame_id:
                main_frame = frame.copy()
        else:
            break
        ret, frame = cap.read()
        current_frame_id += 1
    cap.release()
    return main_frame, frames


def frame_file_exists(frame_file):
    for directory in os.listdir(output_dir):
        for split in os.listdir(os.path.join(output_dir, directory)):
            if os.path.isfile(os.path.join(output_dir, directory, split, frame_file)):
                return True
    return False



bbox_dict = {}
with open(csv_file) as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
        split = row[0]
        video_key = row[1]
        frame_id = int(row[2])
        label_id = int(row[3])
        left = int(row[4])
        top = int(row[5])
        width = int(row[6])
        height = int(row[7])
        key = f'{video_key}_{frame_id:05d}'
        bbox_dict[key] = [split, label_id, left, top, width, height]


keys = list(bbox_dict.keys())
keys.sort()

key_id = 0
while key_id < len(keys):
    key = keys[key_id]
    splits = key.split('_')
    video_key = splits[0]
    frame_id = int(splits[1])
    value = bbox_dict[key]
    split = value[0]
    label_id = value[1]
    bbox = value[2:]
    frame_file = f'{key}_{label_id}.jpg'
    if not frame_file_exists(frame_file):
        print(key, end=' ')
        main_frame, frames = get_adjacent_frames(video_key, frame_id)
        processed_frames = []
        color = colors[label_id]
        for frame in frames:
            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 5)
            frame = cv2.resize(frame, (960, 540))
            processed_frames.append(frame)
        quit_program = False
        while True:
            quit_infinite = False
            for frame in processed_frames:
                cv2.imshow('labeler', frame)
                k = cv2.waitKey(int(1000/25))
                if k == 27: # Esc key
                    quit_infinite = True
                    quit_program = True
                    break
                if k == -1: # waitKey timeout
                    continue
                if k == 100: # 'd' key
                    print('correct')
                    dst = os.path.join(correct_dir, split, frame_file)
                    cv2.imwrite(dst, main_frame)
                    last_frame = dst
                    quit_infinite = True
                    break
                if k == 115: # 's' key
                    print('incomplete')
                    dst = os.path.join(incomplete_dir, split, frame_file)
                    cv2.imwrite(dst, main_frame)
                    last_frame = dst
                    quit_infinite = True
                    break
                if k == 97: # 'a' key
                    print('incorrect')
                    dst = os.path.join(incorrect_dir, split, frame_file)
                    cv2.imwrite(dst, main_frame)
                    last_frame = dst
                    quit_infinite = True
                    break
                if k == 49: # '1' key
                    print('adj frames: 0')
                    num_adjacent_frames = 0
                    key_id -= 1
                    quit_infinite = True
                    break
                if k == 50: # '2' key
                    print('adj frames: 12')
                    num_adjacent_frames = 12
                    key_id -= 1
                    quit_infinite = True
                    break
                if k == 51: # '3' key
                    print('adj frames: 25')
                    num_adjacent_frames = 25
                    key_id -= 1
                    quit_infinite = True
                    break
                if k == 52: # '4' key
                    print('adj frames: 100')
                    num_adjacent_frames = 100
                    key_id -= 1
                    quit_infinite = True
                    break
                if k == 119: # 'w' key
                    print('rewind')
                    if last_frame is not None:
                        os.remove(last_frame)
                    last_frame = None
                    key_id -= 2
                    quit_infinite = True
                    break
                else:
                    print(k)
            if quit_infinite:
                break
        if quit_program:
            break
    key_id +=1 
cv2.destroyAllWindows()

