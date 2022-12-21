# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 04:36:30 2022

@author: repfe
"""


#%% convert drinking/waiting json to standard format
import os
import json

root_dir = r'D:\UW\OneDrive - UW-Madison\UW\Courses\Fall 2022\CS 762\Project\test_bounding_boxes'

with open(os.path.join(root_dir, 'drinking', 'drinkingBoundingBoxes.json'), 'r') as f:
    data = json.load(f)
file_dict = data['file']
metadata = data['metadata']
annotations = {}
for key, value in metadata.items():
    filename = file_dict[value['vid']]['fname'].replace(' (1)', '')
    if filename not in annotations:
        annotations[filename] = {
            'filename': filename,
            'size': 0,
            'file_attributes': {},
            'regions': []}
    if '1' in value['av'] and value['av']['1'] == '1':
        xy = value['xy']
        left = max(int(xy[1]), 0)
        top = max(int(xy[2]), 0)
        width = max(int(xy[3]), 0)
        height = max(int(xy[4]), 0)
        annotations[filename]['regions'].append({
            'region_attributes': {},
            'shape_attributes': {
                'name': 'rect',
                'x': left,
                'y': top,
                'width': width,
                'height': height}})
json_object = json.dumps(annotations)
with open(os.path.join(root_dir, 'nothing', 'nothing2_annotations_json.json'), 'w') as outfile:
    outfile.write(json_object)


#%% 
import os
import json
import cv2
import csv
import numpy as np

root_dir = r'D:\UW\OneDrive - UW-Madison\UW\Courses\Fall 2022\CS 762\Project'
bbox_dir = os.path.join(root_dir, 'test_bounding_boxes')
src_img_dir = os.path.join(root_dir, 'data', 'random_frames_v1', 'test')
superimposed_dir = os.path.join(root_dir, 'data', 'test_frames_v2_superimposed_manual')
cropped_dir = os.path.join(root_dir, 'data', 'test_frames_v2_cropped_manual')
background_dir = os.path.join(root_dir, 'background_frames')
change_video = '1572166740'
background_frames = []
for i in range(4):
    background_frames.append(cv2.imread(os.path.join(background_dir, f'bg{i}.jpg')))


def get_background_frame(video_key, bw):
    camera_angle = 0 if int(video_key) < int(change_video) else 1
    bg_id = 2 * camera_angle + bw
    return background_frames[bg_id]


coords_csv_rows = []
# classes = ['interaction', 'drinking', 'waiting']
# classes = ['drinking', 'interaction']
classes = ['interaction', 'drinking', 'waiting', 'nothing1', 'nothing2']
class_ids = {'interaction': '1', 'drinking': '5', 'waiting': '7', 'nothing1': '0', 'nothing2': '0'}
for classname in classes:
    with open(os.path.join(bbox_dir, classname, f'{classname}_annotations_json.json'), 'r') as f:
        annotations = json.load(f)
    for key, value in annotations.items():
        filename = value['filename']
        video_key = filename.split('_')[0]
        frame = cv2.imread(os.path.join(src_img_dir, filename))
        bw = int(np.mean(np.std(frame, axis=2)) < 1)
        for i, region in enumerate(value['regions']):
            background_frame = get_background_frame(video_key, bw).copy()
            shape_attributes = region['shape_attributes']
            left = shape_attributes['x']
            top = shape_attributes['y']
            width = shape_attributes['width']
            height = shape_attributes['height']
            if width > 0 and height > 0:
                cropped_frame = frame[top:top+height, left:left+width]
                background_frame[top:top+height, left:left+width] = cropped_frame
                label_id = class_ids[classname]
                output_sp_dir = os.path.join(superimposed_dir, 'test', label_id)
                os.makedirs(output_sp_dir, exist_ok=True)
                superimposed_filename = filename[:-4] + f'_{i}_{label_id}_superimposed.jpg'
                cv2.imwrite(os.path.join(output_sp_dir, superimposed_filename), background_frame)
                output_cropped_dir = os.path.join(cropped_dir, 'test', label_id)
                os.makedirs(output_cropped_dir, exist_ok=True)
                cropped_filename = filename[:-4] + f'_{i}_{label_id}_cropped.jpg'
                cv2.imwrite(os.path.join(output_cropped_dir, cropped_filename), cropped_frame)
                coords_csv_rows.append([cropped_filename, left, top, width, height])
            else:
                print(filename, i,' bbox size equals 0!')


with open(os.path.join(cropped_dir, 'coordinates.csv'), 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['filename', 'left', 'top', 'width', 'height'])
    for row in coords_csv_rows:
        writer.writerow(row)
        
