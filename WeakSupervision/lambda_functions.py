import numpy as np
import pandas as pd
from snorkel.labeling import labeling_function
import os,pickle


NOTHING=3
DRINK=2
WAITING=1
GROUP=0
ABSTAIN=-1

label_dict = {1: GROUP, 5: DRINK, 7: WAITING, 0: NOTHING}

with open(os.path.join('results','baseline','saved_models','SVC_10.pkl'),'rb') as fileSVC:
    SVC_model=pickle.load(fileSVC)
    print('successfully loaded SVC model')

# with open(os.path.join('results','baseline','saved_models','Ridge_1.pkl'),'rb') as fileRidge:
#     Ridge_model=pickle.load(fileRidge)
#     print('successfully loaded Ridge')

# @labeling_function()
# def ridge_classifier(x):
#     x = np.array(x[['left', 'top', 'width', 'height']]).reshape(1, -1)
#     return label_dict[SVC_model.predict(x)[0]]

@labeling_function()
def group1(t):
    right1 = t['left1']+t['width1']
    right2 = t['left2']+t['width2']
    bottom1 = t['top1'] + t['height1']
    bottom2 = t['top2'] + t['height2']
    xA= max(t['left1'],t['left2'])
    yA= max( t['top1'],t['top2'])
    xB = min(right1, right2)
    yB = min(bottom1, bottom2)
    inter = (xB - xA) * (yB - yA)
    union = (t['width1']*t['height1'])+(t['width2']*t['height2'])-inter
    IOU = inter / union
    if IOU > 0.05:
        return GROUP
    return ABSTAIN


@labeling_function()
def svm_classifier(x):
    x=np.array(x[['left', 'top', 'width', 'height']]).reshape(1,-1)
    return label_dict[SVC_model.predict(x)[0]]

@labeling_function()
def width1(x):
    # if width is greater than 475 assume that it is a group interaction
    if x['width']  > 475:
        return GROUP
    return ABSTAIN
@labeling_function()
def height1(x):
    # based off height histogram
    if x['height']  < 225 :
        return DRINK
    elif x['height'] > 400 :
        return GROUP
    else:
        return ABSTAIN

@labeling_function()
def top1(x):
    # based off top histogram
    if x['top']> 175:
        return WAITING
    elif x['top']==0:
        return DRINK
    return ABSTAIN

@labeling_function()
def left1(x):
    # based off leftmost coordinate, based on histograms as well.
    if x['left']>1000:
        return DRINK
    return ABSTAIN


@labeling_function()
def drink1(x):
    # now using the
    if not np.all(np.isnan(x['nose'])):
        if x['nose'][0] > x['neck'][0]:
            if x['nose'][0] > 1920 / 2 and x['nose'][0] < 1611 and x['nose'][1] < 1080 / 2:
                return DRINK
    return ABSTAIN


@labeling_function()
def drink2(x):
    if not np.all(np.isnan(x['left_front_paw'])):
        if x['left_front_paw'][0] > x['left_back_paw'][0]:
            if x['left_front_paw'][0] > 1920 / 2 and x['left_front_paw'][0] < 1611 \
                    and x['left_front_paw'][1] < 1080 / 2:
                return DRINK
    return ABSTAIN

@labeling_function()
def nothing1(x):
    if not np.all(np.isnan(x['nose'])):
        if x['nose'][0] > 1920 / 2 and x['nose'][1] > 1080 / 2:
            return NOTHING
    return ABSTAIN

@labeling_function()
def nothing2(x):
    if not np.all(np.isnan(x['nose'])):
        if x['nose'][0] > 1611 and x['nose'][1] < 361:
            return NOTHING
    return ABSTAIN

@labeling_function()
def waiting1(x):
    if not np.all(np.isnan(x['left_front_paw'])):
        if x['left_front_paw'][0] < x['left_back_paw'][0] and x['left_front_paw'][0] < 1920 / 2:
            return WAITING
    return ABSTAIN

@labeling_function()
def waiting2(x):
    if  not np.all(np.isnan(x['left_front_paw'])):
        if x['left_front_paw'][1] < x['left_back_paw'][1] and x['left_front_paw'][0] < 1920 / 2:
            return WAITING
    return ABSTAIN

# width to height ratio indicative of orientation.

# trained SVM...

# number of edges

# trained CNN...

# look into accuracy of label.

# confidence level..

# munching on treats label.

# orientation  in the function.


# do both the Weak supervision then classifier
# do classifier then weak sueprvision

# pipeline stuff.

# weak supervision stuff.

# @labeling_function
# def orientation(x):
#     # if the orientation of the cow is in a certain spot then maybe its facing a certain way.
#     # if its facing right and close to it then yeah. Idk. this may be uncessary.
#     pass

