import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import json
from pipeline_cattle import load_train_with_coordinates,load_df




def make_feature_histograms():
    df_train=load_df()
    for column in df_train.columns:
        fig,ax=plt.subplots(1,1)
        obj_labels=['group','drinking','waiting']
        for j,i in enumerate([1,5,7]):
            ax=df_train[df_train['Y']==i].hist(column=[column],bins=50,alpha=0.3,label=obj_labels[j]
                                            ,ax=ax)
        ax=ax[0]
        ax.set_title(column,fontsize=18)
        ax.set_ylabel('frequency',fontsize=18)
        fig.legend(fontsize=18)
        fig.savefig(os.path.join('results','labeling_funcs',f'{column}_train.png'))


def analyze_orientation_vector(f4o):
    print('starting load')
    df_train = load_train_with_coordinates()
    print('ending load')

    fig, ax = plt.subplots(1, 1)
    fig2,ax2=plt.subplots(1,1)
    obj_labels = ['group', 'waiting','drinking']
    color = ['b', 'g', 'r']
    for j, i in enumerate([0, 1, 2]):
        y_features = df_train[df_train['Y'] == i]
        U,V,X,Y = [],[],[],[]
        for i in range(len(y_features)):
            row=y_features.iloc[i]
            U.append( row[f4o[1]][0]- row[f4o[0]][0])
            V.append(row[f4o[1]][1] - row[f4o[0]][1])
            X.append( row[f4o[0]][0])
            Y.append(row[f4o[0]][1])
        ax.quiver(X,Y,U,V,alpha=0.3,color=color[j],label=obj_labels[j])
        ax2.scatter(U,V,alpha=0.3,c=color[j],label=obj_labels[j])

    fig2.legend()
    ax2.set_title(f'direction orientation vector from {f4o[0]} to {f4o[1]}')
    ax2.set_ylabel('y direction')
    ax2.set_xlabel('x direction')
    fig2.savefig(os.path.join('results','labeling_funcs',f'orientation_direction_{f4o}.png'))

    fig.legend()
    ax.set_title(f'orientation vector from {f4o[0]} to {f4o[1]}')
    fig.savefig(os.path.join('results','labeling_funcs',f'orientation_{f4o}.png'))

def analyze_mmpose_labeling_functions():
    df_train= load_train_with_coordinates()
    mmpose=['left_eye', 'right_eye', 'nose', 'neck', 'root_of_tail'\
        , 'left_shoulder', 'left_elbow', 'left_front_paw', 'right_shoulder', \
    'right_elbow', 'right_front_paw', 'left_hip', 'left_knee', 'left_back_paw', 'right_hip', \
    'right_knee', 'right_back_paw']

    obj_labels = ['group', 'waiting','drinking']
    color=['b','g','r']
    for feature in mmpose:
        fig, ax = plt.subplots(1, 1)
        for j,i in enumerate([0,1,2]):
            col = df_train[df_train['Y'] == i][feature]
            x = [p[0] for p in col]
            y = [p[1] for p in col]
            ax.scatter(x, y, alpha=0.3, label=obj_labels[j],c=color[j])
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.set_title(feature)
        fig.legend()
        fig.savefig(os.path.join('results','labeling_funcs',f'{feature}_2d_scatter.png'))



# y_truth: array of true values (N, 1)
# y_predicted: array of predicted values (N, 1)



if __name__ == '__main__':
    # practice_confusion_matrix()
    # unit_test_confusion_matrix()
    # make_feature_histograms()
    # analyze_mmpose_labeling_functions()
    feature4body= ['root_of_tail','neck']
    analyze_orientation_vector(feature4body)
    feature4head = ['neck', 'nose']
    analyze_orientation_vector(feature4head)