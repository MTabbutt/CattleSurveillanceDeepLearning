from snorkel.labeling import labeling_function
from snorkel.labeling.model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter
import pandas as pd
import matplotlib.pyplot as plt
from snorkel.labeling import LFAnalysis
from snorkel.labeling import PandasLFApplier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
import os
from lambda_functions import *
import pickle
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix


# experiments to run
# 1) accuracy with and without nothing
# 2) accuracy with different combinations of labeling functions and hyperparameters
# 3) nothing or not for WS then just SVM for coordinates.
# 4) look at ho

# 1) make confusion matrix with code sent from Meg.
# 2) see accuracy on a per labeling function basis on train and test set--> cannot use test
        # set to guide results though.
# 3) run as many experiments as you deem fit.
# 4) get labels for all unlabelled data points, using train set, and own intuition only.
#           say that since this is a proof of concept paper using own results only.
# 5)

# look at width +  height  of bounding box to get an idea of if it is a group interaction,
# likely should be in a T formation, so should have prominence in both directions.
# only if multiple cows.


def load_df(dir1=None,dir2=None, labels=[1,5,7]):
    if dir1 is None :
        dir1=os.path.join('valid_random_frames_v2_curated_cropped', f"coordinates.csv")
    if dir2 is None:
        dir2= os.path.join('valid_random_frames_v2_curated_cropped', 'train')
    df = pd.read_csv(dir1).set_index('filename')
    all_fn = []

    y = []
    for i in labels:
        labels = os.listdir(os.path.join(dir2, str(i)))
        all_fn += labels
        y += [i] * len(labels)

    assert len(y) == len(all_fn)

    df_y = pd.DataFrame()
    df_y['filenames'] = all_fn
    df_y['Y'] = y
    df_y = df_y.set_index('filenames')

    df['Y'] = df_y['Y']
    df =  df[~np.isnan(df['Y'])]
    return df
def load_cattle_data():
    df_train=load_df()
    df_test = load_df(os.path.join('test', f"coordinates.csv"),'test',labels=[0,1,5,7])
    return df_train,df_test


def ws_confusion_matrix(y_gold,y_pred):
    abstains=sum(y_pred==-1)
  #  assert sorted(np.unique(y_gold))== [0,1,2,3]
    cm = confusion_matrix(y_gold, y_pred,labels=[-1,0,1,2,3]) / len(y_gold)
    labels = ['Abstain','Group Inter.', 'Waiting', 'Drinking', 'Nothing']
    df = pd.DataFrame(data=cm, index=labels, columns=labels)
    fig,ax= plt.subplots(1,1)
    ax=sns.heatmap(df, annot=True, linewidth=.5, cmap='GnBu',ax=ax)
    ax.set_ylabel('true label')
    ax.set_xlabel('predicted label')
    ax.set_title(f'Confusion matrix')
    return fig,ax

def unit_test_confusion_matrix():
    y_pred = [random.randint(0, 3) for _ in range(100)]
    y_gold = [random.randint(0, 3) for _ in range(100)]
    fig,ax=ws_confusion_matrix(y_gold,y_pred)
    fig.show()
def practice_confusion_matrix():
    y_pred = [random.randint(0, 3) for _ in range(100)]
    y_gold = [random.randint(0, 3) for _ in range(100)]
    print(y_gold, y_pred)
    cm = confusion_matrix(y_gold, y_pred) / 100
    labels = ['Group Inter.', 'Drinking', 'Waiting', 'Nothing']
    df = pd.DataFrame(data=cm, index=labels, columns=labels)
    sns.heatmap(df, annot=True, linewidth=.5,cmap='GnBu')
    plt.title('Confusion matrix for weak supervision')


def empirical_accuracies_labeling_functions():
    df_train=load_train_with_coordinates()
    df_test = load_train_with_coordinates(path=os.path.join('test','coords_with_keypoints_and_y.csv'))
    lfs = [width1,
           height1,
           top1,
           left1,
           drink1,
           drink2,
           nothing1,
           nothing2,
           waiting1,
           waiting2,
           svm_classifier,
           # must include
           # 1) cnn trained on training data.
           # 2) group interactions of multiple bounding boxes
    ]
    # three experiments to run
    # heuristic labeling functions only
    # labeling functions only
    # both --- see accuracy and results on all.


    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train)
    L_test  = applier.apply(df=df_test)
    lf_anal_train=LFAnalysis(L=L_train, lfs=lfs)

    summary_train= lf_anal_train.lf_summary()
    summary_train['Accuracies']=lf_anal_train.lf_empirical_accuracies(df_train['Y'])

    summary_train.plot.bar(y=['Coverage', 'Conflicts', 'Accuracies'], rot=45, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join('results','labeling_funcs','summary_stats_train.png'))
    plt.close()


    lf_anal_test= LFAnalysis(L=L_test,lfs=lfs)
    summary_test = lf_anal_test.lf_summary()
    summary_test['Accuracies'] = lf_anal_test.lf_empirical_accuracies(df_test['Y'])

    summary_test.plot.bar(y=['Coverage','Conflicts','Accuracies'],rot=45,alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join('results','labeling_funcs','summary_stats_test.png'))

    print('stop')

def plot_label_frequency(L,uuid):
    fig,ax=plt.subplots(1,1)
    ax.hist((L != ABSTAIN).sum(axis=1), density=True, bins=range(L.shape[1]),
             alpha=0.3)
    ax.set_xlabel("Number of labels")
    ax.set_ylabel("Fraction of dataset")
    fig.savefig(os.path.join('results','ws',str(uuid),'label_frequencies.png'))
def eval_single_model(model):
    A, acc_test,A_cv,acc_cv= baseline_model(model)
    A_cv=list(np.array(A_cv).reshape(-1))
    acc_cv=list(np.array(acc_cv).reshape(-1))
    plt.semilogx(A_cv,acc_cv, 'x',label=f'cross valid')
    plt.semilogx(A,acc_test,'x',label=f'test')

    plt.title(f'accuracy vs alpha with model: {model}',fontsize=16)
    plt.xlabel('alpha',fontsize=16)
    plt.ylabel('accuracy',fontsize=16)
    plt.legend()
    plt.savefig(os.path.join('results','baseline',f'{model}_5foldCV_test_[0,1,5,7].png'))
    plt.close()

def save_train():
    path = os.path.join('valid_random_frames_v2_curated_cropped', f"coords_with_keypoints.csv")
    df = load_df(path)
    df.to_csv(os.path.join('valid_random_frames_v2_curated_cropped','coords_with_keypoints_and_y.csv'))
    print('successful saved ! ')
def save_test():
    df= load_df(dir1=os.path.join('test', 'coords_with_keypoints.csv'),
            dir2='test', labels=[0, 1, 5, 7])
    df.to_csv(os.path.join('test', 'coords_with_keypoints_and_y.csv'))
    print('successful saved ! ')

def load_train_with_coordinates(path=None):
    if path is None:
        path= os.path.join('valid_random_frames_v2_curated_cropped', f"coords_with_keypoints_and_y.csv")
    df= pd.read_csv(path)
    mmpose= ['left_eye', 'right_eye', 'nose', 'neck', 'root_of_tail'\
        , 'left_shoulder', 'left_elbow', 'left_front_paw', 'right_shoulder', \
    'right_elbow', 'right_front_paw', 'left_hip', 'left_knee', 'left_back_paw', 'right_hip', \
    'right_knee', 'right_back_paw']

    for pose in mmpose:
        df[pose]=df[pose].apply(lambda x:eval(x))

    label_dict = {1:GROUP,5:DRINK,7:WAITING,0:NOTHING}
    df['Y']= df['Y'].apply(lambda  x: label_dict[x])

    return df

def plot_weights(L_train,lfs,label_model):
    fig,ax =plt.subplots(1,1)
    lf_anal_test = LFAnalysis(L=L_train, lfs=lfs)
    summary_test = lf_anal_test.lf_summary()
    summary_test['weights'] = label_model.get_weights()
    ax=summary_test.plot.bar(y='weights', rot=45, alpha=0.5, ax=ax)
    ax.set_title('mu weights for each labeling functions')
    ax.set_ylabel('mu')
    fig.tight_layout()
    return fig,ax



def make_bar_plot_of_Lf_summaries(L_test, lfs, df_test):
    lf_anal_test = LFAnalysis(L=L_test, lfs=lfs)
    summary_test = lf_anal_test.lf_summary()
    summary_test['Accuracies'] = lf_anal_test.lf_empirical_accuracies(df_test['Y'])
    fig, ax = plt.subplots(1, 1)
    ax = summary_test.plot.bar(y=['Coverage', 'Conflicts', 'Accuracies'], rot=45, alpha=0.5, ax=ax)
    fig.tight_layout()
    return fig, ax

def weak_supervision_unlabeled():
    lfs = [width1,
           height1,
           top1,
           left1,
           drink1,
           drink2,
           nothing1,
           nothing2,
           waiting1,
           waiting2,
           svm_classifier,
            group1
           ]

    df_unlabeled = pd.read_csv(os.path.join('unlabeled','coords_with_keypoints_unlabeled.csv')).set_index('filename')
    df_interaction= pd.read_csv(os.path.join('unlabeled','interaction_separated_coordinates_unlabeled.csv')).set_index('filename')
    df_coords=pd.read_csv(os.path.join('unlabeled','coordinates.csv')).set_index('filename')

    mmpose = ['left_eye', 'right_eye', 'nose', 'neck', 'root_of_tail',
              'left_shoulder', 'left_elbow', 'left_front_paw', 'right_shoulder',
              'right_elbow', 'right_front_paw', 'left_hip', 'left_knee', 'left_back_paw', 'right_hip',
              'right_knee', 'right_back_paw']

    for pose in mmpose:
        df_coords[pose] = df_unlabeled[pose].apply(lambda x: eval(x))


    for feature in ['left1','top1','width1','height1','left2','top2','width2','height2']:
        df_coords[feature]= df_interaction[feature]

    ###### ANALYSIS OF LABELING FUNCTIONS ########
    applier = PandasLFApplier(lfs=lfs)
    L= applier.apply(df=df_coords)


    lf_anal_test = LFAnalysis(L=L, lfs=lfs)
    summary_test = lf_anal_test.lf_summary()

    summary_test.to_csv(os.path.join('results','unlabeled','summary.csv'))

    print(summary_test)


    cardinality = 4
    epochs = 1000
    log_freq = 10
    seed = 100

    label_model = LabelModel(cardinality=cardinality, verbose=True)
    label_model.fit(L_train=L, n_epochs=epochs, log_freq=log_freq, seed=seed)

    fig, _ = plot_weights(L, lfs, label_model)
    fig.savefig(os.path.join('results', 'unlabeled','weights.png'))


    predictions= label_model.predict(L=L)
    print(f'found {sum(predictions==-1)} abstains')
    # NOTHING = 3
    # DRINK = 2
    # WAITING = 1
    # GROUP = 0
    # ABSTAIN = -1
    reversedict={-1:0,0:1,1:7,2:5,3:0}
    df=pd.DataFrame(index=df_coords.index)


    df['predictions']=predictions
    df['predictions']= df['predictions'].apply(lambda x:reversedict[x])

    df.to_csv(os.path.join('results','unlabeled','predictions_unlabeled.csv'))


def weak_supervision_pipeline(uuid):

    #### DEFINE TRAIN,TEST SET ALONG WITH LFS ######
    df_train=load_train_with_coordinates()
    df_test = load_train_with_coordinates(path=os.path.join('test','coords_with_keypoints_and_y.csv'))
    lfs = [width1,
           height1,
           top1,
           left1,
           drink1,
           drink2,
           nothing1,
           nothing2,
           waiting1,
           waiting2,
           svm_classifier,

    ]

    ###### ANALYSIS OF LABELING FUNCTIONS ########
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train)
    L_test = applier.apply(df=df_test)

    fig,ax=make_bar_plot_of_Lf_summaries(L_train,lfs,df_train)
    fig.savefig(os.path.join('results','ws',str(uuid),'train_label_acc.png'))

    fig,ax =make_bar_plot_of_Lf_summaries(L_test,lfs,df_test)
    fig.savefig(os.path.join('results','ws',str(uuid),'test_label_acc.png'))

    plot_label_frequency(L_train, uuid)


    ########## MAJORITY LABEL VOTER ###########
    majority_model = MajorityLabelVoter(cardinality=4)
    preds_train_mm = majority_model.predict(L=L_train)

    fig,_=ws_confusion_matrix(df_train['Y'],preds_train_mm)
    fig.savefig(os.path.join('results','ws',str(uuid),'majority_label_confusion_train.png'))

    fig,_ = ws_confusion_matrix(df_test['Y'],majority_model.predict(L=L_test))
    fig.savefig(os.path.join('results','ws',str(uuid),'majority_label_confusion_test.png'))



    # ########## LABEL MODEL ############

    cardinality=4
    epochs=1000
    log_freq=10
    seed=100


    label_model = LabelModel(cardinality=cardinality, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=epochs, log_freq=log_freq, seed=seed)

    fig, _ = ws_confusion_matrix(df_train['Y'], label_model.predict(L=L_train))
    fig.savefig(os.path.join('results', 'ws', str(uuid), 'label_model_confusion_train.png'))

    fig, _ = ws_confusion_matrix(df_test['Y'], label_model.predict(L=L_test))
    fig.savefig(os.path.join('results', 'ws', str(uuid), 'label_model_confusion_test.png'))


    # plot the weights of \mu to show importance of each labeling function!
    fig,_=plot_weights(L_train, lfs, label_model)
    fig.savefig(os.path.join('results','ws',str(uuid),'weights.png'))


    #### get accuracies of all models  #########
    majority_acc = majority_model.score(L=L_train, Y=df_train['Y'], tie_break_policy="random")["accuracy"]
    label_model_acc = label_model.score(L=L_train, Y=df_train['Y'], tie_break_policy="random")["accuracy"]

    ##### test set accuracies
    label_model_acc_test= label_model.score(L=L_test,Y=df_test['Y'],tie_break_policy='random')['accuracy']
    majority_acc_test= majority_model.score(L=L_test,Y=df_test['Y'],tie_break_policy='random')['accuracy']

    # print to terminal
    print(f"{'Label Model Train Accuracy:':<25} {label_model_acc * 100:.1f}%")
    print(f"{'Label Model Test Accuracy:':<25} {label_model_acc_test * 100:.1f}%")
    print(f"{'Majority Vote Accuracy on Train:':<25} {majority_acc * 100:.1f}%")
    print(f"{'Majority Test Accuracy:':<25} {majority_acc_test * 100:.1f}%")


    res = pd.DataFrame()


    res['labeling_funcs']=lfs
    res['uuid'] = uuid
    res['train-mm']=majority_acc
    res['test-mm']=majority_acc_test
    res['train-lm']=label_model_acc
    res['test-lm']=label_model_acc_test
    res['epochs']=  epochs
    res['cardinality']=cardinality
    res['seed']=seed
    res['log_freq']=log_freq
    res.to_csv(os.path.join('results','ws',str(uuid),'results.csv'))





def baseline_model(model_name):
    df_train, df_test = load_cattle_data()
    # we need to do cross validation.
    acc_test=[]
    acc_cv=[]
    A= [0.001, 0.01, 0.1, 1, 10,100]
    A_cv=[]
    for alpha in A:
        # 1) for each value of alpha we need to run 5 fold cross validation.
        if model_name == 'Ridge':
            model = RidgeClassifier(alpha=alpha)
        elif model_name == 'SVC':
            model = SVC(C=alpha)
        else:
            raise Exception('Ahhh, running a model which doesnt exist')

        scores_cv = cross_val_score(model, df_train[['left', 'top', 'width', 'height']],
                                    df_train['Y'], cv=5)
        A_cv.append([alpha]*5)
        acc_cv.append(scores_cv)

        model.fit(df_train[['left','top','width','height']],df_train['Y'])
        with open(os.path.join('results','baseline','saved_models',
                               f'{model_name}_{alpha}.pkl'),'ab') as f:
            pickle.dump(model,f)
        acc_test.append(model.score(df_test[['left','top','width','height']],df_test['Y']))

    return A, acc_test,A_cv,acc_cv

if __name__ == '__main__':
    weak_supervision_unlabeled()
    # weak_supervision_pipeline(3)
    # empirical_accuracies_labeling_functions()
    # save_test()
    # save_train()
    # eval_single_model('Ridge')
    # eval_single_model('SVC')
    # df=pd.read_csv(os.path.join('test','coordinates.csv'))
    # print(df)