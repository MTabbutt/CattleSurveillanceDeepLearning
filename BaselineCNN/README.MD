
# BaselineCNN:

This is the code for some basic pre-processing scripts, running of the CNN, testing the CNN, and the experimental visualizations such as accuracy curves, statistics calculations and confusion matracies. 

## Codes: 

### 02 - Dataset exploration: 
Provided from the previous work, not used in our work. 

### 03 - CNN training and evaluation: 
Provided from the previous work, adapted for our work. Most adaptations was to the training input features and hyperparameters of the model. 

### 04 - CNN vizualization:
Provided from the previous work, not used in our work. 

___ 

### 10 - PreProcessingForGPURuns:
This is code to do some of the preprocessing including, taking a random subselection of video frames for the baseline CNN recreation, and downsizing and greyscaling the images. This was necessary as the 30GB of video data sat locally and the transfer speeds onto the shared GPU resources weren't great. This notebook also does specific formatting of arranging training and testing sets into a specific directory structure and writing out the truth labels of both to a json. 

### 20 - CNN training and Evaluation - Megan Refactor
This is a very similar notebook to 02 above, but the inputs are now already downsized and greyscaled and in a slightly different directory structure. Most of what was changed here in the CNN was file IO, and some hyperparamters changes. This notebook will take in a training set, split it into training and validation and run it through the CNN model saving the best model. The second half of the notebook has routines provided for testing as well but we didn't use it, there is a seperate notebook below for that. 

### 30 - Preparing Vizuals for Experiments:
This again is the testing code from above just seperated out, that we did not use. The code from the previous paper samples the testing set randomly an appropriate amount of times. We instead opted for the non-random approach of always running each testing image through once. That code is below. 

### 31 - Predicting on Test Sets:
This the main testing code that was used. It runs through the testing set, predicts on the model, saves those, and calculates common statistics for the testing set. It also makes confusion matracies based on these results and accuracy curves from the model's training. 

### 32 - Plotting Confusion Matrix Only:
This is a redundant code that just does the confusion matrix plots, that was convienent for making paper figures. 
