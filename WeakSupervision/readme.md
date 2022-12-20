## Weak Supervision 

To create the conda environment to re-create the results of the weak supervision predictions run

`conda create --name cattle`

`conda activate cattle`

`conda install -c conda-forge snorkel=0.9.9`

`conda install numpy matplotlib pandas`

`conda install -c anaconda scikit-learn` 

`conda install -c anaconda seaborn` 

and now the environment is created to run the weak supervision code.

### Dataset curation 

Make the following directories in this directory and post the following data in them to run the code. 

Test set in /test/ : https://drive.google.com/drive/folders/15eVgKiVeFdhuR91ARb_eiaV0rtrt6fKl?usp=sharing

Train set in /valid_random_frames_v2_curated_cropped/  : https://drive.google.com/drive/folders/1dJI-upHXQ47Wji4d4XWBIISRomtk1MFn?usp=sharing 

Unlabeled set in /unlabeled/ : https://drive.google.com/drive/folders/11ZcCVpo9gjVISKYDPMz66ET5IKk5Y7cU?usp=sharing 

Put this file in unlabeled directory : https://drive.google.com/file/d/1fbdrpMu0nQZaI7Zq_J3D9XLUMMNuc7Ch/view?usp=sharing 

### Main Pipeline 

To run the main pipeline look to `pipeline_cattle.py`. Many functions are defined which run different parts of the paper.
You must uncomment different parts to run different analysis in the paper. **I would recommend contacting Bryce Johnson (email listed below) if you
are consider running this code.** For reference the two main functions are `weak_supervision_pipeline(uuid)` which requires as input only the unique identifier of the experiment one is running. The second main function is `weak_supervision_unlabeled()`, which labels all the data in the unlabeled dataset. The function `empirical_accuracies_labeling_functions()` analyzes and saves the accuracies of different functions, and `eval_single_model('Ridge')` or `eval_single_model('SVC')` run the hyperparameter sweeps and outputs for the trained labeling functions. 

### Defining a labeling function 

All labeling functions are defined in `lambda_functions.py`. Here you must use the decorator @labeling_function() before you write your own labeling functions. 

### Analysis of labeling functions 

To run the code to analyze labeling functions, look at distributions, histograms, and define new metrics, look to `analysis.py`. 


Corresponding Authors: Bryce Johnson (bcjohnson7@wisc.edu), Huan Liang (hliang74@wisc.edu) 
