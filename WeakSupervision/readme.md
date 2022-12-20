## Weak Supervision 

To create the conda environment to re-create the results of the weak supervision predictions run

`conda create --name cattle`

`conda activate cattle`

`conda install -c conda-forge snorkel=0.9.9`

`conda install numpy matplotlib pandas`

`conda install -c anaconda scikit-learn` 

`conda install -c anaconda seaborn` 

and now the environment is created to run the weak supervision code. 

### Main Pipeline 

To run the main pipeline look to `pipeline_cattle.py`. Many functions are defined which run different parts of the paper.
You must uncomment different parts to run different analysis in the paper. I would recommend contacting Bryce Johnson (email listed below) if you
are consider running this code. 


### Defining a labeling function 

All labeling functions are defined in `lambda_functions.py`. Here you must use the decorator @labeling_function() before you write your own labeling functions. 

### Analysis of labeling functions 

To run the code to analyze labeling functions, look at distributions, histograms, and define new metrics, look to `analysis.py`. 


Corresponding Authors: Bryce Johnson (bcjohnson7@wisc.edu), Huan Liang (hliang74@wisc.edu) 
