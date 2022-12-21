## Conda environment
Requires OpenCV

`conda install -c conda-forge opencv`

## Code files
### generate_training_set_list.py:
Generates a list of potential training/validation frames, following rules such as containing a single cow in the frame

### curate_training_set.py:
User Interface for curating the training set. Renders each frame from a random selection of potential training images, and separates them into folders according to user judgment on whether the bounding box and class label are correct

### generate_superimposed_and_cropped_training.py:
Generates training/validation image datasets based on the curated list of potential training frames built using generate_training_set_list.py and curate_random_valid_frames.py

### generate_manual_test_set.py:
Generates test set based on manual annotations. JSON files for annotations were generated using VGG Image Annotator (https://www.robots.ox.ac.uk/~vgg/software/via/)

### generate_unlabeled_set.py:
Generates unlabeled set based on similar rules as the training set was generated, but without using labels or curating
