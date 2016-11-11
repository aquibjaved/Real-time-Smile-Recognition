# smiledetector
Smile Detection using OpenCV and scikit-learn

--- INSTALLATION INSTRUCTIONS ---

1. Download ZIP archive of project.
2. Make sure Python 2.7 is installed on your machine.
3. Install Anaconda: https://www.continuum.io/downloads
4. Install OpenCV: http://opencv.org/downloads.html
  - set environment Path to contain following: (path to opencv folder)\build\python\2.7\x64(if using 64 bit)
4. Open the Anaconda console, and install the following packages using "conda (package name)"
  - scikit-learn
  - matplotlib
5. You should now have all the necessary packages for the program to work.
6. You can use the provided training set, or if you would like to create your own test results file:
  - open anaconda console
  - navigate to project directory
  - use the commmand "IPython notebook" to open the IPython notebook.  
  - select get_training_data.ipynb and run code to train on dataset.
  - you should now have a saved results xml file with your training results.
7. open up your compilation tool of choice (in my demo I used powershell, but any other console/IDE
works just fine)
8. use the command "python smiledetector.py" and the program should run.

--- HOW TO RUN/COMPILE: ---

use the command "python smiledetector.py" in the project's directory

--- ALGORITHM ---

- supervised learning: train a classifier on faces dataset
- target values: smiling/ non-smiling
- load faces dataset training results in json format
- visualize training set data/results
- use a support vector classifier to learn from 400 smiles/non-smiles
- support vector classifier uses linear kernel to find
patterns
- use 5 fold cross validation to train classifier
- original sample randomly partitioned into k equal sized subsamples
- of k subsamples, a single subsample is retained as the validation data
for the testing model, where the remaining k - 1 samples are used as
training data.
- evaluate cross validation
- (classifier, X)
- around 0.8 mean score
- confusion matrix shows errors classifier made during training
- predict classification of a new image
- import test image
- detect face with haar cascade classifier
- cascade of features:
- edge features
- line features
- four-rectangle features
- needs lots of positive images (images of faces) and negative images
(images without faces) to train the classifier.
- extract features and compare with original face region
- transform extracted image (200x200 pixels) to 64x64 pixels so that it
matches Olivetti dataset face format
- detect face function uses frame from camera as input, convert to
grayscale, apply haar cascade, return gray images
- extract features from previously detected face, using args to stretch
image vertically and horizontally
(I found that 0.11 and 0.17 work)
- use wrapper call to predict function of svc to predict if extracted
faces are smiling
# Smile-Recognition-
