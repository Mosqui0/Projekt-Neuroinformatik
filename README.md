# Projekt-Neuroinformatik

## Proejct structure
###Related_Work:
Contains some papers as PDF used in our documentation as related work.

###Code:
The jupyter notebook model.ipynb contains the build pipeline of our proposed model. Please note that in order to use this pipeline,
the necessary data have to be located in parent directory from root of this repository.

Evaluation and training methods are located in the respective subdirectories.
The code for hand segmentation and bounding box prediction is located in the Training_Evaluation_Segmentation.ipynb file in the HandSeg directory.
The .py files contain helper-functions for creating training/validation/test splits of the HandSeg dataset as well as the implementation of the segmentation model.

Add PixelWiseRegression information
