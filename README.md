# Projekt-Neuroinformatik

## Project structure
### Related_Work:
Contains some papers as PDF used in our documentation as related work.

### Code:
The jupyter notebook model.ipynb contains the build pipeline of our proposed model. Please note that in order to use this pipeline,
the necessary data have to be located in parent directory from root of this repository.

#Evaluation and training methods are located in the respective subdirectories.
The code for hand segmentation and bounding box prediction is located in the Training_Evaluation_Segmentation.ipynb file in the HandSeg directory.
The .py files contain helper-functions for creating training/validation/test splits of the HandSeg dataset as well as the implementation of the segmentation model.

#Pixelwise Regression: 
For training, add the ICVL Dataset in the respective Folder: '/Code/PixelwiseRegression/Data/ICVL. To start the training, execute: 'python train.py --dataset ICVL'. You can add more flags more introduced in the code. An example is the flag: '--batch-size' if there are problems with processing power. Most of the general helper functions are probided in the 'utils.py' file. To test random images of the testing folder of the dataset, execute the 'test_samples.py'. There 
The evaluation got calculated in the 'getmeanDev.py'.
In the folder 'Code/PixelwiseRegression/Model', there are all the checkpoints of the training iterations.
The File 'Code/PixelwiseRegression/ClassifyASL.py' includes all functions to assign the joint locations to a predicted sign.
