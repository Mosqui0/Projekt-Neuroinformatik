import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass # use to compute center of mass (row, col)
import cv2
import os, random, time, struct, re
import math

from PixelwiseRegression.utils import load_bin, load_model, draw_skeleton, select_gpus, center_crop, \
    generate_com_filter, floodFillDepth, generate_heatmap, random_rotated, generate_kernel
from PixelwiseRegression.model import PixelwiseRegression
from PixelwiseRegression.classifyASL import *
from PixelwiseRegression.test_samples import draw_skeleton

def loadModel(imgSource):
    #Model Parameter
    joint_number=16
    
    model_parameters = {
        "stage" : 2, 
        "label_size" : 64,
        "features" : 128, 
        "level" : 4,
        "norm_method" : 'instance',
        "heatmap_method" : 'softmax',
    }
    
    
    #Mapping of the index of the value  to the finger
    Index = [0, 4, 5, 6]
    Mid = [0, 7, 8, 9]
    Ring = [0, 10, 11, 12]
    Small = [0, 13, 14, 15]
    Thumb = [0, 1, 2, 3]
    config = [Thumb, Index, Mid, Ring, Small]
    
    select_gpus("0")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = PixelwiseRegression(joint_number, **model_parameters)
    model_name = "ICVL_default_46.pt"
    #model_name = "ICVL_default_final"
    load_model(model, os.path.join('PixelwiseRegression/Model', model_name), eval_mode=True)
    model = model.to(device)


    #Load image
    #img = cv2.imread("./Data/ICVL/image_0001.png")
    #imgSource = plt.imread("./PixelwiseRegression/Data/ICVL/image_0001.png")

    img = imgSource*65535
    cube_size=125
    fx=241.42
    fy=241.42
    image_size=128
    label_size=64
    #mean without extreme high values like background or nan
    mean = np.mean(img[img < 10000])
    #TODO Is this mean calculation valid over more than this example?
    #print("Mean without backgruond", mean)
    #calc com (source: datasets 209)
    if math.isnan(np.mean(img[img < 0])):
        mean = np.mean(img[img < mean])
    else:
        print("WARNING; Value less than zero. TODO")

    #mean = 360.554
    _com = center_of_mass(img > 0)
    _com = np.array([_com[1], _com[0], mean])
    #print("_com ", _com)
    image = img.copy()
    #print("Image shape before crop", image.shape)
    com = _com.copy()

    # crop the image
    du = cube_size / com[2] * fx
    dv = cube_size / com[2] * fy
    box_size = int(du + dv)
    #print("BoxSize int: ", box_size)
    box_size = max(box_size, 2)
    #print("BoxSize maxxed: ", box_size)
    crop_img = center_crop(image, (com[1], com[0]), box_size)
    crop_img = crop_img * np.logical_and(crop_img > com[2] - cube_size, crop_img < com[2] + cube_size)
    #print("Image shape after crop", crop_img.shape)
    # norm the image and uvd to COM
    crop_img[crop_img > 0] -= com[2] # center the depth image to COM

    com[0] = int(com[0])
    com[1] = int(com[1])
    box_size = crop_img.shape[0] # update box_size
    print("Updated box size, ", box_size)
    # resize the image and uvd
    try:
        img_resize = cv2.resize(crop_img, (image_size, image_size))
        #print("imgresizedshape: ", img_resize.shape)
    except:
        # probably because size is zero
        print("resize error")
        raise ValueError("Resize error")

    # Generate label_image and mask
    label_image = cv2.resize(img_resize, (label_size, label_size))
    #print("label_image: ", label_image.shape)
    is_hand = label_image != 0
    mask = is_hand.astype(float)
    

    
    normalized_img = img_resize / cube_size
    normalized_label_img = label_image / cube_size
    
    
    # Convert to torch format
    #print(normalized_img.shape)
    normalized_img = torch.from_numpy(normalized_img).float().unsqueeze(0)
    #print(normalized_img.shape)
    normalized_label_img = torch.from_numpy(normalized_label_img).float().unsqueeze(0)
    mask = torch.from_numpy(mask).float().unsqueeze(0)
    box_size = torch.tensor(box_size).float()
    cube_size = torch.tensor(cube_size).float()
    com = torch.from_numpy(com).float()
    #print("img: {}, label_img: {}, mask: {}".format(normalized_img.shape, normalized_label_img.shape, mask.shape))
    #return normalized_img, normalized_label_img, mask, box_size, cube_size, com
    
    #load model
    img = torch.reshape(normalized_img, (1,1,128, 128))
    label_img = torch.reshape(normalized_label_img, (1,1,64, 64))
    mask = torch.reshape(mask, (1,1,64, 64))
    
    #Expected object of device type cuda, transform from cpu
    img = img.to(device, non_blocking=True)
    label_img = label_img.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
        
    results = model(img, label_img, mask)

    _heatmaps, _depthmaps, _uvd = results[-1]
    _uvd = _uvd.detach().cpu().numpy()
    img = img.cpu().numpy()
    printPrediction(_uvd)
    drawSkeleton(img, _uvd)

''' transforms uvd to xyz format'''
def uvdtoXYZ(_uvd):
    #Denormalize the Uv coordinates
    joints44 = _uvd[0,:,:2] * (512 - 1) + np.array([512 // 2, 512 // 2])
    #convert to xy or int? coordinates
    _joint44 = [(int(joints44[i][0]), int(joints44[i][1])) for i in range(joints44.shape[0])]
    return _joint44
    
''' Predicts the ASL sign '''
def printPrediction(_uvd):
    _joint44 = uvdtoXYZ(_uvd)
    palmRadius = calcPalm(_joint44)
    fingers = fingerStretched(_joint44, palmRadius)
    prediction = classifyHandSign(fingers)
    print("Prediction: ", prediction)
    return 
    
def drawSkeleton(img, _uvd):
    #Draw Skeleton
    #print(xyz_pre)
    skeleton_pre, xyz_pre = draw_skeleton(img[0,0], _uvd[0,:,:2], skeleton_mode=0)
    skeleton_pre = np.clip(skeleton_pre, 0, 1) 
    
    #print skeleton of prediction
    cv2.imshow("predict", skeleton_pre)
    cv2.moveWindow("predict", 50,50)

    print("Waiting for key to Press: s for saving the images, q for quitting (Press command with images as scene).")
    ch = cv2.waitKey(0)
    if ch == ord('s'):
        cv2.destroyAllWindows()
        plt.imsave(os.path.join("skeleton", args.dataset, "predict", "{}.jpg".format(index)), skeleton_pre)
        index += 1
    elif ch == ord('q'):
        ...



