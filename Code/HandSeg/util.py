import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt

# input is a (480, 640, 2) np.array -> interpret the softmax values in third dimension
# returns a (480, 640) segmented image
def interpretPrediction(pred):
    # create np.array to save the interpretation
    interpretation = np.zeros((480, 640))
    height, width, _ = pred.shape
    for i in range(height):
        for j in range(width):
            # create a list of the probabilites from each class
            value_list = [pred[i,j,0], pred[i,j,1]]
            # find out, which class has been predicted
            largest = max(value_list)
            index_of_largest = value_list.index(largest)
            # store new value in the interpretation array
            interpretation[i,j] = index_of_largest
    return interpretation

# draw a bounding box on mask based on class values in img
# returns bounding box values as (x, y, height, width)
def drawBoxes(mask, img):
    # Iterate all colors in mask
    for color in np.unique(mask):

        # Color 0 is assumed to be background or artifacts
        if color == 0:
          continue

        # Determine bounding rectangle w.r.t. all pixels of the mask with
        # the current color
        x, y, w, h = cv2.boundingRect(np.uint8(mask == color))
        # Draw bounding rectangle to color image
        out = cv2.rectangle(img.copy(), (x, y), (x+w, y+h), (255, int(color), 0), 2)

        # Show image with bounding box
        plt.imshow(out); plt.title('img_' + str(color)); plt.show()
    return (x,y,w,h)


# preprocessing
def preprocess(img):
    return (img / img.max()) * 255
	
def preprocess2(img):
    img = img - img.min()
    #img = img.max() - img
    img = tf.where(img == 4, 0, img).numpy()
    return img