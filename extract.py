from audioop import mul
import numpy as np
import cv2
from matplotlib import pyplot as plt



def main(source):
    image = source.copy()
    # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
    th = 20 # defines the value below which a pixel is considered "black"
    black_pixels = np.where(
        (image[:, :, 0] < th) & 
        (image[:, :, 1] < th) & 
        (image[:, :, 2] < th)
    )

    # set those pixels to white
    image[black_pixels] = [255, 255, 255]

    not_white_pixels = np.where(
        (image[:, :, 0] != 255) | 
        (image[:, :, 1] != 255) | 
        (image[:, :, 2] != 255)
    )

    image[not_white_pixels] = [0, 0, 0]
    image = ~image



    output = cv2.connectedComponentsWithStats(
        image[:,:,0], 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output


    components = []
    # loop over the number of unique connected component labels
    for i in range(0, numLabels):
        # skip background
        if i == 0:
            continue 
        
        # get stats
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        components.append(source[y:y+h, x:x+w])
    
    return components