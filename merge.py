import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def main():
  images = []

  for filename in os.listdir('results-raw'):
    if not (filename.endswith('.jpg') or filename.endswith('.png')):
        continue
    
    images.append(cv2.imread('results-raw/{}'.format(filename)))

  maxW = max(map(lambda item: np.shape(item)[0], images))
  maxH = max(map(lambda item: np.shape(item)[1], images))

  merged = np.zeros((maxW, maxH, 3))
  
  for img in images:
    padded = cv2.resize(img, dsize=(np.shape(merged)[1], np.shape(merged)[0]))
    merged = np.add(padded, merged)

  merged = np.round(merged / len(images), 0).astype(np.int32)

  plt.imshow(merged)
  plt.show()



if __name__ == '__main__':
  main()

