from cv2 import GaussianBlur
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

  merged /= len(images)
  merged = np.mean(merged, axis=2)

  # add gaussian blut
  gauss_size = int(min(np.shape(merged)[:2]) * 0.08)
  if gauss_size % 2 == 0:
    gauss_size += 1
  img = cv2.GaussianBlur(img, (gauss_size, gauss_size), 150)

  # normalization
  coeficient = 255 / np.max(merged)
  merged *= coeficient

  # closing
  merged = np.round(merged, 0).astype(np.uint8)
  merged = cv2.dilate(merged, np.ones((25, 25), np.uint8))
  merged = cv2.erode(merged, np.ones((25, 25), np.uint8), iterations=2)


  plt.imshow(merged)
  plt.show()



if __name__ == '__main__':
  main()

