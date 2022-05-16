import json
import sys
from cv2 import GaussianBlur, merge
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import extract

def main(save_to):
  raw_images = []
  extracted_images = []
  # counter = []

  for filename in os.listdir('results-raw'):
    if not (filename.endswith('.jpg') or filename.endswith('.png')):
        continue
    
    raw_images.append(cv2.imread('results-raw/{}'.format(filename)))
    # info = json.load(open('results-raw/{}.json'.format(filename.replace('result', 'info'))))
    # counter.append(info['amount'])

  for image in raw_images:
    extracted_images.extend(extract.main(image))

  maxW = max(map(lambda item: np.shape(item)[0], extracted_images))
  maxH = max(map(lambda item: np.shape(item)[1], extracted_images))

  merged = np.zeros((maxW, maxH, 3))
  
  for img in extracted_images:
    padded = cv2.resize(img, dsize=(np.shape(merged)[1], np.shape(merged)[0]))
    merged = np.add(padded, merged)

  # add gaussian blur
  gauss_size = int(min(np.shape(merged)[:2]) * 0.08)
  if gauss_size % 2 == 0:
    gauss_size += 1
  img = cv2.GaussianBlur(img, (gauss_size, gauss_size), 150)

  # normalization
  merged_norm = np.zeros(np.shape(merged))
  merged = cv2.normalize(merged, merged_norm, 0, 255, cv2.NORM_MINMAX)

  # closing
  merged = np.round(merged_norm, 0).astype(np.uint8)
  merged = cv2.dilate(merged, np.ones((25, 25), np.uint8))
  merged = cv2.erode(merged, np.ones((25, 25), np.uint8), iterations=2)

  plt.imshow(merged)
  plt.savefig('merged/{}.png'.format(save_to))


if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Usage: python3 merge.py <save_to>')
    print('<save_to> is filename of the picture that the script produces without extension')
    sys.exit(1)
  main(sys.argv[1])

