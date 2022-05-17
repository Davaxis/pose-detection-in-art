import json
import sys
from cv2 import GaussianBlur, merge
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import extract

def main():
  for folder in os.listdir('results-raw'):
    if os.path.isdir('results-raw/' + folder) and any(f.endswith(".jpg") or f.endswith(".png") for f in os.listdir('results-raw/' + folder)):
      print("merging {}".format(folder))
      compute_merged('results-raw/' + folder)

def remove_computed(path):
  for filename in os.listdir(path):
    if filename.endswith(".png") or filename.endswith(".jpg"):
      os.remove('{}/{}'.format(path, filename))

def compute_merged(path):
  raw_images = []
  extracted_images = []
  for filename in os.listdir(path):
    if not (filename.endswith('.jpg') or filename.endswith('.png')):
        continue
    
    raw_images.append(cv2.imread('{}/{}'.format(path, filename)))
    # info = json.load(open('results-raw/{}.json'.format(filename.replace('result', 'info'))))
    # counter.append(info['amount'])

  for image in raw_images:
    extracted_images.extend(extract.main(image))

  maxW = max(map(lambda item: np.shape(item)[0], extracted_images))
  maxH = max(map(lambda item: np.shape(item)[1], extracted_images))

  merged = np.zeros((maxW, maxH, 3))
  
  del_index = []
  for index, img in enumerate(extracted_images):
    if img.shape[0] < maxW/4 or img.shape[1] < maxH/4:
      del_index.append(index)
    else:
      padded = cv2.resize(img, dsize=(np.shape(merged)[1], np.shape(merged)[0]))
      merged = np.add(padded, merged)

  for index in sorted(del_index, reverse=True):
    del extracted_images[index]

  # grayscale
  merged = np.mean(merged, axis=2)

  # normalization
  merged_norm = np.zeros(np.shape(merged))
  merged = cv2.normalize(merged, merged_norm, 0, 255, cv2.NORM_MINMAX)
  merged_norm = np.round(merged_norm, 0).astype(np.uint8)

  thresh = np.max(merged_norm) - 30
  mask = np.zeros_like(merged_norm)
  mask[merged_norm >= thresh] = 255

  # get most similar image
  best_image = None
  best_score = 0
  for img in extracted_images:
    img = cv2.resize(img, dsize=(np.shape(mask)[1], np.shape(mask)[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    score = np.sum(mask * img)
    if score > best_score:
      best_score = score
      best_image = img

  # add gaussian blur
  gauss_size = int(min(np.shape(merged)[:2]) * 0.05)
  if gauss_size % 2 == 0:
    gauss_size += 1
  img = cv2.GaussianBlur(img, (gauss_size, gauss_size), 0)

  # closing
  merged = cv2.dilate(merged, np.ones((15, 15), np.uint8))
  merged = cv2.erode(merged, np.ones((15, 15), np.uint8), iterations=2)

  plt.imshow(best_image)
  plt.savefig('{}/best.png'.format(path))
  plt.imshow(merged_norm)
  plt.savefig('{}/merged.png'.format(path))


if __name__ == '__main__':
  main()

