import os
import re
import sys
sys.path.append('.')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config

import json

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)


model = get_model('vgg19')     
model.load_state_dict(torch.load(args.weight))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

for filename in os.listdir('pictures'):
    if not (filename.endswith('.jpg') or filename.endswith('.png')):
        continue

    test_image = './pictures/{}'.format(filename)
    oriImg = cv2.imread(test_image) # B,G,R order
    shape_dst = np.min(oriImg.shape[0:2])

    # Get results of original image

    with torch.no_grad():
        paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')
            
    humans = paf_to_pose_cpp(heatmap, paf, cfg)

    info = {'amount': len(humans)}
            
    out = draw_humans(oriImg, humans)
    outRaw = draw_humans(np.zeros_like(oriImg), humans)
    cv2.imwrite('results/result-{}'.format(filename), out)   
    cv2.imwrite('results-raw/result-{}'.format(filename), outRaw)
    with open('results-raw/info-{}.json'.format(filename), 'w') as outfile:
        json.dump(info, outfile)

