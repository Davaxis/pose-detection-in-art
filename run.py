import json
from lib.config import cfg, update_config
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.network import im_transform
from lib.network.rtpose_vgg import get_model
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
import pylab as plt
import numpy as np
import matplotlib
import argparse
import scipy
import time
import math
import cv2
import os
import re
import sys
import merge
sys.path.append('.')
# from scipy.ndimage.morphology import generate_binary_structure
# from scipy.ndimage.filters import gaussian_filter, maximum_filter


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

print('Pose estimation...')
for folder in sorted(os.listdir('pictures')):
    if os.path.isdir('pictures/' + folder):
        # merge.remove_computed('results/' + folder)
        # merge.remove_computed('results-raw/' + folder)
        print(folder)
        for filename in os.listdir('pictures/' + folder):
            if not (filename.endswith('.jpg') or filename.endswith('.png')):
                continue

            test_image = 'pictures/{}/{}'.format(folder, filename)
            oriImg = cv2.imread(test_image)  # B,G,R order
            shape_dst = np.min(oriImg.shape[0:2])

            # Get results of original image

            with torch.no_grad():
                paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')

            humans = paf_to_pose_cpp(heatmap, paf, cfg)
            info = {'amount': len(humans)}
            out = draw_humans(oriImg, humans)
            outRaw = draw_humans(np.zeros_like(oriImg), humans)

            # crete folder if not exist
            if not os.path.exists('results/' + folder):
                os.makedirs('results/' + folder)
            if not os.path.exists('results-raw/' + folder):
                os.makedirs('results-raw/' + folder)
            
            # save image
            cv2.imwrite('results/' + folder + '/' + filename, out)
            cv2.imwrite('results-raw/' + folder + '/' + filename, outRaw)
            # with open('results-info/{}/info-{}.json'.format(folder, filename[:-4]), 'w') as outfile:
            #     json.dump(info, outfile)


# merge.main()