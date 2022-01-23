import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

from __future__ import division
from __future__ import print_function

import os
import cv2
import json, yaml
import numpy as np
from PIL import Image
from collections import OrderedDict
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi

from pycocotools.coco import COCO



def __get_annotation__(mask, image_id, box = None, class_id):
  
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  segmentation = []
  for contour in contours:
      # Valid polygons have >= 6 coordinates (3 points)
      if contour.size >= 6:
          segmentation.append(contour.flatten().tolist())
  RLEs = cocomask.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
  RLE = cocomask.merge(RLEs)
  # RLE = cocomask.encode(np.asfortranarray(mask))
  area = cocomask.area(RLE)
  [x, y, w, h] = cv2.boundingRect(mask)

  labelMask = np.expand_dims(mask, axis=2)
  labelMask = labelMask.astype('uint8')
  labelMask = np.asfortranarray(labelMask)
  Rs = cocomask.encode(labelMask)
  assert len(Rs) == 1
  Rs = Rs[0]

  return Rs, [x, y, w, h], area #segmentation durch Rs
  
    
  
