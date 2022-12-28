import numpy as np
import cv2
import glob
import json

gt_files = glob.glob('./trainset/*.json')
for file in gt_files:
    name = file[11:-5]
    #print(name)
    with open(file, 'r', encoding='utf8', errors='ignore') as j:
        gt = json.load(j)
    polygon = np.array(gt['polygon'])
    polygon /= 4.0
    bg = np.zeros((56, 56), dtype=np.uint8)

    mask = cv2.fillPoly(bg, np.int32([polygon]), 255)

    mask = mask > 128
    mask = np.asarray(mask, dtype=np.double)
    gx, gy = np.gradient(mask)

    boundary = gy * gy + gx * gx

    boundary[boundary != 0.0] = 255.0

    boundary = np.asarray(boundary, dtype=np.uint8)

    cv2.imwrite('./trainset/' + str(name) + 'b.png', boundary)