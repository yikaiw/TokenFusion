import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import skimage
import colorsys
import random
from skimage import io
import argparse
import datasets.transforms as T
import copy
import glob
import re
torch.set_grad_enabled(False)

CLASSES = ['bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser', 'night_stand', 'bookshelf', 'bathtub']

def make_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    scales = [800]

    if image_set == 'val' or image_set == 'train':
        return T.Compose([
            T.RandomResize([scales[-1]], max_size=scales[-1] * 1333 // 800),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def plot_gt(im, labels, bboxes_scaled, output_dir):
    tl = 3
    tf = max(tl-1, 1)
    tempimg = copy.deepcopy(im)
    color = [255,0,0]
    for label, (xmin, ymin, xmax, ymax) in zip(labels.tolist(), bboxes_scaled.tolist()):
        c1, c2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        cv2.rectangle(tempimg, c1, c2, color, tl, cv2.LINE_AA)
        text = f'{CLASSES[label]}'
        t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(tempimg, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(tempimg, text, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    fname = os.path.join(output_dir,'gt_img.png')
    cv2.imwrite(fname, tempimg)
    print(f"{fname} saved.")

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b
def draw_bbox_in_img(fname, bbox_scaled, score, color=[0,255,0]):
    tl = 3
    tf = max(tl-1,1) # font thickness
    # color = [0,255,0]
    im = cv2.imread(fname)
    for p, (xmin, ymin, xmax, ymax) in zip(score, bbox_scaled.tolist()):
        c1, c2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        cv2.rectangle(im, c1, c2, color, tl, cv2.LINE_AA)
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, text, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.imwrite(fname, im)

def plot_results(cv2_img, prob, boxes, output_dir):
    tl = 3 # thickness line
    tf = max(tl-1,1) # font thickness
    tempimg = copy.deepcopy(cv2_img)
    color = [0,0,255]
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        c1, c2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        cv2.rectangle(tempimg, c1, c2, color, tl, cv2.LINE_AA)
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(tempimg, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(tempimg, text, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    fname = os.path.join(output_dir,'pred_img.png')
    cv2.imwrite(fname, tempimg)
    print(f"{fname} saved.")
    
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def save_pred_fig(output_dir, output_dic, keep):
    # im = Image.open(os.path.join(output_dir, "img.png"))
    im = cv2.imread(os.path.join(output_dir, "img.png"))
    h, w = im.shape[:2]
    bboxes_scaled = rescale_bboxes(output_dic['pred_boxes2d'][0, keep].cpu(), (w,h))
    prob = output_dic['pred_logits2d'].softmax(-1)[0, :, :-1]
    scores = prob[keep]
    plot_results(im, scores, bboxes_scaled, output_dir)

def save_gt_fig(output_dir, gt_anno):
    im = cv2.imread(os.path.join(output_dir, "img.png"))
    h, w = im.shape[:2]
    bboxes_scaled = rescale_bboxes(gt_anno['boxes'], (w,h))
    labels = gt_anno['labels']
    plot_gt(im, labels, bboxes_scaled, output_dir)

def get_one_query_meanattn(vis_attn,h_featmap,w_featmap):
    mean_attentions = vis_attn.mean(0).reshape(h_featmap, w_featmap)
    mean_attentions = nn.functional.interpolate(mean_attentions.unsqueeze(0).unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
    return mean_attentions

def get_one_query_attn(vis_attn, h_featmap, w_featmap, nh):
    attentions = vis_attn.reshape(nh, h_featmap, w_featmap)
    # attentions = vis_attn.sum(0).reshape(h_featmap, w_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
    return attentions
