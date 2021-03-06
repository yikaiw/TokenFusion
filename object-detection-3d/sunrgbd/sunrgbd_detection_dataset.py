# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for 3D object detection on SUN RGB-D (with additional support for ImVoteNet).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Modified by: Xinlei Chen
Date: 2020

"""
import os
import sys
import numpy as np
import tqdm
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset
import scipy.io as sio # to load .mat files for depth points
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import sunrgbd_utils
from model_util_sunrgbd import SunrgbdDatasetConfig

DC = SunrgbdDatasetConfig()  # dataset specific config
MAX_NUM_OBJ = 64  # maximum number of objects allowed per scene
MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1

NUM_CLS = 10  # sunrgbd number of classes
MAX_NUM_2D_DET = 100  # maximum number of 2d boxes per image
MAX_HEIGHT = 530
MAX_WIDTH = 730
MAX_NUM_PIXEL = MAX_HEIGHT * MAX_WIDTH  # maximum number of pixels per image
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class SunrgbdDetectionVotesDataset(Dataset):
    def __init__(self, split_set='train', num_points=20000, use_color=False, use_height=False, 
                 use_img=True, use_v1=True, augment=False, scan_idx_list=None, data_root=None):

        assert (num_points <= 50000)
        self.use_v1 = use_v1
        self.train = split_set == 'train'
        if use_v1:
            self.data_path = os.path.join(data_root, f'sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_{split_set}')
        else:
            self.data_path = os.path.join(data_root, f'sunrgbd/sunrgbd_pc_bbox_votes_50k_v2_{split_set}')

        self.raw_data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval')
        self.scan_names = sorted(list(set([os.path.basename(x)[0:6] for x in os.listdir(self.data_path)])))
        if False and scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        self.use_img = use_img
        # Total feature dimensions: geometric(5)+semantic(NUM_CLS)+texture(3) 
        self.image_feature_dim = NUM_CLS + 8

        #pickle_filename = os.path.join(self.data_path, 'all_images.pkl')
        #with open(pickle_filename, 'rb') as f:
        #    self.images_list = pickle.load(f)
        #print(f"{pickle_filename} loaded successfully !!!")

        if use_img:
            pickle_filename = os.path.join(self.data_path, 'all_obbs2d_modified_nearest_has_empty.pkl')
            with open(pickle_filename, 'rb') as f:
                self.bboxes2d_list = pickle.load(f)
            print(f"{pickle_filename} loaded successfully !!!")

        pickle_filename = os.path.join(self.data_path, 'all_obbs_modified_nearest_has_empty.pkl')
        with open(pickle_filename, 'rb') as f:
            self.bboxes_list = pickle.load(f)
        print(f"{pickle_filename} loaded successfully !!!")

        pickle_filename = os.path.join(self.data_path, 'all_pc_modified_nearest_has_empty.pkl')
        with open(pickle_filename, 'rb') as f:
            self.point_cloud_list = pickle.load(f)
        print(f"{pickle_filename} loaded successfully !!!")

        pickle_filename = os.path.join(self.data_path, 'all_point_labels_nearest_has_empty.pkl')
        with open(pickle_filename, 'rb') as f:
            self.point_labels_list = pickle.load(f)
        print(f"{pickle_filename} loaded successfully !!!")

    def __len__(self):
        return len(self.point_cloud_list)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            point_obj_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
            point_instance_label: (N,) with int values in -1,...,num_box, indicating which object the point belongs to, -1 means a backgound point.
            scan_idx: int scan index in scan_names list
            max_gt_bboxes: unused
        """
        scan_name = self.scan_names[idx]
        point_cloud = self.point_cloud_list[idx]  # Nx6
        bboxes = self.bboxes_list[idx]  # K,8
        point_obj_mask = self.point_labels_list[idx][:, 0]
        point_instance_label = self.point_labels_list[idx][:, -1]
        # print(scan_name)
        if self.use_img:
            bboxes2d = self.bboxes2d_list[idx]
            # Read camera parameters
            calib_lines = [line for line in open(os.path.join(self.raw_data_path, 'calib', scan_name + '.txt')).readlines()]
            calib_Rtilt = np.reshape(np.array([float(x) for x in calib_lines[0].rstrip().split(' ')]), (3, 3), 'F')
            calib_K = np.reshape(np.array([float(x) for x in calib_lines[1].rstrip().split(' ')]), (3, 3), 'F')
            # Read image
            full_img = sunrgbd_utils.load_image(os.path.join(self.raw_data_path, 'image', scan_name + '.jpg'))
            fx, fy = MAX_WIDTH / full_img.shape[1], MAX_HEIGHT / full_img.shape[0]
            full_img = cv2.resize(full_img, None, fx=fx, fy=fy)
            full_img_height, full_img_width = full_img.shape[0], full_img.shape[1]

            # ------------------------------- 2D IMAGE VOTES ------------------------------
            ##cls_id_list = self.cls_id_map[scan_name]
            #cls_score_list = self.cls_score_map[scan_name]

            # Semantic cues: one-hot vector for class scores
            #cls_score_feats = np.zeros((1 + MAX_NUM_2D_DET,NUM_CLS), dtype=np.float32)
            # First row is dumpy feature
            #len_obj = len(cls_id_list)
            #if len_obj:
            #    ind_obj = np.arange(1, len_obj + 1)
            #    ind_cls = np.array(cls_id_list)
            #    cls_score_feats[ind_obj, ind_cls] = np.array(cls_score_list)

            # Texture cues: normalized RGB values
            # full_img = (full_img - 128.) / 255.
            # Serialize data to 1D and save image size so that we can recover the original location in the image
            # full_img_1d = np.zeros((MAX_NUM_PIXEL * 3), dtype=np.float32)
            # full_img_1d[:full_img_height * full_img_width * 3] = full_img.flatten()

        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]
        else:
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        scale_ratio = 1.
        if self.augment:
            flip_flag = np.random.random() > 0.5
            if flip_flag:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                bboxes[:, 0] = -1 * bboxes[:, 0]
                bboxes[:, 6] = np.pi - bboxes[:, 6]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = sunrgbd_utils.rotz(rot_angle)

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 6] -= rot_angle

            if self.use_img:
                R_inverse = np.copy(np.transpose(rot_mat))
                if flip_flag:
                    R_inverse[0, :] *= -1
                # Update Rtilt according to the augmentation
                # R_inverse (3x3) * point (3x1) transforms an augmented depth point
                # to original point in upright_depth coordinates
                calib_Rtilt = np.dot(np.transpose(R_inverse), calib_Rtilt)

            # Augment RGB color
            if self.use_color:
                rgb_color = point_cloud[:, 3:6] + MEAN_COLOR_RGB
                rgb_color *= (1 + 0.4 * np.random.random(3) - 0.2) # brightness change for each channel
                rgb_color += (0.1 * np.random.random(3) - 0.05) # color shift for each channel
                rgb_color += np.expand_dims((0.05 * np.random.random(point_cloud.shape[0]) - 0.025), -1) # jittering on each pixel
                rgb_color = np.clip(rgb_color, 0, 1)
                # randomly drop out 30% of the points' colors
                rgb_color *= np.expand_dims(np.random.random(point_cloud.shape[0]) > 0.3, -1)
                point_cloud[:, 3:6] = rgb_color - MEAN_COLOR_RGB

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            if self.use_img:
                calib_Rtilt = np.dot(np.array([[scale_ratio, 0, 0], [0, scale_ratio, 0], [0, 0, scale_ratio]]), calib_Rtilt)
            scale_ratio_expand = np.expand_dims(np.tile(scale_ratio, 3), 0)
            point_cloud[:, 0:3] *= scale_ratio_expand
            bboxes[:, 0:3] *= scale_ratio_expand
            bboxes[:, 3:6] *= scale_ratio_expand
            if self.use_height:
                point_cloud[:, -1] *= scale_ratio

        # ------------------------------- LABELS ------------------------------
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        label_mask = np.zeros((MAX_NUM_OBJ))
        label_mask[0:bboxes.shape[0]] = 1
        max_bboxes = np.zeros((MAX_NUM_OBJ, 8))
        max_bboxes[0:bboxes.shape[0], :] = bboxes

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            box3d_center = bbox[0:3]
            angle_class, angle_residual = DC.angle2class(bbox[6])
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here 
            box3d_size = bbox[3:6] * 2
            size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class])
            box3d_centers[i, :] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i, :] = box3d_size

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes[:, 0:3] += 1000.0
        if self.use_img:
            target_bboxes2d = np.zeros((MAX_NUM_OBJ, 4))
        size_gts = np.zeros((MAX_NUM_OBJ, 3))
        calib = sunrgbd_utils.SUNRGBD_Calibration(calib_Rtilt, calib_K) if self.use_img else None
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            corners_3d, corners_2d = sunrgbd_utils.compute_boxes(bbox[0:3], bbox[3:6], bbox[6], calib)
            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            target_bbox = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, xmax - xmin, ymax - ymin, zmax - zmin])
            target_bboxes[i, :] = target_bbox
            size_gts[i, :] = target_bbox[3:6]

            if self.use_img:
                xmin, ymin, xmax, ymax = bboxes2d[i]
                target_bbox2d = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin])
                target_bboxes2d[i, :] = target_bbox2d

        if self.use_img:
            target_bboxes2d = box_cxcywh_to_xyxy(target_bboxes2d)
            target_bboxes2d[:, 0::2] = np.clip(target_bboxes2d[:, 0::2], 0, full_img_width) / full_img_width
            target_bboxes2d[:, 1::2] = np.clip(target_bboxes2d[:, 1::2], 0, full_img_height) / full_img_height
            target_bboxes2d = box_xyxy_to_cxcywh(target_bboxes2d)

        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_obj_mask = point_obj_mask[choices]
        point_instance_label = point_instance_label[choices]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:, 0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        ret_dict['size_gts'] = size_gts.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:, -1]  # from 0 to 9
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['point_obj_mask'] = point_obj_mask.astype(np.int64)
        ret_dict['point_instance_label'] = point_instance_label.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['max_gt_bboxes'] = max_bboxes

        if self.use_img:
            ret_dict['scale'] = np.array(scale_ratio).astype(np.float32)
            ret_dict['calib_Rtilt'] = calib_Rtilt.astype(np.float32)
            ret_dict['calib_K'] = calib_K.astype(np.float32)
            # full_img.shape: [530, 730, 3] -> [3, 530, 730]
            ret_dict['full_img'] = np.transpose(full_img.astype(np.float32), (2, 0, 1))
            mean = np.array(MEAN, dtype=np.float32)[:, np.newaxis, np.newaxis]
            std = np.array(STD, dtype=np.float32)[:, np.newaxis, np.newaxis]
            ret_dict['full_img'] = (ret_dict['full_img'] / 255. - mean) / std
            ret_dict['full_img_width'] = np.array(full_img_width).astype(np.int64)
            ret_dict['full_img_height'] = np.array(full_img_height).astype(np.int64)
            ret_dict['target_box2d'] = target_bboxes2d.astype(np.float32)

        return ret_dict


def box_cxcywh_to_xyxy(x):
    """Convert for numpy
    """
    x_c, y_c, w, h = x.transpose(1, 0)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.stack(b, axis=-1)


def box_xyxy_to_cxcywh(x):
    """Convert for numpy
    """
    x0, y0, x1, y1 = x.transpose(1, 0)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return np.stack(b, axis=-1)
