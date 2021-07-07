#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np
import pyclipper
from numpy.linalg import norm
from tensorflow.python.keras import backend as K
eps = 1e-10


def polygon_to_rbox(xy):
    # center point plus width, height and orientation angle
    tl, tr, br, bl = xy
    # length of top and bottom edge
    dt, db = tr-tl, bl-br
    # center is mean of all 4 vetrices
    cx, cy = c = np.sum(xy, axis=0) / len(xy)
    # width is mean of top and bottom edge length
    w = (norm(dt) + norm(db)) / 2.
    # height is distance from center to top edge plus distance form center to bottom edge
    h = norm(np.cross(dt, tl-c))/(norm(dt)+eps) + norm(np.cross(db, br-c))/(norm(db)+eps)
    # angle is mean of top and bottom edge angle
    theta = (np.arctan2(dt[0], dt[1]) + np.arctan2(db[0], db[1])) / 2.
    return np.array([cx, cy, w, h, theta])


def rot_matrix(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st],[st, ct]])


def rbox_to_polygon(rbox):
    cx, cy, w, h, theta = rbox
    box = np.array([[-w,h],[w,h],[w,-h],[-w,-h]]) / 2.
    box = np.dot(box, rot_matrix(theta))
    box += rbox[:2]
    return box


def fscore(precision, recall, beta=1):
    """
    Computes the F score.
    
    The F score is the weighted harmonic mean of precision and recall.
    
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    
    With beta = 1, this is equivalent to a F-measure (F1 score). With beta < 1, 
    assigning correct classes becomes more important, and with beta > 1 the 
    metric is instead weighted towards penalizing incorrect class assignments.
    
    # Arguments
        precision: Scalar or array.
        recall: Array of same shape as precision.
        beta: Scalar.
    
    # Return
        score: Array of same shape as precision and recall.
    """
    eps = K.epsilon()
    p = precision
    r = recall
    bb = beta ** 2
    score = (1 + bb) * (p * r) / (bb * p + r + eps)
    return score


def evaluate_polygonal_results(ground_truth, detection_results, iou_thresh=0.5):
    """
    Evaluate polygonal text detection results and return TP, FP, FN.
    
    # Arguments
        ground_truth: List of ground truth polygonal with
            shape (objects, 4 x xy)
        detection_results: List of corresponding detection polygonal with 
            shape (objects, 4 x xy)
        image_size: Input size of detector network.
        iou_thresh: Minimum intersection over union required to associate
            a detected polygon box to a ground truth polygon box.
    
    # Returns
        TP: True Positive detections
        FP: False Positive detections
        FN: False Negative detections
    """
    
    # we do not sort by confidence here
    # all detections are of class text
    
    gt = ground_truth
    dt = detection_results
    
    TP = []
    FP = []
    FN_sum = 0
    
    num_groundtruth_boxes = 0 # has to be TP_sum + FN_sum
    num_detections = 0
    
    for i in range(len(gt)): # samples
        gt_polys = [np.reshape(gt[i][j,:], (-1, 2)) for j in range(len(gt[i]))]
        dt_polys = [np.reshape(dt[i][j,:], (-1, 2)) for j in range(len(dt[i]))]
        
        # prepare polygons, pyclipper, is much faster
        scale = 1e5
        gt_polys = [np.asarray(p*scale, dtype=np.int64) for p in gt_polys]
        dt_polys = [np.asarray(p*scale, dtype=np.int64) for p in dt_polys]

        num_dt = len(dt_polys)
        num_gt = len(gt_polys)
        
        num_groundtruth_boxes += num_gt
        num_detections += num_dt
        
        TP_img = np.zeros(num_dt)
        FP_img = np.zeros(num_dt)
        
        assignment = np.zeros(num_gt, dtype=np.bool)
        
        for k in range(len(dt[i])): # dt
            poly1 = dt_polys[k]
            gt_iou = []
            for j in range(len(gt[i])): # gt
                poly2 = gt_polys[j]
                
                # intersection over union, pyclipper
                pc = pyclipper.Pyclipper()
                pc.AddPath(poly1, pyclipper.PT_CLIP, True)
                pc.AddPath(poly2, pyclipper.PT_SUBJECT, True)
                I = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
                if len(I) > 0:
                    U = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
                    Ia = pyclipper.Area(I[0])
                    Ua = pyclipper.Area(U[0])
                    IoU = Ia / Ua
                else:
                    IoU = 0.0

                gt_iou.append(IoU)
            gt_iou = np.array(gt_iou)
            max_gt_idx = np.argmax(gt_iou)
            dt_idx = k
            
            if gt_iou[max_gt_idx] > iou_thresh:
                if not assignment[max_gt_idx]:
                    TP_img[dt_idx] = 1
                    assignment[max_gt_idx] = True
                    continue
            FP_img[dt_idx] = 1
        
        FN_img_sum = np.sum(np.logical_not(assignment))
            
        TP.append(TP_img)
        FP.append(FP_img)
        FN_sum += FN_img_sum
        
    TP = np.concatenate(TP)
    FP = np.concatenate(FP)
    
    TP_sum = np.sum(TP)
    FP_sum = np.sum(FP)
    
    return TP_sum, FP_sum, FN_sum

