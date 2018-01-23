# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
import numpy as np
import numpy.random as npr

from .generate_anchors import generate_anchors
from ..utils.cython_bbox import bbox_overlaps, bbox_intersections

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import bbox_transform

# <<<< obsolete


def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, feat_stride_vec=[4,8,16,32,64],
                        anchor_scales=[2, 4, 8, 16, 32]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    Parameters
    ----------
    rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
    gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
    gt_ishard: (G, 1), 1 or 0 indicates difficult or not
    dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
    im_info: a list of [image_height, image_width, scale_ratios]
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
    rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
                            that are the regression objectives
    rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
    rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
                            beacuse the numbers of bgs and fgs mays significiantly different
    """
    # allow boxes to sit over the edge by a small amount
    _allowed_border = 1000
    im_info = im_info[0]

    fpn_args = []
    fpn_anchors_fid = np.zeros(0).astype(int)
    fpn_anchors = np.zeros([0, 4])
    fpn_labels = np.zeros(0)
    fpn_inds_inside = []

    fpn_size = len(rpn_cls_score) #[P2,P3,P4,P5,P6]
    for i in range(fpn_size): 
        _anchors = generate_anchors(scales=np.array([anchor_scales[i]]))
        _num_anchors = _anchors.shape[0]
        _feat_stride = feat_stride_vec[i]
        # map of shape (..., H, W)
        #height, width = rpn_cls_score.shape[1:3]


        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert rpn_cls_score[i].shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        # pytorch (bs, c, h, w)
        height, width = rpn_cls_score[i].shape[2:4]

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * _feat_stride
        shift_y = np.arange(0, height) * _feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # in W H order
        # K is H x W
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = _num_anchors
        K = shifts.shape[0]
        all_anchors = (_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -_allowed_border) &
            (all_anchors[:, 1] >= -_allowed_border) &
            (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
            )[0]

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        # (A)
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)

        fpn_anchors_fid = np.hstack((fpn_anchors_fid, len(inds_inside)))
        fpn_anchors = np.vstack((fpn_anchors, anchors))
        fpn_labels = np.hstack((fpn_labels, labels))
        fpn_inds_inside.append(inds_inside)
        fpn_args.append([height, width, A, total_anchors])

    if len(gt_boxes) > 0:
        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt), shape is A x G
        overlaps = bbox_overlaps(
            np.ascontiguousarray(fpn_anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)  # (A)
        max_overlaps = overlaps[np.arange(len(fpn_anchors)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)  # G
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            fpn_labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        fpn_labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        fpn_labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            fpn_labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        fpn_labels[:]=0

    # preclude dontcare areas
    if dontcare_areas is not None and dontcare_areas.shape[0] > 0:
        # intersec shape is D x A
        intersecs = bbox_intersections(
            np.ascontiguousarray(dontcare_areas, dtype=np.float),  # D x 4
            np.ascontiguousarray(fpn_anchors, dtype=np.float)  # A x 4
        )
        intersecs_ = intersecs.sum(axis=0)  # A x 1
        fpn_labels[intersecs_ > cfg.TRAIN.DONTCARE_AREA_INTERSECTION_HI] = -1

    # preclude hard samples that are highly occlusioned, truncated or difficult to see
    if cfg.TRAIN.PRECLUDE_HARD_SAMPLES and gt_ishard is not None and gt_ishard.shape[0] > 0:
        assert gt_ishard.shape[0] == gt_boxes.shape[0]
        gt_ishard = gt_ishard.astype(int)
        gt_hardboxes = gt_boxes[gt_ishard == 1, :]
        if gt_hardboxes.shape[0] > 0:
            # H x A
            hard_overlaps = bbox_overlaps(
                np.ascontiguousarray(gt_hardboxes, dtype=np.float),  # H x 4
                np.ascontiguousarray(fpn_anchors, dtype=np.float))  # A x 4
            hard_max_overlaps = hard_overlaps.max(axis=0)  # (A)
            fpn_labels[hard_max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = -1
            max_intersec_label_inds = hard_overlaps.argmax(axis=1)  # H x 1
            fpn_labels[max_intersec_label_inds] = -1  #

    # subsample positive labels if we have too many
    #num_fg = fpn_labels.shape[0] if cfg.TRAIN.RPN_BATCHSIZE == -1 else int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    num_fg = fpn_labels.shape[0] if cfg.TRAIN.RPN_BATCHSIZE == -1 else int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(fpn_labels >= 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        fpn_labels[disable_inds] = -1

    # subsample negative labels if we have too many
    #num_bg =  fpn_labels.shape[0] if cfg.TRAIN.RPN_BATCHSIZE == -1 else cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    num_bg = fpn_labels.shape[0] if cfg.TRAIN.RPN_BATCHSIZE == -1 else cfg.TRAIN.RPN_BATCHSIZE - np.sum(fpn_labels >= 1)
    bg_inds = np.where(fpn_labels == 0)[0]
    fpn_anchors_fid = np.hstack((0, fpn_anchors_fid.cumsum()))
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        fpn_labels[disable_inds] = -1
        # print "was %s inds, disabling %s, now %s inds" % (
        # len(bg_inds), len(disable_inds), np.sum(labels == 0))

    fpn_bbox_targets = np.zeros((len(fpn_anchors), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        fpn_bbox_targets[fpn_labels >= 1, :] = bbox_transform(fpn_anchors[fpn_labels >= 1, :], gt_boxes[argmax_overlaps[fpn_labels >= 1], :4])
         # fpn_bbox_targets[:] = bbox_transform(fpn_anchors, gt_boxes[argmax_overlaps, :4])
    # fpn_bbox_targets = (fpn_bbox_targets - np.array(cfg.TRAIN.BBOX_MEANS)) / np.array(cfg.TRAIN.BBOX_STDS)
    fpn_bbox_weights = np.zeros((len(fpn_anchors), 4), dtype=np.float32)

    fpn_bbox_weights[fpn_labels >= 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
    fpn_bbox_outside_weights = np.zeros((len(fpn_anchors), 4), dtype=np.float32)

    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
        # num_examples = np.sum(labels >= 0) + 1
        # positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        # negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        positive_weights = np.ones((1, 4))
        negative_weights = np.zeros((1, 4))
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            (np.sum(fpn_labels == 1)) + 1)
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            (np.sum(fpn_labels == 0)) + 1)
    fpn_bbox_outside_weights[fpn_labels == 1, :] = positive_weights
    fpn_bbox_outside_weights[fpn_labels == 0, :] = negative_weights

    label_list = []
    bbox_target_list = []
    bbox_weight_list = []
    bbox_outside_weight_list = []
    for feat_id in range(len(feat_stride_vec)):
        height, width, A, total_anchors = fpn_args[feat_id]
        # map up to original set of anchors
        labels = _unmap(fpn_labels[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=-1)
        bbox_targets = _unmap(fpn_bbox_targets[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=0)
        bbox_weights = _unmap(fpn_bbox_weights[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=0)
        bbox_outside_weights = _unmap(fpn_bbox_outside_weights[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=0)
        
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        rpn_labels = labels.reshape((1, 1, A * height, width)).transpose(0, 2, 3, 1)

        bbox_targets = bbox_targets.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2) 
       
        bbox_weights = bbox_weights.reshape((1, height, width, A * 4)).transpose((0, 3, 1, 2))   
    
        bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4)).transpose((0, 3, 1, 2))

        label_list.append(rpn_labels)
        bbox_target_list.append(bbox_targets)
        bbox_weight_list.append(bbox_weights)
        bbox_outside_weight_list.append(bbox_outside_weights)

    return label_list, bbox_target_list, bbox_weight_list, bbox_outside_weight_list
    

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
