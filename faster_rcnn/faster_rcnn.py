import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.timer import Timer
from utils.blob import im_list_to_blob
from fast_rcnn.nms_wrapper import nms
from rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes

import network
from network import Conv2d, FC
from roi_pooling.modules.roi_pool import RoIPool
from resnet import resnet101

def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class RPN(nn.Module):
    _feat_stride = [4,8,16,32,64]
    anchor_scales = [2, 4, 8, 16, 32]
    num_anchor_ratio = 3
    def __init__(self, size = 5, training= False):
        super(RPN, self).__init__()

        self.training = training
        self.size = size
        rpn_conv, score_conv, bbox_conv = self._make_layer(size)
        self.rpn_conv = nn.ModuleList(rpn_conv)
        self.score_conv = nn.ModuleList(score_conv)
        self.bbox_conv = nn.ModuleList(bbox_conv)
        self.relu = nn.ReLU(inplace=True) 
        # loss
        self.cross_entropy = None
        self.loss_box = None

    def _make_layer(self, p_size):
        rpn_conv = []
        score_conv = []
        bbox_conv = []
        for i in range(p_size):
            rpn_conv += [nn.Conv2d(in_channels=256, out_channels = 512, kernel_size=3, stride=1, padding=1, bias=False)]
            score_conv += [nn.Conv2d(in_channels=512, out_channels = self.num_anchor_ratio * 2, kernel_size=1, stride=1, padding=0, bias=False)]
            bbox_conv += [nn.Conv2d(in_channels=512, out_channels = self.num_anchor_ratio * 4, kernel_size=1, stride=1, padding=0,bias=False)]
        return rpn_conv, score_conv,bbox_conv 

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, P_features, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        assert self.size == len(P_features), "should be same"
        rpn_cls_vec = []
        rpn_cls_prob_vec = []
        rpn_cls_prob_reshape_vec = []
        rpn_bbox_vec = []
        rpn_bbox_vec_var = []
        for i in range(self.size):
            p_rpn_conv = self.rpn_conv[i](P_features[i])
            # rpn score
            p_rpn_score = self.score_conv[i](p_rpn_conv)
            p_rpn_score = self.relu(p_rpn_score)

            p_rpn_score_shape = p_rpn_score.size()
            p_rpn_score_reshape = p_rpn_score.view((p_rpn_score_shape[0], 2, -1, p_rpn_score_shape[3]))
            p_rpn_score_prob = F.softmax(p_rpn_score_reshape)
            p_rpn_score_prob_reshape = p_rpn_score_prob.view(p_rpn_score_prob.size()[0], self.num_anchor_ratio * 2, -1, p_rpn_score_prob.size()[3])

            # rpn boxes
            p_rpn_bbox = self.bbox_conv[i](p_rpn_conv)

            rpn_cls_vec.append(p_rpn_score.data.cpu().numpy())
            rpn_cls_prob_vec.append(p_rpn_score_reshape)
            rpn_cls_prob_reshape_vec.append(p_rpn_score_prob_reshape.data.cpu().numpy())
            rpn_bbox_vec.append(p_rpn_bbox.data.cpu().numpy())
            rpn_bbox_vec_var.append(p_rpn_bbox)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.proposal_layer(rpn_cls_prob_reshape_vec[:-1], rpn_bbox_vec[:-1], im_info,
                                   cfg_key, self._feat_stride[:-1], self.anchor_scales[:-1])
        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.anchor_target_layer(rpn_cls_vec, gt_boxes, gt_ishard, dontcare_areas,
                                                im_info, self._feat_stride, self.anchor_scales)
            self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_prob_vec, rpn_bbox_vec_var, rpn_data)
        return rois

    def build_loss(self, rpn_cls_score_reshape_vec, rpn_bbox_pred_vec, rpn_data):
        rpn_cross_entropy_total = 0
        rpn_loss_box_total = 0
        for i in range(len(rpn_cls_score_reshape_vec)):
            # classification loss
            rpn_cls_score = rpn_cls_score_reshape_vec[i].permute(0, 2, 3, 1).contiguous().view(-1, 2)
            rpn_label = rpn_data[0][i].view(-1)

            rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze()).cuda()
            if len(rpn_keep.size())==0:continue
            rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

            # box loss
            rpn_bbox_targets = rpn_data[1][i]
            rpn_bbox_inside_weights = rpn_data[2][i]
            rpn_bbox_outside_weights = rpn_data[3][i]
    
            rpn_bbox_pred = rpn_bbox_pred_vec[i]
            rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
            rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)

            rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False) / (fg_cnt + 1e-4)
            
            rpn_cross_entropy_total += rpn_cross_entropy
            rpn_loss_box_total += rpn_loss_box
        
        return rpn_cross_entropy_total, rpn_loss_box_total

    @staticmethod
    def reshape_layer(x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        # x = x.permute(0, 2, 3, 1)
        return x

    @staticmethod
    def proposal_layer(rpn_cls_prob_reshape_vec, rpn_bbox_pred_vec, im_info, cfg_key, _feat_stride, anchor_scales):
        x = proposal_layer_py(rpn_cls_prob_reshape_vec, rpn_bbox_pred_vec, im_info, cfg_key, _feat_stride, anchor_scales)
        x = network.np_to_variable(x, is_cuda=True)
        return x.view(-1, 5)

    @staticmethod
    def anchor_target_layer(rpn_cls_score_vec, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales):
        """
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
        rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
                        that are the regression objectives
        rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
        rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
        beacuse the numbers of bgs and fgs mays significiantly different
        """
        #rpn_cls_score = rpn_cls_score.data.cpu().numpy()
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            anchor_target_layer_py(rpn_cls_score_vec, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales)
        """
        rpn_labels = network.np_to_variable(rpn_labels, is_cuda=True, dtype=torch.LongTensor)
        rpn_bbox_targets = network.np_to_variable(rpn_bbox_targets, is_cuda=True)
        rpn_bbox_inside_weights = network.np_to_variable(rpn_bbox_inside_weights, is_cuda=True)
        rpn_bbox_outside_weights = network.np_to_variable(rpn_bbox_outside_weights, is_cuda=True)
        """
        rpn_labels = [network.np_to_variable(rpn_labels[i], is_cuda=True, dtype=torch.LongTensor) for i in range(len(rpn_labels))]
        rpn_bbox_targets = [network.np_to_variable(rpn_bbox_targets[i], is_cuda=True) for i in range(len(rpn_bbox_targets))]
        rpn_bbox_inside_weights = [network.np_to_variable(rpn_bbox_inside_weights[i], is_cuda=True) for i in range(len(rpn_bbox_inside_weights))]
        rpn_bbox_outside_weights = [network.np_to_variable(rpn_bbox_outside_weights[i], is_cuda=True) for i in range(len(rpn_bbox_outside_weights))]

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

class FPN(nn.Module):
    def __init__(self):  
        super(FPN, self).__init__()
        self.C5_conv = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, bias=False)
        self.C4_conv = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=False)
        self.C3_conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False)
        self.C2_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, bias=False)

        #self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear')
        self.maxpooling = nn.MaxPool2d(kernel_size=1, stride=2)

        self.P2_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.P3_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.P4_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        #self.P5_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, C_vector):
        C1, C2, C3, C4, C5 = C_vector[0], C_vector[1], C_vector[2], C_vector[3], C_vector[4]    

        _, _, c2_height, c2_width = C2.size()
        _, _, c3_height, c3_width = C3.size()
        _, _, c4_height, c4_width = C4.size()

        P5 = self.C5_conv(C5)
        #P5 = self.P5_conv(P5)
        P6 = self.maxpooling(P5)
        #print "self.upsample(P5).size():\t", self.upsample(P5).size()
        #print "self.C4_conv(C4).size():\t", self.C4_conv(C4).size()
        #P4 = self.upsample(P5) + self.C4_conv(C4)
        P4 = F.upsample(P5, size=(c4_height, c4_width), mode='bilinear') + self.C4_conv(C4)
        P4 = self.P4_conv(P4)

        #P3 = self.upsample(P4) + self.C3_conv(C3)
        P3 = F.upsample(P4, size=(c3_height, c3_width), mode='bilinear') + self.C3_conv(C3)
        P3 = self.P3_conv(P3)

        #P2 = self.upsample(P3) + self.C2_conv(C2)
        P2 = F.upsample(P3, size=(c2_height, c2_width), mode='bilinear') + self.C2_conv(C2)
        P2 = self.P2_conv(P2)
        
        return P2, P3, P4, P5, P6

class FasterRCNN(nn.Module):
    n_classes = 21
    classes = np.asarray(['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    SCALES = (800,)
    MAX_SIZE = 1000

    def __init__(self, classes=None, debug=False, training = False):
        super(FasterRCNN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)

        self.resnet = resnet101()
        self.fpn = FPN()
        self.training = training
        self.rpn_net = RPN(training=self.training) 
        self.roi_pool_vec = nn.ModuleList([RoIPool(7, 7, 1.0/i) for i in [4,8,16,32]])
        self.fc6 = nn.Linear(in_features=256*7*7, out_features=1024)
        self.fc7 = nn.Linear(in_features=1024, out_features=1024)
        self.relu = nn.ReLU(inplace=True)
        self.score_fc = nn.Linear(in_features=1024, out_features=self.n_classes)
        self.bbox_fc = nn.Linear(in_features=1024, out_features=self.n_classes * 4)
        # loss
        self.cross_entropy = None
        self.loss_box = None

        # for log
        self.debug = debug

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, im_data, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        im_data = Variable(torch.from_numpy(im_data).type(torch.FloatTensor))
        is_cuda = True
        if is_cuda: im_data = im_data.cuda()
        im_data = im_data.permute(0, 3, 1, 2)
        C1, C2, C3, C4, C5 = self.resnet(im_data)
        P2, P3, P4, P5, P6 = self.fpn([C1, C2, C3, C4, C5]) 

        rois = self.rpn_net([P2, P3, P4, P5, P6], im_info, gt_boxes, gt_ishard, dontcare_areas)
        if self.training:
            roi_data = self.proposal_target_layer(rois.data.cpu().numpy(), gt_boxes, self.n_classes)
            rois = roi_data[0]
        else:
            rois = self.as_rois_mrcnn(rois.data.cpu().numpy())

        # roi pool
        pooled_features_vec = []
        P_vec = [P2, P3, P4, P5]
        rois_vec = []
        for i in range(4):
            pooled_feature_tmp = self.roi_pool_vec[i](P_vec[i], rois[i])
            pooled_features_vec.append(pooled_feature_tmp) 

        pooled_features = torch.cat(pooled_features_vec, 0) 
        x = pooled_features.view(pooled_features.size()[0], -1)
        x = self.fc6(x)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc7(x)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)

        cls_score = self.score_fc(x)
        cls_prob = F.softmax(cls_score)
        bbox_pred = self.bbox_fc(x)

        if self.training:
            self.cross_entropy, self.loss_box = self.build_loss(cls_score, bbox_pred, roi_data)

        return cls_prob, bbox_pred, rois

    def build_loss(self, cls_score, bbox_pred, roi_data):
        label = roi_data[1].squeeze()
        bbox_targets = roi_data[2]
        bbox_inside_weights = roi_data[3]
        bbox_outside_weights = roi_data[4]
        # classification loss
        fg_cnt = torch.sum(label.data.ne(0))
        bg_cnt = label.data.numel() - fg_cnt

        # for log
        if self.debug:
            maxv, predict = cls_score.data.max(1)
            self.tp = torch.sum(predict[:fg_cnt].eq(label.data[:fg_cnt])) if fg_cnt > 0 else 0
            self.tf = torch.sum(predict[fg_cnt:].eq(label.data[fg_cnt:]))
            self.fg_cnt = fg_cnt
            self.bg_cnt = bg_cnt
        
        ce_weights = torch.ones(cls_score.size()[1])
        ce_weights[0] = float(fg_cnt) / bg_cnt
        ce_weights = ce_weights.cuda()
        cross_entropy = F.cross_entropy(cls_score, label, weight=ce_weights, ignore_index=-1)

        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)

        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-4)
        return cross_entropy, loss_box

    @staticmethod
    def proposal_target_layer(rpn_rois, gt_boxes, num_classes):
        """
        ----------
        rpn_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
        # gt_ishard: (G, 1) {0 | 1} 1 indicates hard
        dontcare_areas: (D, 4) [ x1, y1, x2, y2]
        num_classes
        ----------
        Returns
        ----------
        rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
        bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
        bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        """
        #rpn_rois = rpn_rois.data.cpu().numpy()
        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            proposal_target_layer_py(rpn_rois, gt_boxes, num_classes)
        # print labels.shape, bbox_targets.shape, bbox_inside_weights.shape
        rois = [network.np_to_variable(rois[i], is_cuda=True) for i in range(len(rois))]
        labels = network.np_to_variable(labels, is_cuda=True, dtype=torch.LongTensor)
        bbox_targets = network.np_to_variable(bbox_targets, is_cuda=True)
        bbox_inside_weights = network.np_to_variable(bbox_inside_weights, is_cuda=True)
        bbox_outside_weights = network.np_to_variable(bbox_outside_weights, is_cuda=True)
        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    @staticmethod
    def as_rois_mrcnn(rois):
        w = (rois[:,3]-rois[:,1])
        h = (rois[:,4]-rois[:,2])
        s = w * h
        k0 =4
        s[s<=0]=1e-6
        layer_indexs = np.floor(k0+np.log2(np.sqrt(s)/224))

        layer_indexs[layer_indexs<2]=2
        layer_indexs[layer_indexs>5]=5

        rois_all =[]
        total = 0
        for i in range(4):
            index = (layer_indexs == (i + 2))
            num_index = sum(index)
            total += num_index
            if num_index == 0:
                rois_ = np.zeros((1*4, 5), dtype=rois.dtype)
            else:
                rois_ = rois[index, :]
            rois_all.append(rois_)
        rois_final = [network.np_to_variable(rois_all[i], is_cuda=True) for i in range(len(rois_all))]
        return rois_final
        
    def interpret_faster_rcnn(self, cls_prob, bbox_pred, rois, im_info, im_shape, nms=True, clip=True, min_score=0.0):
        # find class
        scores, inds = cls_prob.data.max(1)
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()

        keep = np.where((inds > 0) & (scores >= min_score))
        scores, inds = scores[keep], inds[keep]

        # Apply bounding-box regression deltas
        keep = keep[0]
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)

        rois_tmp = [rois[i].data.cpu().numpy() for i in range(len(rois))]
        rois = np.concatenate(rois_tmp, axis=0) 
        boxes = rois[keep, 1:5] / im_info[0][2]
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        if clip:
            pred_boxes = clip_boxes(pred_boxes, im_shape)

        # nms
        if nms and pred_boxes.shape[0] > 0:
            pred_boxes, scores, inds = nms_detections(pred_boxes, scores, 0.3, inds=inds)

        return pred_boxes, scores, self.classes[inds]

    def detect(self, image, thr=0.3):
        im_data, im_scales = self.get_image_blob(image)
        print im_data.shape
        im_info = np.array(
            [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
            dtype=np.float32)

        cls_prob, bbox_pred, rois = self(im_data, im_info)
        pred_boxes, scores, classes = \
            self.interpret_faster_rcnn(cls_prob, bbox_pred, rois, im_info, image.shape, min_score=thr)
        return pred_boxes, scores, classes

    def get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
            im (ndarray): a color image in BGR order
        Returns:
            blob (ndarray): a data blob holding an image pyramid
            im_scale_factors (list): list of image scales (relative to im) used
                in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in self.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > self.MAX_SIZE:
                im_scale = float(self.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)
        return blob, np.array(im_scale_factors)

    def get_image_blob_noscale(self, im):
        im_orig = im.astype(np.float32, copy=True)
        #im_orig -= self.PIXEL_MEANS

        processed_ims = [im]
        im_scale_factors = [1.0]

        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

