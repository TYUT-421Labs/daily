# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np


class Ocean_(nn.Module):
    def __init__(self):
        super(Ocean_, self).__init__()
        self.features = None  # feature extract tools
        self.connect_model = None  # connect model 1*1 1*2 2*1
        self.align_head = None  # align head network
        self.zf = None  # initial
        self.criterion = nn.BCEWithLogitsLoss()  # criterion
        self.neck = None  # 1024 -> 256
        self.search_size = 255  # search size -> 255
        self.score_size = 25  # score size -> 25 -> 17
        self.batch = 32 if self.training else 1  # batch size 32 -> 16
        self.grids()  # unknown

    def feature_extractor(self, x, online=False):
        return self.features(x)

    def extract_for_online(self, x):
        xf = self.feature_extractor(x, online=True)
        return xf

    def connector(self, template_feature, search_feature):
        pred_score = self.connect_model(template_feature, search_feature)
        return pred_score

    def _cls_loss(self, pred, label, select):
        """
            single classification loss
        """
        if len(select.size()) == 0: return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)  # the same as tf version

    def _weighted_BCE(self, pred, label):
        """
            batch classification loss
        """
        pred = pred.view(-1)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze().cuda()
        neg = label.data.eq(0).nonzero().squeeze().cuda()

        loss_pos = self._cls_loss(pred, label, pos)
        loss_neg = self._cls_loss(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def _weighted_BCE_align(self, pred, label):
        """
        align batch classification loss
        """
        pred = pred.view(-1)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze().cuda()
        neg = label.data.eq(0).nonzero().squeeze().cuda()

        loss_pos = self._cls_loss(pred, label, pos)
        loss_neg = self._cls_loss(pred, label, neg)

        return loss_pos * 0.5 + loss_neg * 0.5

    def _IOULoss(self, pred, target, weight=None):
        """
            compute single IOU loss, regression loss
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

    def add_iouloss(self, bbox_pred, reg_target, reg_weight):
        """
        :param bbox_pred:
        :param reg_target:
        :param reg_weight:
        :param grid_x:  used to get real target bbox
        :param grid_y:  used to get real target bbox
        :return:
        """

        bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_target_flatten = reg_target.reshape(-1, 4)
        reg_weight_flatten = reg_weight.reshape(-1)
        pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1)

        bbox_pred_flatten = bbox_pred_flatten[pos_inds]
        reg_target_flatten = reg_target_flatten[pos_inds]

        loss = self._IOULoss(bbox_pred_flatten, reg_target_flatten)

        return loss

    # ---------------------
    # classification align
    # ---------------------
    def grids(self):
        """
            each element of feature map on input search image
            :return: H*W*2 (position for each element)
        """
        sz = self.score_size
        stride = 8

        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * stride + self.search_size // 2
        self.grid_to_search_y = y * stride + self.search_size // 2

        self.grid_to_search_x = torch.Tensor(self.grid_to_search_x).unsqueeze(0).unsqueeze(0).cuda()
        self.grid_to_search_y = torch.Tensor(self.grid_to_search_y).unsqueeze(0).unsqueeze(0).cuda()

        self.grid_to_search_x = self.grid_to_search_x.repeat(self.batch, 1, 1, 1)
        self.grid_to_search_y = self.grid_to_search_y.repeat(self.batch, 1, 1, 1)

    def pred_to_image(self, bbox_pred):
        self.grid_to_search_x = self.grid_to_search_x.to(bbox_pred.device)
        self.grid_to_search_y = self.grid_to_search_y.to(bbox_pred.device)

        pred_x1 = self.grid_to_search_x - bbox_pred[:, 0, ...].unsqueeze(1)  # 17*17
        pred_y1 = self.grid_to_search_y - bbox_pred[:, 1, ...].unsqueeze(1)  # 17*17
        pred_x2 = self.grid_to_search_x + bbox_pred[:, 2, ...].unsqueeze(1)  # 17*17
        pred_y2 = self.grid_to_search_y + bbox_pred[:, 3, ...].unsqueeze(1)  # 17*17

        pred = [pred_x1, pred_y1, pred_x2, pred_y2]

        pred = torch.cat(pred, dim=1)

        return pred

    def align_label(self, pred, target, weight):
        # calc predicted box iou (treat it as aligned label)

        pred = pred.permute(0, 2, 3, 1)  # [B, 25, 25, 4]
        pred_left = pred[..., 0]
        pred_top = pred[..., 1]
        pred_right = pred[..., 2]
        pred_bottom = pred[..., 3]

        target_left = target[..., 0]
        target_top = target[..., 1]
        target_right = target[..., 2]
        target_bottom = target[..., 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)

        ious = torch.abs(weight * ious)  # delete points out of object

        ious[ious < 0] = 0
        ious[ious >= 1] = 1

        return ious

    def offset(self, boxes, featmap_sizes):
        """
        align 用的 offset， 之后看align的时候看
        refers to Cascade RPN
        Params:
        box_list: [N, 4]   [x1, y1, x2, y2] # predicted bbox
        """

        def _shape_offset(boxes, stride):
            ks = 3
            dilation = 1
            pad = (ks - 1) // 2
            idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
            xx, yy = torch.meshgrid(idx, idx)
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)

            pad = (ks - 1) // 2
            idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
            yy, xx = torch.meshgrid(idx, idx)  # return order matters
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            w = (boxes[:, 2] - boxes[:, 0] + 1) / stride
            h = (boxes[:, 3] - boxes[:, 1] + 1) / stride
            w = w / (ks - 1) - dilation
            h = h / (ks - 1) - dilation
            offset_x = w[:, None] * xx  # (NA, ks**2)
            offset_y = h[:, None] * yy  # (NA, ks**2)
            return offset_x, offset_y

        def _ctr_offset(boxes, stride, featmap_size):
            feat_h, feat_w = featmap_size
            image_size = self.search_size

            assert len(boxes) == feat_h * feat_w

            x = (boxes[:, 0] + boxes[:, 2]) * 0.5
            y = (boxes[:, 1] + boxes[:, 3]) * 0.5

            # # compute centers on feature map
            # x = (x - (stride - 1) * 0.5) / stride
            # y = (y - (stride - 1) * 0.5) / stride

            # different here for Siamese
            # use center of image as coordinate origin
            x = (x - image_size * 0.5) / stride + feat_w // 2
            y = (y - image_size * 0.5) / stride + feat_h // 2

            # compute predefine centers
            # different here for Siamese
            xx = torch.arange(0, feat_w, device=boxes.device)
            yy = torch.arange(0, feat_h, device=boxes.device)
            yy, xx = torch.meshgrid(yy, xx)
            xx = xx.reshape(-1).type_as(x)
            yy = yy.reshape(-1).type_as(y)

            offset_x = x - xx  # (NA, )
            offset_y = y - yy  # (NA, )
            return offset_x, offset_y

        num_imgs = len(boxes)
        dtype = boxes[0].dtype
        device = boxes[0][0].device

        featmap_sizes = featmap_sizes[2:]

        offset_list = []
        for i in range(num_imgs):
            c_offset_x, c_offset_y = _ctr_offset(boxes[i], 8, featmap_sizes)
            s_offset_x, s_offset_y = _shape_offset(boxes[i], 8)

            # offset = ctr_offset + shape_offset
            offset_x = s_offset_x + c_offset_x[:, None]
            offset_y = s_offset_y + c_offset_y[:, None]

            # offset order (y0, x0, y1, x0, .., y9, x8, y9, x9)from torch.autograd import Variable
            offset = torch.stack([offset_y, offset_x], dim=-1)
            offset = offset.reshape(offset.size(0), -1).unsqueeze(0)  # [NA, 2*ks**2]
            offset_list.append(offset)

        offsets = torch.cat(offset_list, 0)
        return offsets

    def template(self, z):
        """
            for track init
        """
        _, self.zf = self.feature_extractor(z)

        if self.neck is not None:
            _, self.zf = self.neck(self.zf, crop=True)

        if self.align_head is not None:
            self.update_flag = True
        else:
            pass

    def track(self, x):
        """
            for track
        """
        _, xf = self.feature_extractor(x)

        if self.neck is not None:
            xf = self.neck(xf)

        if self.align_head is not None:
            if self.update_flag:
                self.batch = 1
                self.search_size = x.size(-1)
                self.score_size = (self.search_size - 127) // 8 + 1 + 8
                self.grids()
                self.update_flag = False

            bbox_pred, cls_pred, cls_feature, reg_feature = self.connect_model(xf, self.zf)
            bbox_pred_to_img = self.pred_to_image(bbox_pred)
            offsets = self.offset(bbox_pred_to_img.permute(0, 2, 3, 1).reshape(bbox_pred_to_img.size(0), -1, 4),
                                  bbox_pred.size())
            cls_align = self.align_head(reg_feature, offsets)

            return cls_pred, bbox_pred, cls_align
        else:
            bbox_pred, cls_pred, _, _ = self.connect_model(xf, self.zf)

            return cls_pred, bbox_pred

    def forward(self, template, search, label=None, reg_target=None, reg_weight=None):
        """
        :return: for train
        label：这是目标类别标签，通常是一个二进制值（0 或 1），表示目标是否存在于搜索区域中。对于目标检测和跟踪任务，该值用于计算分类损失。

        reg_target：     这是目标回归目标，表示边界框的真实坐标。在目标检测和跟踪任务中，这些值用于计算边界框预测的回归损失。
                        回归损失的计算方法因任务而异，但通常包括平滑 L1 损失、IoU 损失等。

        reg_weight：     这是一个权重因子，用于平衡不同样本之间的回归损失。
                        权重因子可用于处理类别不平衡问题，例如在目标检测任务中，某一类目标可能比其他类更为稀少。
                        通过为稀少类别分配更高的权重，可以增加该类别在损失函数中的重要性。
        """
        # _ 指的是前几个 stage 的输出结果，一个 list
        _, zf = self.feature_extractor(template)
        _, xf = self.feature_extractor(search)

        # 对通道进行下采样，同时进行裁减，干掉padding，可以改，直接删掉都可以
        if self.neck is not None:
            _, zf = self.neck(zf, crop=True)
            xf = self.neck(xf, crop=False)

        # depth-wise cross correlation --> tower --> box pred
        if self.align_head is not None:
            # align head 先不看，似乎是特征对齐用的
            bbox_pred, cls_pred, cls_feature, reg_feature = self.connect_model(xf, zf)

            bbox_pred_to_img = self.pred_to_image(bbox_pred)
            offsets = self.offset(bbox_pred_to_img.permute(0, 2, 3, 1).reshape(bbox_pred_to_img.size(0), -1, 4),
                                  bbox_pred.size())
            cls_align = self.align_head(reg_feature, offsets)

            # add iou loss
            reg_loss = self.add_iouloss(bbox_pred, reg_target, reg_weight)

            # add cls loss
            align_cls_label = self.align_label(bbox_pred, reg_target, reg_weight)
            cls_loss_ori = self._weighted_BCE(cls_pred, label)
            cls_loss = self.criterion(cls_align.squeeze(), align_cls_label)

            if torch.isnan(cls_loss):
                cls_loss = 0 * cls_loss_ori

            return cls_loss_ori, cls_loss, reg_loss
        else:
            # 直接计算损失，connect model，出的结果是 bbox pred和 cls pred
            bbox_pred, cls_pred, _, _ = self.connect_model(xf, zf)
            # 计算 reg loss
            reg_loss = self.add_iouloss(bbox_pred, reg_target, reg_weight)
            # cls loss
            cls_loss = self._weighted_BCE(cls_pred, label)
            return cls_loss, None, reg_loss


"""
from lib.models.connect import box_tower, AdjustLayer, AlignHead, Corr_Up, MultiDiCorr, OceanCorr
from lib.models.backbones import ResNet50

if __name__ == '__main__':
    ocean = Ocean_()
    ocean.features = ResNet50(used_layers=[3], online=False)  # in param
    ocean.neck = AdjustLayer(in_channels=1024, out_channels=256)
    ocean.connect_model = box_tower(inchannels=256, outchannels=256, towernum=4)
    ocean.align_head = AlignHead(256, 256) if True else None
    print(ocean)
"""
