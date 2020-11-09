"""
Pascal VOC evaluator that works in distributed mode.
"""
import torch
import numpy as np

from collections import defaultdict
from util.misc import all_gather
from util.box_ops import box_cxcywh_to_xyxy
from util.box_np_ops import box_iou_np


class VocEvaluator(object):

    CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(self):
        self.img_ids = []
        self.dt_annos = []
        self.gt_annos = []

    def update(self, predictions, targets):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.append(img_ids)
        for pred in predictions.values():
            for key, value in pred.items():
                pred[key] = value.cpu()
            self.dt_annos.append(pred)
        for target in targets:
            orig_target_size = target['orig_size']
            img_h, img_w = torch.split(orig_target_size, 1, dim=0)
            target['boxes'] = box_cxcywh_to_xyxy(target['boxes']) * torch.cat([img_w, img_h, img_w, img_h], dim=0)
            for key, value in target.items():
                target[key] = value.cpu()
            self.gt_annos.append(target)

    def synchronize_between_processes(self):
        all_img_ids = all_gather(self.img_ids)
        all_dt_annos = all_gather(self.dt_annos)
        all_gt_annos = all_gather(self.gt_annos)

        merged_img_ids = []
        for p in all_img_ids:
            merged_img_ids.extend(p)

        merged_dt_annos = []
        for p in all_dt_annos:
            merged_dt_annos.extend(p)

        merged_gt_annos = []
        for p in all_gt_annos:
            merged_gt_annos.extend(p)

        self.img_ids = list(merged_img_ids)
        self.dt_annos = list(merged_dt_annos)
        self.gt_annos = list(merged_gt_annos)

    def accumulate(self):
        prec, rec = self.calc_detection_voc_pred_rec(self.gt_annos, self.dt_annos)
        self.aps = self.calc_detection_voc_ap(prec, rec, use_07_metric=False)

    def summarize(self):
        print('Mean AP = {:.4f}'.format(np.mean(self.aps[1:])))
        print('~~~~~~~~')
        print('Results:')
        for i, ap in enumerate(self.aps[1:]):
            print('{:}, {:.3f}'.format(type(self).CLASSES[i + 1], ap))
        print('{:.3f}'.format(np.mean(self.aps[1:])))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')

    def calc_detection_voc_pred_rec(self, gt_annos, dt_annos, iou_thresh=0.5):
        """
        The code is based on the evaluation code used in PASCAL VOC Challenge.
        """
        n_pos = defaultdict(int)
        score = defaultdict(list)
        match = defaultdict(list)
        for gt_ann, dt_ann in zip(gt_annos, dt_annos):
            pred_bbox = dt_ann['boxes'].numpy()
            pred_label = dt_ann['labels'].numpy()
            pred_score = dt_ann['scores'].numpy()
            gt_bbox = gt_ann['boxes'].numpy()
            gt_label = gt_ann['labels'].numpy()
            gt_difficult = gt_ann['difficult'].numpy()

            for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
                pred_mask_l = pred_label == l
                pred_bbox_l = pred_bbox[pred_mask_l]
                pred_score_l = pred_score[pred_mask_l]
                # sort by score
                order = pred_score_l.argsort()[::-1]
                pred_bbox_l = pred_bbox_l[order]
                pred_score_l = pred_score_l[order]

                gt_mask_l = gt_label == l
                gt_bbox_l = gt_bbox[gt_mask_l]
                gt_difficult_l = gt_difficult[gt_mask_l]

                n_pos[l] += np.logical_not(gt_difficult_l).sum()
                score[l].extend(pred_score_l)

                if len(pred_bbox_l) == 0:
                    continue
                if len(gt_bbox_l) == 0:
                    match[l].extend((0,) * pred_bbox_l.shape[0])
                    continue

                # VOC evaluation follows integer typed bounding boxes.
                pred_bbox_l = pred_bbox_l.copy()
                pred_bbox_l[:, 2:] += 1
                gt_bbox_l = gt_bbox_l.copy()
                gt_bbox_l[:, 2:] += 1
                iou = box_iou_np(pred_bbox_l, gt_bbox_l)
                gt_index = iou.argmax(axis=1)
                # set -1 if there is no matching ground truth
                gt_index[iou.max(axis=1) < iou_thresh] = -1
                del iou

                selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
                for gt_idx in gt_index:
                    if gt_idx >= 0:
                        if gt_difficult_l[gt_idx]:
                            match[l].append(-1)
                        else:
                            if not selec[gt_idx]:
                                match[l].append(1)
                            else:
                                match[l].append(0)
                        selec[gt_idx] = True
                    else:
                        match[l].append(0)

        n_fg_class = max(n_pos.keys()) + 1
        prec = [None] * n_fg_class
        rec = [None] * n_fg_class

        for l in n_pos.keys():
            score_l = np.array(score[l])
            match_l = np.array(match[l], dtype=np.int8)

            order = score_l.argsort()[::-1]
            match_l = match_l[order]

            tp = np.cumsum(match_l == 1)
            fp = np.cumsum(match_l == 0)

            # If an element of fp + tp is 0,
            # the corresponding element of prec[l] is nan.
            prec[l] = tp / (fp + tp)
            # if n_pos[l] is 0, rec[l] is None.
            if n_pos[l] > 0:
                rec[l] = tp / n_pos[l]

        return prec, rec

    def calc_detection_voc_ap(self, prec, rec, use_07_metric=False):
        """Calculate average precisions based on evaluation code of PASCAL VOC.
        This function calculates average precisions
        from given precisions and recalls.
        The code is based on the evaluation code used in PASCAL VOC Challenge.
        Args:
            prec (list of numpy.array): A list of arrays.
                :obj:`prec[l]` indicates precision for class :math:`l`.
                If :obj:`prec[l]` is :obj:`None`, this function returns
                :obj:`numpy.nan` for class :math:`l`.
            rec (list of numpy.array): A list of arrays.
                :obj:`rec[l]` indicates recall for class :math:`l`.
                If :obj:`rec[l]` is :obj:`None`, this function returns
                :obj:`numpy.nan` for class :math:`l`.
            use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
                for calculating average precision. The default value is
                :obj:`False`.
        Returns:
            ~numpy.ndarray:
            This function returns an array of average precisions.
            The :math:`l`-th value corresponds to the average precision
            for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
            :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
        """

        n_fg_class = len(prec)
        ap = np.empty(n_fg_class)
        for l in range(n_fg_class):
            if prec[l] is None or rec[l] is None:
                ap[l] = np.nan
                continue

            if use_07_metric:
                # 11 point metric
                ap[l] = 0
                for t in np.arange(0.0, 1.1, 0.1):
                    if np.sum(rec[l] >= t) == 0:
                        p = 0
                    else:
                        p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                    ap[l] += p / 11
            else:
                # correct AP calculation
                # first append sentinel values at the end
                mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
                mrec = np.concatenate(([0], rec[l], [1]))

                mpre = np.maximum.accumulate(mpre[::-1])[::-1]

                # to calculate area under PR curve, look for points
                # where X axis (recall) changes value
                i = np.where(mrec[1:] != mrec[:-1])[0]

                # and sum (\Delta recall) * prec
                ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap
