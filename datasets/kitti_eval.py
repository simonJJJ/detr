"""
KITTI evaluator that works in distributed mode.
"""
import os
import io as sysio
import torch
import numpy as np
import numba

from util.misc import all_gather
from util.box_ops import box_cxcywh_to_xyxy


class KittiEvaluator(object):
    CLASS_TO_NAME = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'car',
        6: 'tractor',
        7: 'trailer',
    }
    OVERLAP_0_7 = np.array([[0.7, 0.5, 0.5, 0.7,
                             0.5, 0.7, 0.7, 0.7], [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7]])
    OVERLAP_0_5 = np.array([[0.7, 0.5, 0.5, 0.7,
                             0.5, 0.5, 0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5]])

    def __init__(self, current_classes):
        self.class_to_name = type(self).CLASS_TO_NAME
        self.name_to_class = {v: n for n, v in self.class_to_name.items()}
        if not isinstance(current_classes, (list, tuple)):
            current_classes = [current_classes]
        current_classes_int = []
        for curcls in current_classes:
            if isinstance(curcls, str):
                current_classes_int.append(self.name_to_class[curcls])
            else:
                current_classes_int.append(curcls)
        self.current_classes = current_classes_int
        min_overlaps = np.stack([type(self).OVERLAP_0_7, type(self).OVERLAP_0_5], axis=0)
        self.min_overlaps = min_overlaps[:, :, self.current_classes]

        self.img_ids = []
        self.dt_annos = []
        self.gt_annos = []

    def update(self, predictions, targets):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        for pred in predictions.values():
            for key, value in pred.items():
                pred[key] = value.cpu().numpy()
            self.dt_annos.append(pred)
        for target in targets:
            orig_target_size = target['orig_size']
            img_h, img_w = torch.split(orig_target_size, 1, dim=0)
            target['boxes'] = box_cxcywh_to_xyxy(target['boxes']) * torch.cat([img_w, img_h, img_w, img_h], dim=0)
            for key, value in target.items():
                target[key] = value.cpu().numpy()
            self.gt_annos.append(target)

    def synchronize_between_processes(self):
        all_img_ids = all_gather(self.img_ids)
        all_dt_annos = all_gather(self.dt_annos)
        all_gt_annos = all_gather(self.gt_annos)

        merge_img_ids = []
        for p in all_img_ids:
            merge_img_ids.extend(p)

        merged_dt_annos = []
        for p in all_dt_annos:
            merged_dt_annos.extend(p)

        merged_gt_annos = []
        for p in all_gt_annos:
            merged_gt_annos.extend(p)

        merged_img_ids = np.array(merge_img_ids)
        merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
        merged_dt_annos = np.array(merged_dt_annos)[idx]
        merged_gt_annos = np.array(merged_gt_annos)[idx]
        self.img_ids = list(merged_img_ids)
        self.dt_annos = list(merged_dt_annos)
        self.gt_annos = list(merged_gt_annos)

    def accumulate(self):
        self.mAPbbox = self.eval_bbox(self.gt_annos, self.dt_annos, self.current_classes, self.min_overlaps)

    def summarize(self):
        result = ''
        for j, curcls in enumerate(self.current_classes):
            # mAP threshold array: [num_minoverlap, metric, class]
            # mAP result: [num_class, num_diff, num_minoverlap]
            for i in range(self.min_overlaps.shape[0]):
                result += print_str(
                    (f"{self.class_to_name[curcls]} "
                     "AP@{:.2f}, {:.2f}, {:.2f}:".format(*self.min_overlaps[i, :, j])))
                result += print_str((f"bbox AP:{self.mAPbbox[j, 0, i]:.2f}, "
                                     f"{self.mAPbbox[j, 1, i]:.2f}, "
                                     f"{self.mAPbbox[j, 2, i]:.2f}"))
        print(result)

    def eval_bbox(self, gt_annos, dt_annos, current_classes, min_overlaps, difficultys=[0, 1, 2]):
        ret = self.eval_class(gt_annos, dt_annos, current_classes, difficultys, 0, min_overlaps)
        mAP_bbox = get_mAP(ret["precision"])
        return mAP_bbox

    def eval_class(self, gt_annos, dt_annos, current_classes, difficultys, metric, min_overlaps, num_parts=50):
        """Kitti eval."""
        assert len(gt_annos) == len(dt_annos)
        num_examples = len(gt_annos)
        split_parts = get_split_parts(num_examples, num_parts)

        rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
        overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
        N_SAMPLE_PTS = 41
        num_minoverlap = len(min_overlaps)
        num_class = len(current_classes)
        num_difficulty = len(difficultys)
        precision = np.zeros(
            [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
        recall = np.zeros(
            [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
        for m, current_class in enumerate(current_classes):
            for l, difficulty in enumerate(difficultys):
                rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
                (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
                 dontcares, total_dc_num, total_num_valid_gt) = rets
                for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                    thresholdss = []
                    for i in range(len(gt_annos)):
                        rets = compute_statistics_jit(
                            overlaps[i],
                            gt_datas_list[i],
                            dt_datas_list[i],
                            ignored_gts[i],
                            ignored_dets[i],
                            dontcares[i],
                            metric,
                            min_overlap=min_overlap,
                            thresh=0.0,
                            compute_fp=False)
                        tp, fp, fn, similarity, thresholds = rets
                        thresholdss += thresholds.tolist()
                    thresholdss = np.array(thresholdss)
                    thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                    thresholds = np.array(thresholds)
                    pr = np.zeros([len(thresholds), 4])
                    idx = 0
                    for j, num_part in enumerate(split_parts):
                        gt_datas_part = np.concatenate(
                            gt_datas_list[idx:idx + num_part], 0)
                        dt_datas_part = np.concatenate(
                            dt_datas_list[idx:idx + num_part], 0)
                        dc_datas_part = np.concatenate(
                            dontcares[idx:idx + num_part], 0)
                        ignored_dets_part = np.concatenate(
                            ignored_dets[idx:idx + num_part], 0)
                        ignored_gts_part = np.concatenate(
                            ignored_gts[idx:idx + num_part], 0)
                        fused_compute_statistics(
                            parted_overlaps[j],
                            pr,
                            total_gt_num[idx:idx + num_part],
                            total_dt_num[idx:idx + num_part],
                            total_dc_num[idx:idx + num_part],
                            gt_datas_part,
                            dt_datas_part,
                            dc_datas_part,
                            ignored_gts_part,
                            ignored_dets_part,
                            metric,
                            min_overlap=min_overlap,
                            thresholds=thresholds)
                        idx += num_part
                    for i in range(len(thresholds)):
                        recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                        precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    for i in range(len(thresholds)):
                        precision[m, l, k, i] = np.max(
                            precision[m, l, k, i:], axis=-1)
                        recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)

        ret_dict = {
            "recall": recall,
            "precision": precision,
        }
        return ret_dict


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis.
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["labels"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["labels"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["boxes"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["boxes"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_samples_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        thresholds.append(score)
        current_recall += 1 / (num_samples_pts - 1.0)
    return thresholds


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == -1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_bboxes = dt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            # only a tp add a threshold.
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff

    return tp, fp, fn, similarity, thresholds[:thresh_idx]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'car', 'tractor', 'trailer']
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["labels"])
    num_dt = len(dt_anno["labels"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["boxes"][i]
        gt_name = CLASS_NAMES[gt_anno["labels"][i] - 1]
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])):
            # if gt_anno["difficulty"][i] > difficulty or gt_anno["difficulty"][i] == -1:
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in range(num_gt):
        if CLASS_NAMES[gt_anno["labels"][i] - 1] == "DontCare":
            dc_bboxes.append(gt_anno["boxes"][i])
    for i in range(num_dt):
        if (CLASS_NAMES[dt_anno["labels"][i] - 1].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["boxes"][i, 3] - dt_anno["boxes"][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = gt_annos[i]["boxes"]
        dt_datas = np.concatenate([
            dt_annos[i]["boxes"], dt_annos[i]["scores"][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)
