"""
This code is modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/kitti2d_dataset.py
"""
import numpy as np
import os.path as osp
import pickle
from torch.utils.data import Dataset

import datasets.transforms as T


class Kitti2DDataset(Dataset):
    r"""KITTI 2D Dataset.

    This class serves as the API for experiments on the `KITTI Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    CLASSES = ('car', 'pedestrain', 'cyclist')
    """
    Annotation format:
    [
        {
            'image': {
                'image_idx': 0,
                'image_path': 'training/image_2/000000.png',
                'image_shape': array([ 370, 1224], dtype=int32)
            },
            'point_cloud': {
                 'num_features': 4,
                 'velodyne_path': 'training/velodyne/000000.bin'
             },
             'calib': {
                 'P0': <np.ndarray> (4, 4),
                 'P1': <np.ndarray> (4, 4),
                 'P2': <np.ndarray> (4, 4),
                 'P3': <np.ndarray> (4, 4),
                 'R0_rect':4x4 np.array,
                 'Tr_velo_to_cam': 4x4 np.array,
                 'Tr_imu_to_velo': 4x4 np.array
             },
             'annos': {
                 'name': <np.ndarray> (n),
                 'truncated': <np.ndarray> (n),
                 'occluded': <np.ndarray> (n),
                 'alpha': <np.ndarray> (n),
                 'bbox': <np.ndarray> (n, 4),
                 'dimensions': <np.ndarray> (n, 3),
                 'location': <np.ndarray> (n, 3),
                 'rotation_y': <np.ndarray> (n),
                 'score': <np.ndarray> (n),
                 'index': array([0], dtype=int32),
                 'group_ids': array([0], dtype=int32),
                 'difficulty': array([0], dtype=int32),
                 'num_points_in_gt': <np.ndarray> (n),
             }
        }
    ]
    """
    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)

        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            self._set_group_flag()

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        self.data_infos = pickle.load(ann_file)
        self.cat2label = {
            cat_name: i
            for i, cat_name in enumerate(self.CLASSES)
        }
        return self.data_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images without ground truths."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if len(img_info['annos']['name']) > 0:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - bboxes (np.ndarray): Ground truth bboxes.
                - labels (np.ndarray): Labels of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        annos = info['annos']
        gt_names = annos['name']
        gt_bboxes = annos['bbox']
        difficulty = annos['difficulty']

        # remove classes that is not needed
        selected = self.keep_arrays_by_name(gt_names, self.CLASSES)
        gt_bboxes = gt_bboxes[selected]
        gt_names = gt_names[selected]
        difficulty = difficulty[selected]
        gt_labels = np.array([self.cat2label[n] for n in gt_names])
        anns_results = dict(
            bboxes=gt_bboxes.astype(np.float32),
            labels=gt_labels,
        )
        return anns_results

    def prepare_train_img(self, idx):
        """Training image preparation.

        Args:
            index (int): Index for accessing the target image data.

        Returns:
            dict: Training image data dict after preprocessing
                corresponding to the index.
        """
        img_raw_info = self.data_infos[idx]['image']
        img_info = dict(filename=img_raw_info['image_path'])
        ann_info = self.get_ann_info(idx)
        if len(ann_info['bboxes']) == 0:
            return None
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return results

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes')

        return class_names

    def keep_arrays_by_name(self, gt_names, used_classes):
        """Keep useful ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be keeped.
        """
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds


def list_from_file(filename, prefix='', offset=0, max_num=0):
    """copied from mmcv"""
    cnt = 0
    item_list = []
    with open(filename, 'r') as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if max_num > 0 and cnt >= max_num:
                break
            item_list.append(prefix + line.rstrip('\n'))
            cnt += 1
    return item_list


def make_kitti_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            normalize,
        ])

    if image_set == 'val':
        return normalize

    raise ValueError(f'unknwon {image_set}')


def build(ann_file):
    dataset = Kitti2DDataset(ann_file, pipeline)
    return dataset