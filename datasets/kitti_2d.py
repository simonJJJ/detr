"""
Kitti 2D Detection datasets.
"""
from pathlib import Path

import torch
import torchvision
import os
import numpy as np
from PIL import Image
import time

import datasets.transforms as T


class KITTI2D(torchvision.datasets.VisionDataset):
    """
    Args:
        image_folder (string):
        ann_folder (string):
        index_file (string):
        transform (callable, optional):
        target_transform (callable, optional):
        transforms (callable, optional):
    """

    def __init__(self, image_folder,
                 ann_folder,
                 index_file,
                 mode,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(KITTI2D, self).__init__(image_folder, transforms=None, transform=None, target_transform=None)
        self._transforms = transforms
        self.ann_folder = ann_folder
        self.class_names = ["Car", "Pedestrian", "Cyclist"]
        self.sample_ids = self._read_imageset_file(index_file)
        self.mode = mode
        self.anns = []
        self.load_anns()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        """
        while True:
            sample_id = self.sample_ids[index]
            img = Image.open(os.path.join(self.root, '%06d.png' % sample_id)).convert('RGB')
            img, target = self.prepare(img, index)
            if target is None and self.mode == "train":
                index = self._rand_another(index)
                continue
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            return img, target

    # https://github.com/skyhehe123/SA-SSD/blob/master/mmdet/datasets/kitti.py#L123
    def _rand_another(self, index):
        pool = np.where(np.array(self.sample_ids) != self.sample_ids[index])[0]
        return np.random.choice(pool)

    @staticmethod
    def _read_imageset_file(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        return [int(line) for line in lines]

    def load_anns(self):
        print('loading kitti annotations into memory...')
        tic = time.time()
        for idx in self.sample_ids:
            data_list = [line.rstrip().split(" ") for line in open(os.path.join(self.ann_folder, '%06d.txt' % idx))]
            for data in data_list:
                data[1:] = list(map(float, data[1:]))
            self.anns.append(data_list)
        print('Done (t={:0.2f}s'.format(time.time() - tic))

    def prepare(self, image, idx):
        w, h = image.size

        data_list = self.anns[idx]
        boxes = [data[4:8] for data in data_list if data[0] not in ["DontCare"]]
        classes = [data[0] for data in data_list if data[0] not in ["DontCare"]]
        assert len(classes) == len(boxes)

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = ['Car' if n == 'Van' else n for n in classes]
        classes = np.array(classes)
        selected = [i for i in range(len(classes)) if classes[i] in self.class_names]
        boxes = boxes[selected, :]
        classes = classes[selected]
        classes = torch.tensor([self.class_names.index(n) + 1 for n in classes], dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        assert len(classes) == len(boxes)
        if len(classes) == len(boxes) == 0:
            return image, None

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


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


def build(image_set, args):
    root = Path(args.kitti_path)
    assert root.exists(), f'provided KITTI path {root} does not exit'
    PATHS = {
        "train": (root / "training" / "image_2", root / "training" / "label_2", root / "ImageSets" / 'train.txt'),
        "val": (root / "training" / "image_2", root / "training" / "label_2", root / "ImageSets" / 'val.txt')
    }

    img_folder, ann_folder, index_file = PATHS[image_set]
    dataset = KITTI2D(img_folder, ann_folder, index_file, image_set, transforms=make_kitti_transforms(image_set))
    return dataset


if __name__ == "__main__":
    # unit test
    import argparse
    import sys
    sys.path.append(os.path.dirname(__file__))
    parser = argparse.ArgumentParser('KITTI Dataset Unit Test')
    parser.add_argument('--kitti_path', type=str)
    args = parser.parse_args()
    dataset_train = build('train', args)
    dataset_val = build('val', args)
    for i, data in enumerate(dataset_train):
        print("img shape: ", data[0].size(), "target dict: ", data[1])
        if i >= 100:
            break
    for i, data in enumerate(dataset_val):
        print("img shape: ", data[0].size(), "target dict: ", data[1])
        if i >= 100:
            break
