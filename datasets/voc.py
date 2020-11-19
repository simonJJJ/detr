"""
Pascal VOC detection dataset.
"""
import torch
import torchvision
import os
import time
import io
import xml.etree.ElementTree as ET
import datasets.transforms as T
import util.misc as utils

from pathlib import Path
from PIL import Image


class PascalVOC(torchvision.datasets.VisionDataset):

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
                 transforms=None,
                 use_difficult=False):
        super(PascalVOC, self).__init__(image_folder, transforms=None, transform=None, target_transform=None)
        self._transforms = transforms
        self.ann_folder = ann_folder
        self.class_names = type(self).CLASSES
        self.sample_ids = self._read_imageset_file(index_file)
        self.mode = mode
        self.keep_difficult = use_difficult
        self.anns = []
        self.load_anns()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        """
        sample_id = self.sample_ids[index]
        img = self.BytePILRead(os.path.join(self.root, "%s.jpg") % sample_id)
        target = self.anns[index]
        target = ET.parse(target).getroot()
        target = self._preprocess_annotation(target)
        target["image_id"] = torch.tensor(int(sample_id))
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    @staticmethod
    def _read_imageset_file(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    def load_anns(self):
        print('loading pascal voc annotations into memory...')
        tic = time.time()
        for image_id in self.sample_ids:
            label_path = os.path.join(
                self.ann_folder, "%s.xml"
            ) % image_id
            self.anns.append(label_path)
        print('Done (t={:0.2f}s'.format(time.time() - tic))

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_names.index(name))
            difficult_boxes.append(difficult)

        size = target.find("size")
        h, w = tuple(map(int, (size.find("height").text, size.find("width").text)))
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(gt_classes)
        difficult_boxes = torch.tensor(difficult_boxes)

        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        difficult_boxes = difficult_boxes[keep]
        assert len(labels) == len(boxes)

        res = {
            "boxes": boxes,
            "labels": labels,
            "difficult": difficult_boxes,
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
        }
        return res

    def BytePILRead(self, filepath):
        img_bytes = self.HardDistGet(filepath)
        buff = io.BytesIO(img_bytes)
        img = Image.open(buff)
        return img

    def HardDistGet(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf


def make_pascalvoc_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.Resize((512, 512)),
            normalize,
        ])
    elif image_set == 'val':
        return normalize

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    assert image_set in ["train", "val"]
    root = Path(args.voc_path)
    assert root.exists(), f'provided VOC path {root} dose not exit'
    PATHS = {
        "train": (root / "VOC0712" / "JPEGImages", root / "VOC0712" / "Annotations", root / "VOC0712" / "ImageSets" / "Main" / 'trainval.txt'),
        "val": (root / "VOCtest" / "JPEGImages", root / "VOCtest" / "Annotations", root / "VOCtest" / "ImageSets" / "Main" / 'test.txt'),
    }

    img_folder, ann_folder, index_file = PATHS[image_set]
    dataset = PascalVOC(img_folder, ann_folder, index_file, image_set, transforms=make_pascalvoc_transforms(image_set))
    return dataset
